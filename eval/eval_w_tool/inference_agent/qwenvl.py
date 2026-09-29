"""
Inference engine for QwenVL without tool calling.
No system prompt, no multi-turn dialogue. Single-turn inference only.
"""
import os
import json
import time
import logging
from io import BytesIO
import base64 as b64
import asyncio
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from PIL import Image
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGER_LOGGING_LEVEL", "INFO"))
if not logger.handlers:
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logger.addHandler(handler)

try:
    from ..config import Config
    from ..datasets.base import DatasetBase
except ImportError:
    from config import Config
    from datasets.base import DatasetBase

from .image_utils import (
    encode_pil_image_to_base64,
    constrain_image_size,
    count_tokens,
)
from .prompts import (
    USER_PROMPT_TEMPLATE_QWEN,
    ANSWER_START_TOKEN,
    ANSWER_END_TOKEN,
)


class InferenceEngine:
    """Inference engine for QwenVL baseline (no tool, no system prompt, single-turn)."""

    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_url,
        )
        self.eval_model_name = config.eval_model_name
        if self.eval_model_name is None:
            logger.warning(f"eval_model_name is None, use model_name: {config.model_name}")
            self.eval_model_name = config.model_name
        self.sample_n = config.sample_n
        self.pixel_budget_per_image = config.pixel_budget_per_image

    async def infer_sample(self, sample: Dict[str, Any], dataset: DatasetBase, sample_n: Optional[int] = None) -> Dict[str, Any]:
        if sample_n is None:
            sample_n = self.sample_n if self.sample_n is not None else 1

        batch_start_time = time.time()

        result_dict = {
            "index": sample.get('index'),
            "question": sample.get('question'),
            "options": sample.get('options', None),
            'answer': sample.get('answer'),
            "pred_ans": [None] * sample_n,
            "pred_output": [None] * sample_n,
            "status": [None] * sample_n,
            "traj_info": [None] * sample_n,
        }

        async def single_inference_with_index(idx_n: int):
            output_text, print_messages, status, traj_info = await self._async_rollout_a_request(dataset, sample)
            return (output_text, print_messages, status, traj_info)

        tasks = [single_inference_with_index(idx_n) for idx_n in range(sample_n)]
        sample_results = await asyncio.gather(*tasks)

        for idx_n, (output_text, print_messages, status, traj_info) in enumerate(sample_results):
            result_dict["pred_ans"][idx_n] = output_text
            result_dict["pred_output"][idx_n] = print_messages
            result_dict["status"][idx_n] = status

            if traj_info:
                traj_info_cleaned = traj_info.copy()
                if 'original_pil_img' in traj_info_cleaned:
                    del traj_info_cleaned['original_pil_img']
                result_dict["traj_info"][idx_n] = traj_info_cleaned
            else:
                result_dict["traj_info"][idx_n] = traj_info

        batch_end_time = time.time()
        batch_total_time = batch_end_time - batch_start_time

        result_dict["batch_perf_stats"] = {
            "sample_n": sample_n,
            "batch_total_time": batch_total_time,
            "avg_time_per_sample": batch_total_time / sample_n if sample_n > 0 else 0.0,
            "individual_times": [ti['perf_stats']['total_time'] if ti and 'perf_stats' in ti else 0.0
                                 for ti in result_dict["traj_info"]],
            "max_individual_time": max([ti['perf_stats']['total_time'] if ti and 'perf_stats' in ti else 0.0
                                        for ti in result_dict["traj_info"]], default=0.0),
            "min_individual_time": min([ti['perf_stats']['total_time'] if ti and 'perf_stats' in ti else float('inf')
                                        for ti in result_dict["traj_info"]], default=0.0),
            "parallelization_speedup": sum([ti['perf_stats']['total_time'] if ti and 'perf_stats' in ti else 0.0
                                            for ti in result_dict["traj_info"]]) / batch_total_time if batch_total_time > 0 else 1.0,
        }

        for k, v in sample.items():
            if k not in result_dict:
                result_dict[k] = v

        return result_dict

    async def infer_dataset(
        self,
        dataset: DatasetBase,
        data_slice: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        num_workers: Optional[int] = None,
        save_step: int = 1,
    ) -> List[Dict[str, Any]]:
        num_workers = num_workers or self.config.num_workers
        num_samples = len(data_slice)

        logger.info(f"QwenVL InferenceEngine (no tool, no system prompt)! {self.pixel_budget_per_image=}")
        logger.info(f"Processing {num_samples} samples")
        logger.info(f"Using {num_workers} concurrent workers")

        semaphore = asyncio.Semaphore(num_workers)

        async def process_with_index(idx: int, sample: Dict[str, Any]) -> tuple[int, Optional[Dict[str, Any]]]:
            async with semaphore:
                result = await self.infer_sample(sample, dataset, sample_n=self.sample_n)
                return (idx, result)

        batch_size = 10
        num_batches = (num_samples + batch_size - 1) // batch_size

        logger.info(f"Batch size: {batch_size}, Total batches: {num_batches}")

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            logger.info(f"Saving results to {output_path}...")

        all_results = []
        pending_results = []

        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data_slice[start_idx:end_idx]

            tasks = [process_with_index(start_idx + i, batch_data[i]) for i in range(len(batch_data))]
            batch_results_with_idx = await asyncio.gather(*tasks)

            batch_results = []
            for idx, result in batch_results_with_idx:
                if result is not None:
                    batch_results.append(result)
                    all_results.append(result)
                    pending_results.append(result)

            if output_path and batch_results and (batch_idx + 1) % save_step == 0:
                self._append_batch_results(output_path, pending_results)
                pending_results = []

        if output_path and pending_results:
            self._append_batch_results(output_path, pending_results)

        logger.info(f"✓ Inference complete. Processed {len(all_results)} samples.")

        if all_results:
            logger.info("\n" + "=" * 60)
            logger.info("AGGREGATED PERFORMANCE STATISTICS")
            logger.info("=" * 60)
            agg_stats = self.aggregate_perf_stats(all_results)

            logger.info(f"Total Samples:              {agg_stats['total_samples']}")
            logger.info(f"Avg Image Load Time:        {agg_stats['avg_image_load_time']:.4f}s")
            logger.info(f"Avg Overview Process Time:  {agg_stats['avg_overview_process_time']:.4f}s")
            logger.info(f"Avg API Call Time:          {agg_stats['avg_api_call_time']:.4f}s")
            logger.info(f"Avg Total Time per Sample:  {agg_stats['avg_total_time']:.4f}s")
            logger.info(f"Avg API Calls per Sample:   {agg_stats['avg_num_api_calls']:.2f}")
            logger.info(f"Total API Calls:            {agg_stats['total_api_calls']}")

            if agg_stats['total_batches'] > 0:
                logger.info(f"\n--- Batch Inference Statistics ---")
                logger.info(f"Total Batches:              {agg_stats['total_batches']}")
                logger.info(f"Avg Batch Size (sample_n):  {agg_stats['avg_batch_sample_n']:.2f}")
                logger.info(f"Avg Batch Total Time:       {agg_stats['avg_batch_total_time']:.4f}s")
                logger.info(f"Avg Parallelization Speedup:{agg_stats['avg_batch_speedup']:.2f}x")

            logger.info("=" * 60 + "\n")

        return all_results

    def _append_batch_results(self, output_path: str, batch_results: List[Dict[str, Any]]):
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error appending batch results: {e}", exc_info=True)

    @staticmethod
    def aggregate_perf_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_stats = {
            'image_load_times': [],
            'overview_process_times': [],
            'api_call_times': [],
            'total_times': [],
            'num_api_calls': [],
            'batch_total_times': [],
            'batch_sample_ns': [],
            'batch_speedups': [],
        }

        for result in results:
            if 'traj_info' not in result:
                continue

            traj_infos = result['traj_info']
            if not isinstance(traj_infos, list):
                traj_infos = [traj_infos]

            for traj_info in traj_infos:
                if traj_info is None or 'perf_stats' not in traj_info:
                    continue

                perf = traj_info['perf_stats']
                all_stats['image_load_times'].append(perf['image_load_time'])
                all_stats['overview_process_times'].append(perf['overview_process_time'])
                all_stats['api_call_times'].extend(perf['api_call_times'])
                all_stats['total_times'].append(perf['total_time'])
                all_stats['num_api_calls'].append(len(perf['api_call_times']))

            if 'batch_perf_stats' in result:
                batch_perf = result['batch_perf_stats']
                all_stats['batch_total_times'].append(batch_perf['batch_total_time'])
                all_stats['batch_sample_ns'].append(batch_perf['sample_n'])
                all_stats['batch_speedups'].append(batch_perf['parallelization_speedup'])

        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            'total_samples': len([t for r in results if 'traj_info' in r
                                  for t in (r['traj_info'] if isinstance(r['traj_info'], list) else [r['traj_info']])
                                  if t is not None]),
            'avg_image_load_time': safe_avg(all_stats['image_load_times']),
            'avg_overview_process_time': safe_avg(all_stats['overview_process_times']),
            'avg_api_call_time': safe_avg(all_stats['api_call_times']),
            'avg_total_time': safe_avg(all_stats['total_times']),
            'avg_num_api_calls': safe_avg(all_stats['num_api_calls']),
            'total_api_calls': len(all_stats['api_call_times']),
            'total_batches': len(all_stats['batch_total_times']),
            'avg_batch_total_time': safe_avg(all_stats['batch_total_times']),
            'avg_batch_sample_n': safe_avg(all_stats['batch_sample_ns']),
            'avg_batch_speedup': safe_avg(all_stats['batch_speedups']),
        }

    async def _async_rollout_a_request(self, dataset: DatasetBase, sample: Dict[str, Any]) -> tuple[str, list[dict], str, dict]:
        """
        Single-turn inference: no system prompt, no tool calling, no multi-turn.
        """
        image_path = sample.get('image_path', "")
        traj_info = {
            "original_img_path": image_path,
            "original_pil_img": None,
            "scale": (1.0, 1.0),
            "finish_reason": [],
            "perf_stats": {
                "image_load_time": 0.0,
                "overview_process_time": 0.0,
                "api_call_times": [],
                "total_time": 0.0,
                "overview_tokens": 0,
            }
        }

        total_start_time = time.time()

        orig_pil_img = None
        if isinstance(image_path, str) and image_path != "":
            if os.path.exists(image_path):
                img_load_start = time.time()
                orig_pil_img = Image.open(image_path)
                traj_info['perf_stats']['image_load_time'] = time.time() - img_load_start
                traj_info['original_pil_img'] = orig_pil_img
            else:
                raise ValueError(f"Image not found: {image_path}")
        else:
            raise ValueError(f"Image not found: {image_path}")

        overview_start = time.time()
        if self.pixel_budget_per_image is not None:
            overview_img, scale = constrain_image_size(orig_pil_img, max_pixels=self.pixel_budget_per_image)
        else:
            overview_img, scale = orig_pil_img, (1.0, 1.0)
        b64_ov_img = encode_pil_image_to_base64(overview_img)
        traj_info['scale'] = scale
        traj_info['perf_stats']['overview_process_time'] = time.time() - overview_start
        traj_info['perf_stats']['overview_tokens'] = count_tokens(overview_img)

        dataset_prompt = dataset.format_prompt(sample)
        user_prompt = USER_PROMPT_TEMPLATE_QWEN.format(dataset_prompt=dataset_prompt)

        # No system prompt — only a single user message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ov_img}"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        print_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<dummy_b64_ov_img>"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        response_message = ""
        status = 'success'

        try:
            params = {
                "model": self.eval_model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

            api_call_start = time.time()
            response = await self._fetch_response(params)
            api_call_time = time.time() - api_call_start
            traj_info['perf_stats']['api_call_times'].append(api_call_time)

            response_message = response.choices[0].message.content

            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            traj_info['finish_reason'].append(finish_reason)

            print_messages.append({"role": "assistant", "content": response_message})

        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.error(f"Request timeout for sample idx: {sample.get('index', 'unknown')}: {e}")
            else:
                logger.error(f"Error processing sample {sample.get('index', 'unknown')}: {e}", exc_info=True)
            status = 'error'

        if ANSWER_END_TOKEN in response_message and ANSWER_START_TOKEN in response_message:
            try:
                output_text = response_message.split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0].strip()
            except:
                output_text = response_message
        else:
            output_text = response_message

        traj_info['perf_stats']['total_time'] = time.time() - total_start_time

        return output_text, print_messages, status, traj_info

    @retry(
        stop=stop_after_attempt(1),
        wait=wait_exponential(min=2, max=8),
        reraise=True
    )
    async def _fetch_response(self, params: dict):
        response = await asyncio.wait_for(
            self.client.chat.completions.create(**params),
            timeout=120
        )
        return response
