"""
Inference engine for Mini-o3 style prompting and grounding calls.
"""
import os
import re
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm
from PIL import Image
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

try:
    # Try relative imports first (when used as a package)
    from ..config import Config
    from ..datasets.base import DatasetBase
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config import Config
    from datasets.base import DatasetBase

from .image_utils import (
    encode_pil_image_to_base64,
    constrain_image_size,
    count_tokens,
)
from .pixelreasoner import (
    ImageTreeNode,
    crop_with_fix_retina,
    cropped_image_normalized,
)
from .prompts import (
    INSTRUCTION_PROMPT_SYSTEM_MINIO3,
    USER_PROMPT_TEMPLATE_MINIO3,
    TOOL_CALL_CROP_MULTI_TRUN_PROMPT_MINIO3,
    ANSWER_START_TOKEN,
    ANSWER_END_TOKEN,
)


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGER_LOGGING_LEVEL", "INFO"))
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

GROUNDING_START_TOKEN = "<grounding>"
GROUNDING_END_TOKEN = "</grounding>"
MAX_TURN = 3  # Fixed by evaluation setting.


class InferenceEngine:
    """Main inference engine for Mini-o3 style VQA with grounding calls."""

    def __init__(self, config: Config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_url,
        )
        self.eval_model_name = config.eval_model_name
        if self.eval_model_name is None:
            logger.warning("eval_model_name is None, fallback to model_name: %s", config.model_name)
            self.eval_model_name = config.model_name
        self.sample_n = config.sample_n
        self.pixel_budget_per_image = config.pixel_budget_per_image

    async def infer_sample(self, sample: Dict[str, Any], dataset: DatasetBase, sample_n: Optional[int] = None) -> Dict[str, Any]:
        if sample_n is None:
            sample_n = self.sample_n if self.sample_n is not None else 1

        batch_start_time = time.time()
        result_dict = {
            "index": sample.get("index"),
            "question": sample.get("question"),
            "options": sample.get("options", None),
            "answer": sample.get("answer"),
            "pred_ans": [None] * sample_n,
            "pred_output": [None] * sample_n,
            "status": [None] * sample_n,
            "traj_info": [None] * sample_n,
        }

        async def single_inference_with_index(_: int):
            output_text, print_messages, status, traj_info = await self._async_rollout_a_request(dataset, sample)
            return output_text, print_messages, status, traj_info

        tasks = [single_inference_with_index(i) for i in range(sample_n)]
        sample_results = await asyncio.gather(*tasks)

        for idx_n, (output_text, print_messages, status, traj_info) in enumerate(sample_results):
            result_dict["pred_ans"][idx_n] = output_text
            result_dict["pred_output"][idx_n] = print_messages
            result_dict["status"][idx_n] = status

            if traj_info:
                traj_info_cleaned = traj_info.copy()
                if "original_pil_img" in traj_info_cleaned:
                    del traj_info_cleaned["original_pil_img"]
                result_dict["traj_info"][idx_n] = traj_info_cleaned
            else:
                result_dict["traj_info"][idx_n] = traj_info

        batch_total_time = time.time() - batch_start_time
        result_dict["batch_perf_stats"] = {
            "sample_n": sample_n,
            "batch_total_time": batch_total_time,
            "avg_time_per_sample": batch_total_time / sample_n if sample_n > 0 else 0.0,
            "individual_times": [
                traj_info["perf_stats"]["total_time"] if traj_info and "perf_stats" in traj_info else 0.0
                for traj_info in result_dict["traj_info"]
            ],
            "max_individual_time": max([
                traj_info["perf_stats"]["total_time"] if traj_info and "perf_stats" in traj_info else 0.0
                for traj_info in result_dict["traj_info"]
            ], default=0.0),
            "min_individual_time": min([
                traj_info["perf_stats"]["total_time"] if traj_info and "perf_stats" in traj_info else float("inf")
                for traj_info in result_dict["traj_info"]
            ], default=0.0),
            "parallelization_speedup": sum([
                traj_info["perf_stats"]["total_time"] if traj_info and "perf_stats" in traj_info else 0.0
                for traj_info in result_dict["traj_info"]
            ]) / batch_total_time if batch_total_time > 0 else 1.0,
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

        logger.info("Mini-o3 InferenceEngine")
        logger.info("Processing %d samples", num_samples)
        logger.info("Using %d concurrent workers", num_workers)

        semaphore = asyncio.Semaphore(num_workers)

        async def process_with_index(idx: int, sample: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
            async with semaphore:
                result = await self.infer_sample(sample, dataset, sample_n=self.sample_n)
                return idx, result

        batch_size = 5
        num_batches = (num_samples + batch_size - 1) // batch_size

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            logger.info("Saving results to %s...", output_path)

        all_results = []
        pending_results = []
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data_slice[start_idx:end_idx]

            tasks = [
                process_with_index(start_idx + idx_in_batch, batch_data[idx_in_batch])
                for idx_in_batch in range(len(batch_data))
            ]
            batch_results_with_idx = await asyncio.gather(*tasks)

            for _, result in batch_results_with_idx:
                if result is not None:
                    all_results.append(result)
                    pending_results.append(result)

            if output_path and pending_results and (batch_idx + 1) % save_step == 0:
                self._append_batch_results(output_path, pending_results)
                pending_results = []

        if output_path and pending_results:
            self._append_batch_results(output_path, pending_results)

        logger.info("Inference complete. Processed %d samples.", len(all_results))
        return all_results

    def _append_batch_results(self, output_path: str, batch_results: List[Dict[str, Any]]):
        try:
            with open(output_path, "a", encoding="utf-8") as f:
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Error appending batch results: %s", e, exc_info=True)

    async def _async_rollout_a_request(self, dataset: DatasetBase, sample: Dict[str, Any]) -> Tuple[str, List[dict], str, Dict[str, Any]]:
        image_path = sample.get("image_path", "")
        traj_info = {
            "original_img_path": image_path,
            "original_pil_img": None,
            "image_tree": {},
            "bboxes_list": [],
            "sources_list": [],
            "finish_reason": [],
            "perf_stats": {
                "image_load_time": 0.0,
                "overview_process_time": 0.0,
                "api_call_times": [],
                "tool_call_times": [],
                "total_time": 0.0,
                "overview_tokens": 0,
                "crop_tokens": [],
            },
        }

        total_start_time = time.time()
        status = "success"
        response_message = ""

        if not (isinstance(image_path, str) and image_path and os.path.exists(image_path)):
            raise ValueError(f"Image not found: {image_path}")

        img_load_start = time.time()
        orig_pil_img = Image.open(image_path)
        traj_info["perf_stats"]["image_load_time"] = time.time() - img_load_start
        traj_info["original_pil_img"] = orig_pil_img

        root_node = ImageTreeNode(
            node_id=0,
            pil_image=orig_pil_img,
            parent_id=None,
            bbox_in_parent=None,
            cumulative_transform={
                "bbox_in_root": (0, 0, orig_pil_img.width, orig_pil_img.height),
                "scale_to_root": (1.0, 1.0),
            },
        )
        traj_info["image_tree"][0] = root_node

        overview_start = time.time()
        if self.pixel_budget_per_image is not None:
            overview_img, _ = constrain_image_size(orig_pil_img, max_pixels=self.pixel_budget_per_image)
        else:
            overview_img = orig_pil_img

        overview_node = ImageTreeNode(
            node_id=1,
            pil_image=overview_img,
            parent_id=0,
            bbox_in_parent=(0, 0, 1, 1),
            cumulative_transform={
                "bbox_in_root": (0, 0, orig_pil_img.width, orig_pil_img.height),
                "scale_to_root": (1.0, 1.0),
            },
        )
        traj_info["image_tree"][1] = overview_node
        b64_ov_img = encode_pil_image_to_base64(overview_node.pil_image)
        traj_info["perf_stats"]["overview_process_time"] = time.time() - overview_start
        traj_info["perf_stats"]["overview_tokens"] = count_tokens(overview_node.pil_image)

        dataset_prompt = dataset.format_prompt(sample)
        user_prompt = USER_PROMPT_TEMPLATE_MINIO3.format(dataset_prompt=dataset_prompt)
        messages = [
            {"role": "system", "content": INSTRUCTION_PROMPT_SYSTEM_MINIO3},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ov_img}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        print_messages = [
            {"role": "system", "content": INSTRUCTION_PROMPT_SYSTEM_MINIO3},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<dummy_ov_img_b64>"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        chat_message = messages
        turn_idx = 0
        try:
            while turn_idx < MAX_TURN:
                if ANSWER_START_TOKEN in response_message and ANSWER_END_TOKEN in response_message:
                    break

                params = {
                    "model": self.eval_model_name,
                    "messages": chat_message,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stop": ["<|im_end|>\n".strip(), GROUNDING_END_TOKEN],
                }

                api_call_start = time.time()
                response = await self._fetch_response(params)
                traj_info["perf_stats"]["api_call_times"].append(time.time() - api_call_start)

                response_message = response.choices[0].message.content
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                traj_info["finish_reason"].append(finish_reason)
                if finish_reason == "stop" and GROUNDING_START_TOKEN in response_message and GROUNDING_END_TOKEN not in response_message:
                    response_message += GROUNDING_END_TOKEN

                if GROUNDING_START_TOKEN in response_message:
                    try:
                        bbox2d, source = self._parse_grounding_call(response_message)
                        target_img_idx = self._source_to_image_id(source)
                        if target_img_idx not in traj_info["image_tree"]:
                            raise ValueError(f"Unknown source {source}, target image id {target_img_idx} not found")

                        traj_info["bboxes_list"].append(bbox2d)
                        traj_info["sources_list"].append(source)

                        tool_call_start = time.time()
                        if self.pixel_budget_per_image is not None:
                            new_node = crop_with_fix_retina(
                                tree=traj_info["image_tree"],
                                target_image_id=target_img_idx,
                                bbox_normalized=bbox2d,
                                original_pil_img=traj_info["original_pil_img"],
                                max_pixels=self.pixel_budget_per_image,
                            )
                            cropped_img = new_node.pil_image
                        else:
                            parent_node = traj_info["image_tree"][target_img_idx]
                            cropped_img = cropped_image_normalized(parent_node.pil_image, bbox2d, padding=0.1)
                            new_node_id = len(traj_info["image_tree"])
                            traj_info["image_tree"][new_node_id] = ImageTreeNode(
                                node_id=new_node_id,
                                pil_image=cropped_img,
                                parent_id=target_img_idx,
                                bbox_in_parent=bbox2d,
                                cumulative_transform={
                                    "bbox_in_root": None,
                                    "scale_to_root": (-1.0, -1.0),
                                },
                            )
                        traj_info["perf_stats"]["tool_call_times"].append(time.time() - tool_call_start)
                        traj_info["perf_stats"]["crop_tokens"].append(count_tokens(cropped_img))

                        encoded_img = encode_pil_image_to_base64(cropped_img)
                        followup_text = TOOL_CALL_CROP_MULTI_TRUN_PROMPT_MINIO3.format(
                            action_turn=turn_idx + 1,
                            observation_turn=turn_idx + 1,
                            observation_image="<image>",
                        )
                        followup_prefix, followup_suffix = followup_text.split("<image>", 1)

                        chat_message.extend([
                            {"role": "assistant", "content": response_message},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": followup_prefix},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}},
                                    {"type": "text", "text": followup_suffix},
                                ],
                            },
                        ])
                        print_messages.extend([
                            {"role": "assistant", "content": response_message},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": followup_prefix},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,<dummy_cropped_img>"}},
                                    {"type": "text", "text": followup_suffix},
                                ],
                            },
                        ])
                    except Exception as e:
                        logger.warning("Grounding/tool execution failed, stop rollout. sample=%s err=%s", sample.get("index"), e)
                        break
                else:
                    chat_message.append({"role": "assistant", "content": response_message})
                    print_messages.append({"role": "assistant", "content": response_message})

                turn_idx += 1
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.error("Request timeout for sample idx=%s: %s", sample.get("index", "unknown"), e)
            else:
                logger.error("Error processing sample idx=%s: %s", sample.get("index", "unknown"), e, exc_info=True)
            status = "error"

        output_text = self._extract_answer_text(response_message)
        traj_info["perf_stats"]["total_time"] = time.time() - total_start_time

        image_tree = traj_info.pop("image_tree")
        for node_id, node in image_tree.items():
            image_tree[node_id] = node.to_serializable_dict()
        traj_info["image_tree"] = image_tree

        return output_text, print_messages, status, traj_info

    @staticmethod
    def _parse_grounding_call(response_message: str) -> Tuple[List[float], str]:
        matches = re.findall(r"<grounding>(.*?)</grounding>", response_message, flags=re.DOTALL)
        if not matches:
            raise ValueError("No <grounding> payload found")
        raw = matches[-1].strip()

        try:
            payload = json.loads(raw)
        except Exception:
            payload = eval(raw)

        if isinstance(payload, list):
            payload = payload[0]
        if not isinstance(payload, dict):
            raise ValueError("Grounding payload must be a dict")

        bbox = payload.get("bbox_2d", None)
        source = payload.get("source", "original_image")
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"Invalid bbox_2d in grounding payload: {payload}")
        return [float(v) for v in bbox], str(source)

    @staticmethod
    def _source_to_image_id(source: str) -> int:
        src = source.strip().lower()
        if src == "original_image":
            return 1
        if src.startswith("observation_"):
            obs_idx = int(src.split("_")[-1])
            return obs_idx + 1
        raise ValueError(f"Unknown source type: {source}")

    @staticmethod
    def _extract_answer_text(response_message: str) -> str:
        if ANSWER_START_TOKEN in response_message and ANSWER_END_TOKEN in response_message:
            try:
                answer = response_message.split(ANSWER_START_TOKEN, 1)[1].split(ANSWER_END_TOKEN, 1)[0].strip()
                if r"\boxed{" in answer:
                    return answer.split(r"\boxed{", 1)[1].split("}", 1)[0].strip()
                return answer
            except Exception:
                return response_message
        return response_message

    @retry(
        stop=stop_after_attempt(1),
        wait=wait_exponential(min=2, max=8),
        reraise=True,
    )
    async def _fetch_response(self, params: dict):
        response = await asyncio.wait_for(
            self.client.chat.completions.create(**params),
            timeout=120,
        )
        return response
