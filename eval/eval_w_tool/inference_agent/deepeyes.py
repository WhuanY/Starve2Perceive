"""
Main inference engine with tool calling support.
Handles model inference, tool execution, and result saving.
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
    # Try relative imports first (when used as a package)
    from ..config import Config
    from ..datasets.base import DatasetBase
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config import Config
    from datasets.base import DatasetBase

from .image_utils import (
    IMAGE_FACTOR,
    encode_pil_image_to_base64,
    smart_resize,
    count_tokens,
    constrain_image_size,
    zoom_in_image_w_coordinate_mapping,
)
from .prompts import (
    INSTRUCTION_PROMPT_SYSTEM_DEEPEYES,
    USER_PROMPT_TEMPLATE_DEEPEYES,
    TOOL_CALL_START_TOKEN,
    TOOL_CALL_END_TOKEN,
    ANSWER_START_TOKEN,
    ANSWER_END_TOKEN
)


class InferenceEngine:
    """Main inference engine for VQA with tool calling."""
    
    def __init__(self, config: Config):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration object
        """
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
        """
        Infer a single sample with tool calling support.
        
        Args:
            sample: Data sample from dataset
            dataset: Dataset handler instance
            sample_n: Number of samples to infer
            
        Returns:
            Dictionary containing inference results
        """

        # Use provided sample_n or default from config
        if sample_n is None:
            sample_n = self.sample_n if self.sample_n is not None else 1
        
        # Start timing for batch inference
        batch_start_time = time.time()
        
        result_dict = {
            "index": sample.get('index'),
            "question": sample.get('question'),
            "options": sample.get('options', None),
            'answer': sample.get('answer'),
            "pred_ans": [None] * sample_n,
            "pred_output": [None] * sample_n,
            "status": [None] * sample_n,
            "traj_info": [None] * sample_n,  # Add trajectory info for each sample
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
            
            # Clean traj_info for JSON serialization (remove PIL Image objects)
            if traj_info:
                traj_info_cleaned = traj_info.copy()
                # Remove non-serializable objects
                if 'original_pil_img' in traj_info_cleaned:
                    del traj_info_cleaned['original_pil_img']
                result_dict["traj_info"][idx_n] = traj_info_cleaned
            else:
                result_dict["traj_info"][idx_n] = traj_info
        
        # Calculate batch inference time
        batch_end_time = time.time()
        batch_total_time = batch_end_time - batch_start_time
        
        # Add batch-level performance statistics
        result_dict["batch_perf_stats"] = {
            "sample_n": sample_n,
            "batch_total_time": batch_total_time,
            "avg_time_per_sample": batch_total_time / sample_n if sample_n > 0 else 0.0,
            "individual_times": [traj_info['perf_stats']['total_time'] if traj_info and 'perf_stats' in traj_info else 0.0 
                                 for traj_info in result_dict["traj_info"]],
            "max_individual_time": max([traj_info['perf_stats']['total_time'] if traj_info and 'perf_stats' in traj_info else 0.0 
                                        for traj_info in result_dict["traj_info"]], default=0.0),
            "min_individual_time": min([traj_info['perf_stats']['total_time'] if traj_info and 'perf_stats' in traj_info else float('inf') 
                                        for traj_info in result_dict["traj_info"]], default=0.0),
            "parallelization_speedup": sum([traj_info['perf_stats']['total_time'] if traj_info and 'perf_stats' in traj_info else 0.0 
                                            for traj_info in result_dict["traj_info"]]) / batch_total_time if batch_total_time > 0 else 1.0,
        }

        # adding original fields to result_dict
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
        """
        Run inference on a list of samples asynchronously.
        
        Args:
            dataset: Dataset handler
            data_slice: List of data samples to process
            output_path: Path to save results (JSONL format)
            num_workers: Number of concurrent workers (uses config if None)
            save_step: Number of inference steps (batches) before saving results
            
        Returns:
            List of inference results
        """
        num_workers = num_workers or self.config.num_workers
        
        num_samples = len(data_slice)
        
        # ======== Async Inference =========
        logger.info(f"Inference Engine for DeepEyes! {self.pixel_budget_per_image=}")
        logger.info(f"Processing {num_samples} samples")
        logger.info(f"Using {num_workers} concurrent workers")

        semaphore = asyncio.Semaphore(num_workers)
    
        async def process_with_index(idx: int, sample: Dict[str, Any]) -> tuple[int, Optional[Dict[str, Any]]]:
            """Process a sample and return its index along with the result."""
            async with semaphore:
                result = await self.infer_sample(sample, dataset, sample_n=self.sample_n)
                return (idx, result)
        
        batch_size = 10 # async with fix batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size

        logger.info(f"Batch size: {batch_size}, Total batches: {num_batches}")

        # Create directory if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            logger.info(f"Saving results to {output_path}...")

        all_results = []
        pending_results = []  # Results waiting to be saved
        
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Processing batches"): # control async tasks number
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_data = data_slice[start_idx:end_idx]

            tasks = [process_with_index(start_idx + idx_in_batch, batch_data[idx_in_batch]) for idx_in_batch in range(len(batch_data))]
            batch_results_with_idx = await asyncio.gather(*tasks)
            
            # Extract results from (idx, result) tuples
            batch_results = []
            for idx, result in batch_results_with_idx:
                if result is not None:
                    batch_results.append(result)
                    all_results.append(result)
                    pending_results.append(result)
            
            # Save results every save_step batches (using simple append mode)
            if output_path and batch_results and (batch_idx + 1) % save_step == 0:
                self._append_batch_results(output_path, pending_results)
                pending_results = []  # Clear pending results after saving
        
        # Save any remaining results at the end
        if output_path and pending_results:
            self._append_batch_results(output_path, pending_results)
        
        logger.info(f"✓ Inference complete. Processed {len(all_results)} samples.")
        
        # Print aggregated performance statistics
        if all_results:
            logger.info("\n" + "="*60)
            logger.info("AGGREGATED PERFORMANCE STATISTICS")
            logger.info("="*60)
            agg_stats = self.aggregate_perf_stats(all_results)
            
            # Trajectory-level stats
            logger.info(f"Total Samples:              {agg_stats['total_samples']}")
            logger.info(f"Avg Image Load Time:        {agg_stats['avg_image_load_time']:.4f}s")
            logger.info(f"Avg Overview Process Time:  {agg_stats['avg_overview_process_time']:.4f}s")
            logger.info(f"Avg API Call Time:          {agg_stats['avg_api_call_time']:.4f}s")
            logger.info(f"Avg Tool Call Time:         {agg_stats['avg_tool_call_time']:.4f}s")
            logger.info(f"Avg Total Time per Sample:  {agg_stats['avg_total_time']:.4f}s")
            logger.info(f"Avg API Calls per Sample:   {agg_stats['avg_num_api_calls']:.2f}")
            logger.info(f"Avg Tool Calls per Sample:  {agg_stats['avg_num_tool_calls']:.2f}")
            logger.info(f"Total API Calls:            {agg_stats['total_api_calls']}")
            logger.info(f"Total Tool Calls:           {agg_stats['total_tool_calls']}")
            
            # Batch-level stats (if available)
            if agg_stats['total_batches'] > 0:
                logger.info(f"\n--- Batch Inference Statistics ---")
                logger.info(f"Total Batches:              {agg_stats['total_batches']}")
                logger.info(f"Avg Batch Size (sample_n):  {agg_stats['avg_batch_sample_n']:.2f}")
                logger.info(f"Avg Batch Total Time:       {agg_stats['avg_batch_total_time']:.4f}s")
                logger.info(f"Avg Parallelization Speedup:{agg_stats['avg_batch_speedup']:.2f}x")
                logger.info(f"Avg Parallel Efficiency:    {agg_stats['avg_batch_efficiency']:.1f}%")
                logger.info(f"\nComparison:")
                logger.info(f"  Single Trajectory Time:   {agg_stats['avg_total_time']:.4f}s")
                logger.info(f"  Batch Time (sample_n={agg_stats['avg_batch_sample_n']:.1f}): {agg_stats['avg_batch_total_time']:.4f}s")
                if agg_stats['avg_batch_sample_n'] > 0:
                    expected_seq_time = agg_stats['avg_total_time'] * agg_stats['avg_batch_sample_n']
                    actual_speedup = expected_seq_time / agg_stats['avg_batch_total_time'] if agg_stats['avg_batch_total_time'] > 0 else 0
                    logger.info(f"  Expected Sequential Time: {expected_seq_time:.4f}s")
                    logger.info(f"  Actual Speedup vs Sequential: {actual_speedup:.2f}x")
            
            logger.info("="*60 + "\n")
        
        return all_results

    def _append_batch_results(self, output_path: str, batch_results: List[Dict[str, Any]]):
        """
        Append a batch of results to the output file in append mode.
        Simple and fast - just appends new results without reading existing file.
        
        Args:
            output_path: Path to the JSONL file
            batch_results: List of result dictionaries to append
        """
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Error appending batch results: {e}", exc_info=True)

    @staticmethod
    def print_perf_stats(traj_info: Dict[str, Any], sample_index: int = None):
        """
        Print performance statistics from traj_info.
        
        Args:
            traj_info: Trajectory information containing perf_stats
            sample_index: Optional sample index for identification
        """
        if traj_info is None or 'perf_stats' not in traj_info:
            return
            
        perf = traj_info['perf_stats']
        header = f"Performance Stats for Sample {sample_index}" if sample_index is not None else "Performance Stats"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{header}")
        logger.info(f"{'='*60}")
        logger.info(f"Image Load Time:        {perf['image_load_time']:.4f}s")
        logger.info(f"Overview Process Time:  {perf['overview_process_time']:.4f}s")
        logger.info(f"Number of API Calls:    {len(perf['api_call_times'])}")
        
        if perf['api_call_times']:
            logger.info(f"API Call Times:")
            for i, t in enumerate(perf['api_call_times'], 1):
                logger.info(f"  Round {i}: {t:.4f}s")
            logger.info(f"  Total API Time:       {sum(perf['api_call_times']):.4f}s")
            logger.info(f"  Avg API Time:         {sum(perf['api_call_times'])/len(perf['api_call_times']):.4f}s")
        
        logger.info(f"Number of Tool Calls:   {len(perf['tool_call_times'])}")
        if perf['tool_call_times']:
            logger.info(f"Tool Call Times:")
            for i, t in enumerate(perf['tool_call_times'], 1):
                logger.info(f"  Call {i}: {t:.4f}s")
            logger.info(f"  Total Tool Time:      {sum(perf['tool_call_times']):.4f}s")
            logger.info(f"  Avg Tool Time:        {sum(perf['tool_call_times'])/len(perf['tool_call_times']):.4f}s")
        
        logger.info(f"Total Time:             {perf['total_time']:.4f}s")
        logger.info(f"{'='*60}\n")

    @staticmethod
    def print_batch_perf_stats(batch_perf: Dict[str, Any], sample_index: int = None):
        """
        Print batch-level performance statistics.
        
        Args:
            batch_perf: Batch performance statistics dictionary
            sample_index: Optional sample index for identification
        """
        if batch_perf is None:
            return
            
        header = f"Batch Performance Stats for Sample {sample_index}" if sample_index is not None else "Batch Performance Stats"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{header}")
        logger.info(f"{'='*60}")
        logger.info(f"Sample N (batch size):      {batch_perf['sample_n']}")
        logger.info(f"Batch Total Time:           {batch_perf['batch_total_time']:.4f}s")
        logger.info(f"Avg Time per Sample:        {batch_perf['avg_time_per_sample']:.4f}s")
        logger.info(f"Max Individual Time:        {batch_perf['max_individual_time']:.4f}s")
        logger.info(f"Min Individual Time:        {batch_perf['min_individual_time']:.4f}s")
        logger.info(f"Parallelization Speedup:    {batch_perf['parallelization_speedup']:.2f}x")
        
        if batch_perf['individual_times']:
            logger.info(f"\nIndividual Sample Times:")
            for i, t in enumerate(batch_perf['individual_times'], 1):
                logger.info(f"  Sample {i}: {t:.4f}s")
        
        # Calculate efficiency
        if batch_perf['max_individual_time'] > 0:
            efficiency = (batch_perf['batch_total_time'] / batch_perf['max_individual_time']) * 100
            logger.info(f"\nParallel Efficiency:        {efficiency:.1f}%")
            logger.info(f"  (100% = perfect parallel, batch_time = max_individual_time)")
        
        logger.info(f"{'='*60}\n")


    @staticmethod
    def aggregate_perf_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate performance statistics from multiple results.
        
        Args:
            results: List of result dictionaries containing traj_info
            
        Returns:
            Dictionary with aggregated statistics
        """
        all_stats = {
            'image_load_times': [],
            'overview_process_times': [],
            'api_call_times': [],
            'tool_call_times': [],
            'total_times': [],
            'num_api_calls': [],
            'num_tool_calls': [],
            # Batch-level stats
            'batch_total_times': [],
            'batch_sample_ns': [],
            'batch_speedups': [],
            'batch_efficiencies': [],
        }
        
        for result in results:
            # Process trajectory-level stats
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
                all_stats['tool_call_times'].extend(perf['tool_call_times'])
                all_stats['total_times'].append(perf['total_time'])
                all_stats['num_api_calls'].append(len(perf['api_call_times']))
                all_stats['num_tool_calls'].append(len(perf['tool_call_times']))
            
            # Process batch-level stats
            if 'batch_perf_stats' in result:
                batch_perf = result['batch_perf_stats']
                all_stats['batch_total_times'].append(batch_perf['batch_total_time'])
                all_stats['batch_sample_ns'].append(batch_perf['sample_n'])
                all_stats['batch_speedups'].append(batch_perf['parallelization_speedup'])
                if batch_perf['max_individual_time'] > 0:
                    efficiency = (batch_perf['batch_total_time'] / batch_perf['max_individual_time']) * 100
                    all_stats['batch_efficiencies'].append(efficiency)
        
        # Calculate aggregates
        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        aggregated = {
            'total_samples': len([t for r in results if 'traj_info' in r for t in (r['traj_info'] if isinstance(r['traj_info'], list) else [r['traj_info']]) if t is not None]),
            'avg_image_load_time': safe_avg(all_stats['image_load_times']),
            'avg_overview_process_time': safe_avg(all_stats['overview_process_times']),
            'avg_api_call_time': safe_avg(all_stats['api_call_times']),
            'avg_tool_call_time': safe_avg(all_stats['tool_call_times']),
            'avg_total_time': safe_avg(all_stats['total_times']),
            'avg_num_api_calls': safe_avg(all_stats['num_api_calls']),
            'avg_num_tool_calls': safe_avg(all_stats['num_tool_calls']),
            'total_api_calls': len(all_stats['api_call_times']),
            'total_tool_calls': len(all_stats['tool_call_times']),
            # Batch-level aggregates
            'total_batches': len(all_stats['batch_total_times']),
            'avg_batch_total_time': safe_avg(all_stats['batch_total_times']),
            'avg_batch_sample_n': safe_avg(all_stats['batch_sample_ns']),
            'avg_batch_speedup': safe_avg(all_stats['batch_speedups']),
            'avg_batch_efficiency': safe_avg(all_stats['batch_efficiencies']),
        }
        
        return aggregated

    
    async def _async_rollout_a_request(self, dataset: DatasetBase, sample: Dict[str, Any]) -> tuple[str, list[dict], str]:
        """
        Execute a single inference request with tool calling support.
        
        This method performs one complete inference rollout, which may include:
        - Initial image encoding
        - Multi-turn conversation with tool calls
        - Image cropping and re-analysis
        - Answer extraction
        
        Args:
            sample: Data sample containing question and image info
            dataset: Dataset handler for prompt formatting
            
        Returns:
            tuple: (output_text, print_messages, status)
                - output_text: Extracted answer string
                - print_messages: Conversation history for logging
                - status: 'success' or 'error'
        """
        image_path = sample.get('image_path', "")
        traj_info = {
            "original_img_path": image_path,
            "original_pil_img": None,
            "scale": (1.0, 1.0), # ovewview_w / orig_w, ovewview_h / orig_h
            "bboxes_list": [],
            "finish_reason": [], # finish_reason[i]: i-th tool call finish reason
            # Performance statistics
            "perf_stats": {
                "image_load_time": 0.0,
                "overview_process_time": 0.0,
                "api_call_times": [],
                "tool_call_times": [],
                "total_time": 0.0,
                "overview_tokens": 0, 
                "crop_tokens": [],
            }
        }
        
        # Start total timing
        total_start_time = time.time()
        
        # Handle single image path 
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


        # Time overview image processing (resize + encode)
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
        user_prompt = USER_PROMPT_TEMPLATE_DEEPEYES.format(dataset_prompt=dataset_prompt)
        messages = [
            {
                "role": "system",
                "content": INSTRUCTION_PROMPT_SYSTEM_DEEPEYES
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ov_img}"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        # For saving (without actual image data, use a dummy image)
        print_messages = [
            {
                "role": "system",
                "content": INSTRUCTION_PROMPT_SYSTEM_DEEPEYES
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<dummy_b64_ov_img>"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        chat_message = messages
        response_message = ""
        status = 'success'
        try_count = 0
        
        try:
            while ANSWER_END_TOKEN not in response_message:
                if ANSWER_END_TOKEN in response_message and ANSWER_START_TOKEN in response_message:
                    break
                
                if try_count > 3:
                    logger.info(f"{sample['index']=}try_count > 3, break")
                    break
                
                params = {
                    "model": self.eval_model_name, # openai api server requires this field.
                    "messages": chat_message,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stop": ["<|im_end|>\n".strip(), TOOL_CALL_END_TOKEN],
                }
                
                # API call with retry   
                api_call_start = time.time()
                response = await self._fetch_response(params)
                api_call_time = time.time() - api_call_start
                traj_info['perf_stats']['api_call_times'].append(api_call_time)
                
                response_message = response.choices[0].message.content
                
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                traj_info['finish_reason'].append(finish_reason)
                if finish_reason == 'stop' and TOOL_CALL_START_TOKEN in response_message and TOOL_CALL_END_TOKEN not in response_message:
                    response_message += TOOL_CALL_END_TOKEN
                
                # Handle tool calling
                if TOOL_CALL_START_TOKEN in response_message:
                    tool_call_json = response_message.split(TOOL_CALL_START_TOKEN)[1].split(TOOL_CALL_END_TOKEN)[0].strip()
                    try:
                        action_list = json.loads(tool_call_json)
                    except:
                        action_list = eval(tool_call_json)
                    logger.debug(f"{sample['index']=} {try_count=} {action_list=}")
                    traj_info['bboxes_list'].append(action_list['arguments']['bbox_2d'])

                    # Time tool call (crop + encode)
                    tool_call_start = time.time()
                    bbox_str = action_list['arguments']['bbox_2d']
                    bbox = bbox_str
                    cropped_image = None
                    left, top, right, bottom = bbox

                    if self.pixel_budget_per_image is not None:
                        cropped_image = zoom_in_image_w_coordinate_mapping(bbox, traj_info)
                    else:
                        cropped_image = overview_img.crop((left, top, right, bottom))
                        new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                        cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                    
                    traj_info['perf_stats']['crop_tokens'].append(count_tokens(cropped_image))
                    b64_crop_img = encode_pil_image_to_base64(cropped_image)
                    tool_call_time = time.time() - tool_call_start
                    traj_info['perf_stats']['tool_call_times'].append(tool_call_time)

                    
                    content_f = [{"type": "text", "text":"<tool_response>"}]
                    content_f.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_crop_img}"}})
                    content_f.append({"type": "text", "text": "</tool_response>"})
                    
                    chat_message.extend([
                        {"role": "assistant", "content": response_message},
                        {"role": "user", "content": content_f}
                    ])
                    
                    content_f_p = [{"type": "text", "text":"<tool_response>" }]
                    content_f_p.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,<dummy_b64_crop_img>"}}) # dummy image
                    content_f_p.append({"type": "text", "text": "</tool_response>"})

                    print_messages.extend([
                        {"role": "assistant", "content": response_message},
                        {"role": "user", "content": content_f_p}
                    ])
                else:
                    chat_message.append({"role": "assistant", "content": response_message})
                    print_messages.append({"role": "assistant", "content": response_message})
                
                try_count += 1
                
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
        
        # Calculate total time
        traj_info['perf_stats']['total_time'] = time.time() - total_start_time
        
        return output_text, print_messages, status, traj_info

    @retry(
    stop=stop_after_attempt(1),  # 最多 1 次尝试
    wait=wait_exponential(min=2, max=8),  # 2s, 4s 退避
    reraise=True
)
    async def _fetch_response(self, params: dict):
        response = await asyncio.wait_for(
            self.client.chat.completions.create(**params),
            timeout=60
        )
        return response

if __name__ == "__main__":
    async def test():
        from datasets.generalvqa import GeneralVQADataset
        dataset = GeneralVQADataset("/root/autodl-tmp/benchmarks/testbench")
        
        engine = InferenceEngine(
            config=Config(
                api_key="none", 
                api_url="http://localhost:8000/v1",
                model_name="Qwen2.5-VL-3B-Instruct",
                dataset_name=None,
                dataset_path=None,
                save_path=None,
                eval_model_name="Qwen2.5-VL-3B-Instruct",
                num_workers=1,
                max_tokens=10240,
                temperature=0.0,
            )
        )
        
        # result = await engine.infer_sample(data[0], dataset) # 测试推理单条记录
        sample = dataset.load_data()[0]
        # result = await engine.infer_sample(sample, dataset)
        results = await engine.infer_dataset(
            dataset=dataset,
            data_slice=[sample],
        )
        print(results)

    asyncio.run(test())
