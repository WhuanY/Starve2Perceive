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
from typing import Dict, Any, List, Tuple, Optional
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
    MAX_VIEW_PIXELS,
    encode_pil_image_to_base64,
    constrain_image_size,
    focus, # main function for fix retina
    count_tokens,
)
from .prompts import (
    INSTRUCTION_PROMPT_SYSTEM_PIXELREASONER,
    USER_PROMPT_TEMPLATE_PIXELREASONER,
    TOOL_CALL_START_TOKEN,
    TOOL_CALL_END_TOKEN,
)

def cropped_image_normalized(image, bbox_2d, padding=0.1):
    """
    Crop the image based on the bounding box coordinates.
    """
    img_x, img_y = image.size
    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding, float(bbox_2d[1])-padding, float(bbox_2d[2])+padding, float(bbox_2d[3])+padding)
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding, float(bbox_2d[1])/img_y-padding, float(bbox_2d[2])/img_x+padding, float(bbox_2d[3])/img_y+padding)
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    cropped_img = image.crop((normalized_x1*img_x, normalized_y1*img_y, normalized_x2*img_x, normalized_y2*img_y))
    w, h = cropped_img.size
    assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"

    return cropped_img

class ImageTreeNode:
    def __init__(self, 
                 node_id: int,
                 pil_image: Image,  # The constrained image at this node
                 parent_id: Optional[int],
                 bbox_in_parent: Optional[Tuple[float, float, float, float]],  # normalized [0,1]
                 cumulative_transform: Dict):
        """
        node_id: Unique identifier (1-indexed for user-facing API)
        pil_image: The PIL image at this node (after fix-retina constraint)
        parent_id: ID of parent node (None for root/original)
        bbox_in_parent: The bbox that was cropped from parent (normalized coords in parent's space)
        cumulative_transform: Transformation info to map back to root (original image)
        """
        self.node_id = node_id
        self.pil_image = pil_image
        self.parent_id = parent_id
        self.bbox_in_parent = bbox_in_parent
        self.cumulative_transform = cumulative_transform
        # cumulative_transform contains:
        # - 'bbox_in_root': The region this node represents in root's pixel space
        # - 'scale': self.pil_img.width / self.orig_pil_img.width, self.pil_img.height / self.orig_pil_img.height

    def __repr__(self):
        return f"ImageTreeNode(node_id={self.node_id}, parent_id={self.parent_id}, bbox_in_parent={self.bbox_in_parent}, cumulative_transform={self.cumulative_transform})"

    def to_serializable_dict(self, include_image_data=False):
        result = {
            'node_id': self.node_id,
            'parent_id': self.parent_id,
            'bbox_in_parent': self.bbox_in_parent,
            'cumulative_transform': self.cumulative_transform,
            'image_size': self.pil_image.size, 
        }
        if include_image_data:
            logger.warning("include_image_data is True. This might take time.")
            from .image_utils import encode_pil_image_to_base64
            result['image_base64'] = encode_pil_image_to_base64(self.pil_image)

        return result

def create_dummy_node() -> ImageTreeNode:
    """
    Create a dummy ImageTreeNode for error handling when bbox is invalid.
    Returns a node with a small black square image and dummy metadata.
    """
    # Create a small black square as fallback image
    dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
    
    return ImageTreeNode(
        node_id=-1,  # Use -1 to indicate error/dummy node
        pil_image=dummy_image,
        parent_id=None,
        bbox_in_parent=None,
        cumulative_transform={
            'bbox_in_root': (0, 0, 0, 0),  # Invalid bbox
            'scale_to_root': (0.0, 0.0)     # No scaling
        }
    )

def compute_cumulative_transform(parent_node: ImageTreeNode, 
                                 bbox_in_parent_normalized: Tuple[float, float, float, float],
                                 parent_image_size: Tuple[int, int]) -> Dict:
    """
    Compute transformation from new child node back to root (original image).
    
    Args:
        parent_node: The parent node being cropped
        bbox_in_parent_normalized: Normalized bbox [0,1] in parent's coordinate space
        parent_image_size: (width, height) of parent's constrained image
    
    Returns:
        Dict containing:
        - 'bbox_in_root': (x1, y1, x2, y2) in root's absolute pixel coordinates
        - 'scale_to_root': (scale_x, scale_y) mapping from child to root
    """
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_in_parent_normalized
    parent_w, parent_h = parent_image_size
    
    # Step 1: Convert normalized bbox to parent's pixel coordinates
    x1_parent = x1_norm * parent_w
    y1_parent = y1_norm * parent_h
    x2_parent = x2_norm * parent_w
    y2_parent = y2_norm * parent_h
    
    # Step 2: Map parent's pixel coords to root's pixel coords
    assert parent_node.node_id > 0
    # Parent is already a cropped node
    parent_bbox_in_root = parent_node.cumulative_transform['bbox_in_root']
    parent_x1_root, parent_y1_root, _, _ = parent_bbox_in_root
    
    # Get the region width/height in root space
    parent_root_region_w = parent_bbox_in_root[2] - parent_bbox_in_root[0]
    parent_root_region_h = parent_bbox_in_root[3] - parent_bbox_in_root[1]
    
    # Map: child's pixel position → fraction of parent → absolute position in root
    x1_root = parent_x1_root + (x1_parent / parent_w) * parent_root_region_w
    y1_root = parent_y1_root + (y1_parent / parent_h) * parent_root_region_h
    x2_root = parent_x1_root + (x2_parent / parent_w) * parent_root_region_w
    y2_root = parent_y1_root + (y2_parent / parent_h) * parent_root_region_h

    return {
        'bbox_in_root': (x1_root, y1_root, x2_root, y2_root), # parent node中的image被crop, cropped image的坐标在root node中的位置
        'scale_to_root': None  # Computed after fix-retina resize
    }


def crop_with_fix_retina(tree: Dict[int, ImageTreeNode],
                       target_image_id: int,
                       bbox_normalized: Tuple[float, float, float, float],
                       original_pil_img: Image,
                       max_pixels: int) -> ImageTreeNode:
    """
    Perform zoom operation with fix-retina constraint.
    
    Steps:
    1. Get the target node from tree
    2. Compute cumulative transform to map bbox back to root coordinates
    3. Crop from original image (max resolution)
    4. Apply fix-retina constraint (resize to max_pixels)
    5. Create new node with appropriate transform info
    6. Add to tree
    """
    # Step 1: Get target node:
    target_node = tree[target_image_id]

    # Step 2: Compute where this bbox maps to in the root coordinates
    cumulative_transform = compute_cumulative_transform(
        parent_node = target_node,
        bbox_in_parent_normalized = bbox_normalized,
        parent_image_size = target_node.pil_image.size
    )

    # Step 3: Crop from original image (high-res)
    bbox_in_root = cumulative_transform['bbox_in_root']
    x1, y1, x2, y2 = bbox_in_root

    # Clamp to original image bounds
    orig_w, orig_h = original_pil_img.size
    x1 = max(0, min(x1, orig_w))
    y1 = max(0, min(y1, orig_h))
    x2 = max(0, min(x2, orig_w))
    y2 = max(0, min(y2, orig_h))

    if x1 >= x2 or y1 >= y2:
        logger.warning(f"Invalid bbox! Target Node: {target_node}")
        return create_dummy_node() # TODO: Implement it 

    cropped_high_res = original_pil_img.crop((x1, y1, x2, y2))
    
    # Step 4: Apply fix-retina constraint
    constrained_img, _= constrain_image_size(
        cropped_high_res,
        max_pixels = max_pixels
    ) 

    cumulative_transform['bbox_in_root'] = (x1, y1, x2, y2)

    scale_x = constrained_img.width / original_pil_img.width
    scale_y = constrained_img.height / original_pil_img.height
    cumulative_transform['scale_to_root'] = (scale_x, scale_y)

    # Step 5: Create new node
    new_node_id = len(tree)
    new_node = ImageTreeNode(
        node_id = new_node_id, 
        pil_image = constrained_img,
        parent_id = target_image_id, 
        bbox_in_parent = bbox_normalized,
        cumulative_transform = cumulative_transform,
    )

    tree[new_node_id] = new_node
    
    return new_node

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
        logger.info("PixelReasoner InferenceEngine! ")
        logger.info(f"Processing {num_samples} samples")
        logger.info(f"Using {num_workers} concurrent workers")

        
        semaphore = asyncio.Semaphore(num_workers)
    
        async def process_with_index(idx: int, sample: Dict[str, Any]) -> tuple[int, Optional[Dict[str, Any]]]:
            """Process a sample and return its index along with the result."""
            async with semaphore:
                result = await self.infer_sample(sample, dataset, sample_n=self.sample_n)
                return (idx, result)
        
        batch_size = 5 # async with fix batch_size
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
            "image_tree": {}, # Dict[node_id -> ImageTreeNode]
            "next_node_id": 1, # Counter for node IDs (0 is reserved for original)
            "bboxes_list": [],
            "finish_reason": [],
            "perf_stats": {
                "image_load_time": 0.0,
                "api_call_times": [],
                "tool_call_times": [],
                "total_time": 0.0,
                'overview_tokens': 0,
                "crop_tokens": [],
                'overview_process_time': 0.0
            }
        }
        
        # Start total timing
        total_start_time = time.time()
        
        # Handle single image path 
        loop = asyncio.get_running_loop()
        orig_pil_img = None 
        if isinstance(image_path, str) and image_path != "":
            if os.path.exists(image_path):
                # Time image loading
                img_load_start = time.time()
                orig_pil_img = Image.open(image_path)
                traj_info['perf_stats']['image_load_time'] = time.time() - img_load_start
                traj_info['original_pil_img'] = orig_pil_img
            else:
                raise ValueError(f"Image not found: {image_path}")
        else:
            raise ValueError(f"Image not found: {image_path}")

        root_node = ImageTreeNode(
            node_id = 0, 
            pil_image = orig_pil_img,
            parent_id = None,
            bbox_in_parent = None,
            cumulative_transform = {
                'bbox_in_root': (0, 0, orig_pil_img.width, orig_pil_img.height),
                'scale_to_root': (1.0, 1.0)
            }
        )
        traj_info['image_tree'][0] = root_node

        overview_start = time.time()
        if self.pixel_budget_per_image is not None:
            overview_img, overview_scale = constrain_image_size(orig_pil_img, max_pixels=self.pixel_budget_per_image)
        else:
            overview_img, overview_scale = orig_pil_img, (1.0, 1.0)

        overview_node = ImageTreeNode(
                node_id = 1,
                pil_image = overview_img, # which is orig_pil_img in naive case
                parent_id = 0,
                bbox_in_parent = (0, 0, 1, 1),
                cumulative_transform = {
                    'bbox_in_root': (0, 0, orig_pil_img.width, orig_pil_img.height),
                    'scale_to_root': (1.0, 1.0)
                }
            )
        traj_info['image_tree'][1] = overview_node
        b64_ov_img = encode_pil_image_to_base64(overview_node.pil_image)
        traj_info['perf_stats']['overview_process_time'] = time.time() - overview_start
        traj_info['scale'] = overview_scale
        traj_info['perf_stats']['overview_tokens'] = count_tokens(overview_node.pil_image)
        
        dataset_prompt = dataset.format_prompt(sample)
        user_prompt = USER_PROMPT_TEMPLATE_PIXELREASONER.format(dataset_prompt=dataset_prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_ov_img}"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        if INSTRUCTION_PROMPT_SYSTEM_PIXELREASONER != "":
            messages.insert(0, {
                "role": "system",
                "content": INSTRUCTION_PROMPT_SYSTEM_PIXELREASONER
            })
        
        # For saving (without actual image data)
        print_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<dummy_ori_img_b64>"}},
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        if INSTRUCTION_PROMPT_SYSTEM_PIXELREASONER != "":
            print_messages.insert(0, {
                "role": "system",
                "content": INSTRUCTION_PROMPT_SYSTEM_PIXELREASONER
            })
        
        chat_message = messages
        response_message = ""
        status = 'success'
        try_count = 0
        image_cnts = 1 # start with start token
        
        try:
            while (response_message == "") or (response_message.endswith(TOOL_CALL_END_TOKEN)):
                if r"\boxed{" in response_message:
                    print("Model generates answer. Break")
                    break
                
                if try_count > 3:
                    logger.info(f"{sample['index']=} Try tool call times > 3. Break")
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
                    bbox2d = action_list['arguments']['bbox_2d']
                    target_img_idx = int(action_list['arguments']['target_image'])
                    if target_img_idx == 0:
                        logger.warning("PixelReasoner Choose to Crop Image 0, this is illegal. This trace should be terminaled")
                        break

                    # Time tool call (crop + encode)
                    tool_call_start = time.time()
                    
                    assert action_list['name'] == 'crop_image_normalized', "only image is in the chat history."
                    if self.pixel_budget_per_image is not None:
                        new_node = crop_with_fix_retina(
                            tree = traj_info['image_tree'],
                            target_image_id = target_img_idx,
                            bbox_normalized = bbox2d,
                            original_pil_img = traj_info['original_pil_img'],
                            max_pixels = self.pixel_budget_per_image
                        )
                        cropped_img = new_node.pil_image
                    else: # naive pixel_reasoner implementation
                        parent_node = traj_info['image_tree'][target_img_idx]
                        cropped_img = cropped_image_normalized(
                            parent_node.pil_image,
                            bbox2d, 
                            padding=0.1,
                        )
                        new_node_id = len(traj_info['image_tree'])
                        new_node = ImageTreeNode(
                            node_id = new_node_id,
                            pil_image = cropped_img,
                            parent_id = target_img_idx,
                            bbox_in_parent = bbox2d,
                            cumulative_transform = {
                                'bbox_in_root': None, 
                                'scale_to_root': (-1.0, -1.0) # # No mapping needed, assign with a dummy value
                            }
                        )
                        traj_info['image_tree'][new_node_id] = new_node
                    tool_call_time = time.time() - tool_call_start
                    traj_info['perf_stats']['tool_call_times'].append(tool_call_time)
                    traj_info['perf_stats']['crop_tokens'].append(count_tokens(cropped_img))
                    
                    encoded_img = encode_pil_image_to_base64(cropped_img)
                    
                    content_f = [{"type": "text", "text":"<tool_response>"}]
                    content_f.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}})
                    image_cnts += 1
                    content_f.append({"type": "text", "text": "</tool_response>"})
                    
                    chat_message.extend([
                        {"role": "assistant", "content": response_message},
                        {"role": "user", "content": content_f}
                    ])
                    
                    content_f_p = [{"type": "text", "text":"<tool_response>" }]
                    content_f_p.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,<dummy_b64_fc_img>"}})
                    content_f_p.append({"type": "text", "text": "</tool_response>"})

                    print_messages.extend([
                        {"role": "assistant", "content": response_message},
                        {"role": "user", "content": content_f_p}
                    ])
                else:
                    chat_message.append({"role": "assistant", "content": response_message})
                    print_messages.append({"role": "assistant", "content": response_message})
                
                try_count += 1
                if image_cnts > 16:
                    print("Too many images, break")
                    break
                
        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                logger.error(f"Request timeout for sample idx: {sample.get('index', 'unknown')}: {e}")
            else:
                logger.error(f"Error processing sample {sample.get('index', 'unknown')}: {e}", exc_info=True)
            status = 'error'
        
        if r'\boxed{' in response_message:
            output_text = response_message.split(r'\boxed{')[1].split('}')[0].strip()
        else:
            output_text = response_message
        
        # Calculate total time
        traj_info['perf_stats']['total_time'] = time.time() - total_start_time

        # convert to 
        image_tree = traj_info.pop('image_tree')
        for node_id, node in image_tree.items():
            image_tree[node_id] = node.to_serializable_dict()
        traj_info['image_tree'] = image_tree

        return output_text, print_messages, status, traj_info

    @retry(
    stop=stop_after_attempt(1),  # 最多 1 次尝试
    wait=wait_exponential(min=2, max=8),  # 2s, 4s 退避
    reraise=True
)
    async def _fetch_response(self, params: dict):
        response = await asyncio.wait_for(
            self.client.chat.completions.create(**params),
            timeout=120
        )
        return response

    def selectFrames(images, target_frames):
        """
        Select Frames from a video
        """
        return [images[tgt] for tgt in target_frames]

if __name__ == "__main__":
    async def test():
        # from datasets.hrbench import HRBenchDataset
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

