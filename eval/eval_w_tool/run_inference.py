#!/usr/bin/env python3
"""
Main entry point for running inference on VQA datasets with tool calling support.

Usage:
    python run_inference.py \
        --dataset_path /path/to/hrbench/hr_bench_4k.tsv \
        --dataset_name HRBench4K \
        --save_path /path/to/output \
        --model_name my_model \
        --api_key YOUR_API_KEY \
        --api_url http://localhost:8000/v1 \
        --start_index 0 \
        --end_index 10
    
    # Process all samples from index 0:
    python run_inference.py ... --start_index 0 --end_index -1
    
    # Process samples 10-20:
    python run_inference.py ... --start_index 10 --end_index 20
"""
import argparse
import sys
import os
import json
import asyncio
import logging
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, load_config
from datasets import (
    HRBenchDataset, 
    VStarDataset, 
    FixRetinaSFTDataset, 
    GeneralVQADataset, 
    MMERealWorldLiteDataset,
    VisualProbeDataset,
    TreeBenchDataset,
    TreeVGRRLDataset,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logger.addHandler(handler)

async def main():
    parser = argparse.ArgumentParser(description="Run VQA inference with tool calling")
    
    # Configuration arguments
    parser.add_argument('--api_key', type=str, default=None, help='API key')
    parser.add_argument('--api_url', type=str, default=None, help='API URL')
    parser.add_argument('--model_name', type=str, default=None, help='Model name for saving results')
    parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for inference')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to dataset file or directory')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, required=True, 
                       help='Base path for saving results')
    parser.add_argument('--image_dir', type=str, default=None, 
                       help='Directory for saving images')
                       
    # Inference arguments
    parser.add_argument('--num_workers', type=int, default=None, 
                       help='Number of worker processes')
    parser.add_argument('--max_tokens', type=int, default=10240, 
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, 
                       help='Sampling temperature')
    parser.add_argument('--sample_n', type=int, default=1, 
                       help='Number of samples per question')
    parser.add_argument('--inference_agent', type=str, default="fixretina",
                       help="inference agent choice")
    parser.add_argument('--pixel_budget_per_image', type=int, default=None, 
                       help="pixel budget per image, default to None")
    parser.add_argument('--global_visual_token_constrain', type=int, default=None,
                       help="global visual token constrain, default to None")
    
    # Dataset slicing arguments
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index for dataset (0-based, inclusive)')
    parser.add_argument('--end_index', type=int, default=-1,
                       help='End index for dataset (0-based, exclusive). -1 means process all remaining samples')

    # Regen argument
    parser.add_argument('--regen', action='store_true', 
                       help='Force regeneration of all samples, ignoring existing results.')
    
    # Save step argument
    parser.add_argument('--save_step', type=int, default=1,
                       help='Number of inference steps (batches) before saving results. Default: 1 (save after each batch). Must be >= 1.')
    
    
    args = parser.parse_args()
    
    # Validate save_step
    if args.save_step < 1:
        raise ValueError(f"--save_step must be >= 1, got {args.save_step}")
    
    # load global config
    config = load_config(
        api_key=args.api_key,
        api_url=args.api_url,
        model_name=args.model_name,
        image_dir=args.image_dir,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        save_path=args.save_path,
        eval_model_name=args.eval_model_name,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sample_n=args.sample_n,
        pixel_budget_per_image=args.pixel_budget_per_image,
    )
    
    logger.info(f"Global Config: {config}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}...")
    if config.dataset_name.startswith('HRBench') or config.dataset_name.startswith('hrbench'):
        dataset = HRBenchDataset(config.dataset_path, image_dir=config.image_dir)
    elif config.dataset_name == 'vstar_bench':
        dataset = VStarDataset(config.dataset_path)
    elif config.dataset_name == 'fixretina_sft':
        dataset = FixRetinaSFTDataset(config.dataset_path, image_dir=config.image_dir)
    elif config.dataset_name in ("mme_realworld_lite", "mmerealworldlite"):
        dataset = MMERealWorldLiteDataset(config.dataset_path, image_dir=config.image_dir)
    elif config.dataset_name in ('visualprobe', 'visualprobe_easy', 'visualprobe_medium', 'visualprobe_hard'):
        dataset = VisualProbeDataset(config.dataset_path)
    elif config.dataset_name in ('treebench', "TreeBench"):
        dataset = TreeBenchDataset(config.dataset_path)
    elif config.dataset_name in ('treevgr', 'TreeVGR', 'treevgr_rl'):
        dataset = TreeVGRRLDataset(config.dataset_path)
    else:
        logger.warning(f"Unsupported dataset: {config.dataset_name}. Fall back to naive load")
        dataset = GeneralVQADataset(config.dataset_path)
    
    # Load data
    logger.info("Loading dataset samples...")
    data = dataset.load_data()
    total_samples = len(data)
    logger.info(f"✓ Loaded {total_samples} samples from dataset")
    
    # Apply slicing
    start_index = args.start_index
    end_index = args.end_index
    
    if start_index < 0:
        start_index = 0
    if start_index >= total_samples:
        raise ValueError(f"start_index ({start_index}) is out of range (dataset has {total_samples} samples)")
    
    if end_index == -1:
        end_index = total_samples
    elif end_index > total_samples:
        end_index = total_samples
    
    if end_index <= start_index:
        raise ValueError(f"end_index ({end_index}) must be greater than start_index ({start_index})")
    
    data_slice = data[start_index:end_index]
    num_samples_in_slice = len(data_slice)
    
    logger.info(f"Processing slice: {num_samples_in_slice} samples (indices {start_index} to {end_index-1})")

    # Create inference agent
    if args.inference_agent == "fixretina":
        from inference_agent.fixretina import InferenceEngine
    elif args.inference_agent == "deepeyes":
        from inference_agent.deepeyes import InferenceEngine
    elif args.inference_agent == "pixelreasoner":
        from inference_agent.pixelreasoner import InferenceEngine
    elif args.inference_agent == "visionthink":
        from inference_agent.visionthink import InferenceEngine
    elif args.inference_agent == "adaptvision":
        from inference_agent.adaptvision import InferenceEngine
    elif args.inference_agent == "cof":
        from inference_agent.cof import InferenceEngine
    elif args.inference_agent == "minio3":
        from inference_agent.minio3 import InferenceEngine
    elif args.inference_agent == "qwenvl":
        from inference_agent.qwenvl import InferenceEngine
    else:
        raise ValueError(f"Unsupported inference agent: {args.inference_agent}")
    engine = InferenceEngine(config)
    
    # Determine output path
    output_path = config.get_inference_output_path()
    if start_index > 0 or end_index != total_samples:
        base_path = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'jsonl'
        output_path = f"{base_path}_{start_index}_{end_index-1}.{ext}"
    
    logger.info(f"Output path: {output_path}")
    
    # Load existing results if not regen
    existing_results = {}
    if not args.regen and os.path.exists(output_path):
        logger.info(f"Found existing results at {output_path}. Loading...")
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        existing_results[item['index']] = item
                    except json.JSONDecodeError:
                        logger.info(f"Warning: Skipping invalid JSON line")
            logger.info(f"Loaded {len(existing_results)} existing results.")
        except Exception as e:
            logger.info(f"Error loading existing results: {e}")
    
    # Filter samples to run
    samples_to_run = []
    for sample in data_slice:
        idx = sample['index']
        
        if args.regen:
            samples_to_run.append(sample)
            continue
            
        if idx not in existing_results:
            samples_to_run.append(sample)
        else:
            # Check if status indicates error
            # Assuming status can be a list (for sample_n > 1) or string
            status = existing_results[idx].get('status')
            
            should_rerun = False
            if isinstance(status, list):
                # if all samples in the batch failed, then rerun
                if all(s == 'error' for s in status):
                    should_rerun = True
            elif status == 'error':
                should_rerun = True
                
            if should_rerun:
                samples_to_run.append(sample)
            else:
                pass # Skip successfully processed sample
                
    if not samples_to_run:
        logger.info("All samples in this slice have been successfully processed.")
        return

    logger.info(f"Need to run inference on {len(samples_to_run)} samples.")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    await engine.infer_dataset(
        dataset,
        samples_to_run,
        output_path=output_path,
        num_workers=args.num_workers,
        save_step=args.save_step,
    )
    
    logger.info(f"✓ Successfully updated {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
