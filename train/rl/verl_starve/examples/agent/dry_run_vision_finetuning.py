#!/usr/bin/env python3
"""
Dry-run script to verify if vision encoder parameters are trainable in PPO training.
This script mimics the exact model initialization and optimizer setup from the training script.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch import optim
from transformers import AutoConfig, AutoModelForVision2Seq
import warnings

# Add verl to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.model import get_generation_config, update_model_config
from verl.utils.torch_dtypes import PrecisionType


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def find_vision_encoder_params(model):
    """Find vision encoder parameters in the model."""
    vision_params = {}
    for name, param in model.named_parameters():
        # Common vision encoder parameter names in Qwen2.5-VL
        if any(keyword in name.lower() for keyword in ['vision', 'visual', 'image_processor', 'image_encoder']):
            vision_params[name] = param
    return vision_params


def analyze_vision_params(model, optimizer, rank=0):
    """Analyze vision encoder parameters to check if they are trainable."""
    if rank != 0:
        return
    
    print("\n" + "="*80)
    print("VISION ENCODER PARAMETER ANALYSIS")
    print("="*80)
    
    # Find vision encoder parameters
    vision_params = find_vision_encoder_params(model)
    
    print(f"\nFound {len(vision_params)} vision encoder parameters:")
    for name in list(vision_params.keys())[:10]:  # Show first 10
        print(f"  - {name}")
    if len(vision_params) > 10:
        print(f"  ... and {len(vision_params) - 10} more")
    
    # Check requires_grad status
    trainable_vision_params = {name: param for name, param in vision_params.items() if param.requires_grad}
    frozen_vision_params = {name: param for name, param in vision_params.items() if not param.requires_grad}
    
    print(f"\nVision encoder parameters status:")
    print(f"  - Trainable (requires_grad=True): {len(trainable_vision_params)}")
    print(f"  - Frozen (requires_grad=False): {len(frozen_vision_params)}")
    
    # Count parameters
    trainable_count = sum(p.numel() for p in trainable_vision_params.values())
    frozen_count = sum(p.numel() for p in frozen_vision_params.values())
    total_count = sum(p.numel() for p in vision_params.values())
    
    print(f"\nVision encoder parameter counts:")
    print(f"  - Trainable: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)")
    print(f"  - Frozen: {frozen_count:,} ({frozen_count/total_count*100:.2f}%)")
    print(f"  - Total: {total_count:,}")
    
    # Check optimizer inclusion
    optimizer_param_ids = {id(p) for param_group in optimizer.param_groups for p in param_group['params']}
    vision_in_optimizer = sum(1 for p in vision_params.values() if id(p) in optimizer_param_ids)
    
    print(f"\nVision encoder parameters in optimizer:")
    print(f"  - In optimizer: {vision_in_optimizer}/{len(vision_params)}")
    print(f"  - Percentage: {vision_in_optimizer/len(vision_params)*100:.2f}%")
    
    # Show some examples
    if trainable_vision_params:
        print(f"\nExample trainable vision parameters:")
        for name in list(trainable_vision_params.keys())[:5]:
            param = trainable_vision_params[name]
            print(f"  - {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    if frozen_vision_params:
        print(f"\nExample frozen vision parameters:")
        for name in list(frozen_vision_params.keys())[:5]:
            param = frozen_vision_params[name]
            print(f"  - {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    if len(trainable_vision_params) > 0 and vision_in_optimizer > 0:
        print("✓ Vision encoder parameters ARE trainable and included in optimizer")
        print("✓ Vision module WILL be fine-tuned during PPO training")
    else:
        print("✗ Vision encoder parameters are NOT trainable or NOT in optimizer")
        print("✗ Vision module will NOT be fine-tuned during PPO training")
    print("="*80 + "\n")


def main():
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("="*80)
        print("DRY-RUN: Checking if vision encoder is trainable in PPO training")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Model path: {os.environ.get('REF_MODEL_PATH', '/map-vepfs/haozhe/yhwu/vlmpaper/models/sft/qwen25vl-7b/fixretina-v1_merged/checkpoint-555')}")
    
    # Model path from run.sh
    model_path = os.environ.get(
        'REF_MODEL_PATH', 
        '/map-vepfs/haozhe/yhwu/vlmpaper/models/sft/qwen25vl-7b/fixretina-v1_merged/checkpoint-555'
    )
    
    # Setup device mesh (same as training script)
    device_mesh = create_device_mesh(world_size=world_size, fsdp_size=-1)
    
    # Copy model to local if needed
    local_path = copy_to_local(model_path)
    
    # Load tokenizer and processor
    tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    processor = hf_processor(local_path, trust_remote_code=True)
    
    # Model config (same as training script)
    torch_dtype = torch.float32  # Actor uses float32
    actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
    
    override_config_kwargs = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
    
    if rank == 0:
        print(f"Model config: {actor_model_config}")
    
    # Initialize model (same as training script)
    init_context = get_init_weight_context_manager(
        use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=device_mesh
    )
    
    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
            actor_module_class = AutoModelForVision2Seq
        else:
            raise ValueError("Model is not a vision model!")
        
        actor_module = actor_module_class.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=actor_model_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        
        actor_module.to(torch_dtype)
        actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    dist.barrier()
    
    # Wrap with FSDP (same as training script)
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32
    )
    
    auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config={"min_num_params": 0})
    
    sharding_strategy = get_sharding_strategy(device_mesh)
    
    actor_module_fsdp = FSDP(
        actor_module,
        cpu_offload=None,  # Actor doesn't use CPU offload
        param_init_fn=init_fn,
        use_orig_params=False,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_mesh=device_mesh,
        forward_prefetch=False,
    )
    
    # Create optimizer (same as training script)
    optim_config = type('obj', (object,), {
        'lr': 1e-6,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2
    })()
    
    actor_optimizer = optim.AdamW(
        actor_module_fsdp.parameters(),  # This is the key line - includes ALL parameters
        lr=optim_config.lr,
        betas=optim_config.betas,
        weight_decay=optim_config.weight_decay,
    )
    
    if rank == 0:
        print(f"\nOptimizer created with {len(list(actor_module_fsdp.parameters()))} parameter groups")
    
    # Analyze vision parameters
    # Note: We need to access the underlying model to check parameters
    # FSDP wraps the model, so we need to unwrap it for analysis
    if rank == 0:
        # Analyze vision parameters using original model (before FSDP wrapping)
        analyze_vision_params_fsdp(actor_module_fsdp, actor_optimizer, actor_module, rank)
    
    dist.barrier()
    
    if rank == 0:
        print("\nDry-run completed successfully!")

# Additional function to analyze FSDP-wrapped model
def analyze_vision_params_fsdp(fsdp_model, optimizer, original_model, rank=0):
    """Analyze vision encoder parameters in FSDP-wrapped model."""
    if rank != 0:
        return
    
    print("\n" + "="*80)
    print("VISION ENCODER PARAMETER ANALYSIS (FSDP-wrapped model)")
    print("="*80)
    
    # First, analyze the original model before FSDP wrapping
    vision_params_orig = find_vision_encoder_params(original_model)
    
    print(f"\nFound {len(vision_params_orig)} vision encoder parameters in original model:")
    for name in list(vision_params_orig.keys())[:10]:
        print(f"  - {name}")
    if len(vision_params_orig) > 10:
        print(f"  ... and {len(vision_params_orig) - 10} more")
    
    # Check requires_grad status in original model
    trainable_vision_params_orig = {name: param for name, param in vision_params_orig.items() if param.requires_grad}
    
    print(f"\nOriginal model vision encoder parameters:")
    print(f"  - Trainable (requires_grad=True): {len(trainable_vision_params_orig)}")
    print(f"  - Frozen (requires_grad=False): {len(vision_params_orig) - len(trainable_vision_params_orig)}")
    
    # Get all parameters from FSDP model
    fsdp_param_ids = {id(p) for p in fsdp_model.parameters()}
    optimizer_param_ids = {id(p) for param_group in optimizer.param_groups for p in param_group['params']}
    
    # Match vision parameters with optimizer
    vision_in_optimizer = 0
    vision_trainable_in_optimizer = 0
    
    for name, param in vision_params_orig.items():
        param_id = id(param)
        if param_id in optimizer_param_ids:
            vision_in_optimizer += 1
            if param.requires_grad:
                vision_trainable_in_optimizer += 1
    
    print(f"\nVision encoder parameters in optimizer:")
    print(f"  - Total in optimizer: {vision_in_optimizer}/{len(vision_params_orig)}")
    print(f"  - Trainable in optimizer: {vision_trainable_in_optimizer}/{len(trainable_vision_params_orig)}")
    
    # Count parameters
    trainable_count = sum(p.numel() for p in trainable_vision_params_orig.values())
    total_count = sum(p.numel() for p in vision_params_orig.values())
    
    print(f"\nVision encoder parameter counts:")
    print(f"  - Trainable: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)")
    print(f"  - Total: {total_count:,}")
    
    # Show some examples
    if trainable_vision_params_orig:
        print(f"\nExample trainable vision parameters:")
        for name in list(trainable_vision_params_orig.keys())[:5]:
            param = trainable_vision_params_orig[name]
            in_opt = "✓" if id(param) in optimizer_param_ids else "✗"
            print(f"  {in_opt} {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    if len(trainable_vision_params_orig) > 0 and vision_in_optimizer > 0:
        print("✓ Vision encoder parameters ARE trainable and included in optimizer")
        print("✓ Vision module WILL be fine-tuned during PPO training")
        print(f"  ({vision_in_optimizer}/{len(vision_params_orig)} vision params in optimizer)")
    else:
        print("✗ Vision encoder parameters are NOT trainable or NOT in optimizer")
        print("✗ Vision module will NOT be fine-tuned during PPO training")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


