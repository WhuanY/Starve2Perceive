cd /map-vepfs/haozhe/yhwu/vlmpaper/train/rl/verl_starve
source /map-vepfs/miniconda3/envs/fix_rl/bin/activate

local_dir=/map-vepfs/haozhe/yhwu/vlmpaper/models/rl/qwen25vl3b/agent_vlagent/v2_noBC_datamix1_20260228025509/global_step_30/actor
target_dir=/map-vepfs/haozhe/yhwu/vlmpaper/models/rl/qwen25vl7b/20260228025509_v2_noBC
hf_model_path="/map-vepfs/models/Qwen2.5-VL-7B-Instruct"

# # Step 1: Run model merger to merge weights
python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path $hf_model_path \
    --local_dir $local_dir \
    --target_dir $target_dir

# Step 2: Copy tokenizer files from base model (model.save_pretrained doesn't copy tokenizer)
echo "=========================================="
echo "Copying tokenizer files from $hf_model_path to $target_dir..."
echo "=========================================="

# List of tokenizer and related files to copy
tokenizer_files=(
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "special_tokens_map.json"
    "chat_template.json"
    "preprocessor_config.json"
    "video_preprocessor_config.json"
    "added_tokens.json"
)

copied_count=0
for file in "${tokenizer_files[@]}"; do
    src_file="$hf_model_path/$file"
    if [ -f "$src_file" ]; then
        cp "$src_file" "$target_dir/"
        echo "  ✓ Copied $file"
        copied_count=$((copied_count + 1))
    else
        echo "  ⊘ Skipped $file (not found in source)"
    fi
done

echo "=========================================="
echo "Copied $copied_count tokenizer-related files"
echo "=========================================="

# Step 3: Update config.json to set torch_dtype to bfloat16 (weights are actually bfloat16)
if [ -f "$target_dir/config.json" ]; then
    echo ""
    echo "=========================================="
    echo "Updating config.json: torch_dtype -> bfloat16"
    echo "=========================================="
    python -c "
import json

config_path = '$target_dir/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Update torch_dtype to bfloat16
config['torch_dtype'] = 'bfloat16'
if 'quantization_config' in config and 'torch_dtype' in config['quantization_config']:
    config['quantization_config']['torch_dtype'] = 'bfloat16'

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print('✓ Updated config.json: torch_dtype -> bfloat16')
"
    echo "=========================================="
else
    echo "Warning: config.json not found in $target_dir"
fi

echo ""
echo "=========================================="
echo "Model merger completed successfully!"
echo "Target directory: $target_dir"
echo "=========================================="
