#!/bin/bash

source $(dirname $0)/.env  # 全局信息

# Override SAVE_DIR to use budget_constrain_result
SAVE_DIR="/map-vepfs/haozhe/yhwu/vlmpaper/data/eval_w_tool_budget_constrain_result"

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
eval_model_name="qwen25vl7b_inst_bc_decay_rl_20260220073555_global_step_30"
model_name="qwen25vl7b_inst_bc_decay_rl_20260220073555_global_step_30"  # Used for result filename matching
sample_n=1
temperature=0.0

# Judge configuration (optional, for LLM-based judging)
judge_model_name="gpt-4o-mini"
judge_api_key="${OPENAI_API_KEY}"
judge_api_url="https://aigc.x-see.cn/v1"

# Best-of-N evaluation
bon_n=1

# Filter good trajectories (optional)
# filter_good_traj="--filter_good_traj"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

# List of all benchmarks in budget_constrain_result
DATASETS="hrbench4k hrbench8k mmerealworldlite realworldqa treebench visualprobe_easy visualprobe_medium visualprobe_hard vstar_bench"

for dataset_name in $DATASETS; do
    echo ""
    echo "=========================================="
    echo "JUDGING: $dataset_name"
    echo "=========================================="
    echo "Model:       $model_name"
    echo "Dataset:     $dataset_name"
    echo "Temperature: $temperature"
    echo "Sample N:    $sample_n"
    echo "Save Dir:    $SAVE_DIR"
    echo "=========================================="

    # Construct inference results path (matching the pattern from run_inference.py)
    results_path=$SAVE_DIR/${model_name}/${dataset_name}/${model_name}/${dataset_name}/${dataset_name}_${model_name}_n1_temperature0.0.jsonl
    output_path="${results_path%.jsonl}_judged.jsonl"
    results_dir=$(dirname "$results_path")

    # Check if results file exists
    if [ ! -f "$results_path" ]; then
        echo "Warning: Inference results file not found: $results_path"
        echo "Skipping $dataset_name..."
        continue
    fi

    echo "Evaluating results from: $results_path"
    echo "Output path: $output_path"

    # For visualprobe datasets, use "visualprobe" as dataset_name for evaluation
    eval_dataset_name="$dataset_name"
    if [[ "$dataset_name" =~ ^visualprobe_(easy|medium|hard)$ ]]; then
        eval_dataset_name="visualprobe"
    fi

    # Run evaluation (NOTE: NOT using --count_visual_tokens since qwen25vl7b doesn't have crop_token)
    python run_evaluation.py \
        --results_path "$results_path" \
        --output_path "$output_path" \
        --dataset_name "$eval_dataset_name" \
        --n $bon_n \
        ${filter_good_traj:-} \
        ${judge_api_url:+--base_url "$judge_api_url"} \
        ${judge_api_key:+--openai_api_key "$judge_api_key"} \
        ${judge_model_name:+--judge_model_name "$judge_model_name"} \
        2>&1 | tee $results_dir/evaluation_temp0.0n1.log

    if [ $? -eq 0 ]; then
        echo "✓ Evaluation complete for $dataset_name"
        echo "Results saved to: $output_path"
        echo "Log saved to: $results_dir/evaluation_temp0.0n1.log"
    else
        echo "✗ Evaluation failed for $dataset_name"
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
