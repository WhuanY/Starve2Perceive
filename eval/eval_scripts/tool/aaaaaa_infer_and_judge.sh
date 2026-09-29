#!/bin/bash

# Higher-order script for running inference + judge sequentially
# Usage: ./infer_and_judge.sh <eval_model_name> <dataset_name> [temperature] [sample_n] [--port PORT] [--pixel_budget N] [--inference_agent AGENT]
#
# Examples:
#   ./infer_and_judge.sh "Qwen2.5-VL-32B-Instruct" "vstar_bench"
#   ./infer_and_judge.sh "Qwen2.5-VL-3B-Instruct" "fixretina_sft" 0.3 4
#   ./infer_and_judge.sh "Qwen2.5-VL-7B" "all"   # run hrbench4k,hrbench8k,mmerealworldlite,realworldqa,vstar_bench,treebench,visualprobe_easy/medium/hard
#   ./infer_and_judge.sh "Qwen2.5-VL-7B" "vstar_bench" 0 1 --port 8080
#   ./infer_and_judge.sh "Qwen2.5-VL-7B" "vstar_bench" 0 1 --pixel_budget 12845056
#   ./infer_and_judge.sh "Qwen2.5-VL-7B" "vstar_bench" 0 1 --inference_agent deepeyes
#   ./infer_and_judge.sh "Qwen2.5-VL-7B" "visualprobe" 0 1   # run visualprobe_easy,visualprobe_medium,visualprobe_hard

# Parse arguments
eval_model_name=""
dataset_name=""
temperature=""
sample_n=""
port=""
pixel_budget_per_image=""
inference_agent=""
pos_arg_count=0

# Parse positional and named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            port="$2"
            shift 2
            ;;
        --pixel_budget)
            pixel_budget_per_image="$2"
            shift 2
            ;;
        --inference_agent)
            inference_agent="$2"
            shift 2
            ;;
        *)
            pos_arg_count=$((pos_arg_count + 1))
            case $pos_arg_count in
                1)
                    eval_model_name="$1"
                    ;;
                2)
                    dataset_name="$1"
                    ;;
                3)
                    temperature="$1"
                    ;;
                4)
                    sample_n="$1"
                    ;;
            esac
            shift
            ;;
    esac
done

# Set defaults for optional arguments
temperature="${temperature:-0.0}"
sample_n="${sample_n:-1}"
pixel_budget_per_image="${pixel_budget_per_image:-$((16384 * 28 * 28))}"
inference_agent="${inference_agent:-fixretina}"

# Check required arguments
if [ -z "$eval_model_name" ] || [ -z "$dataset_name" ]; then
    echo "Usage: $0 <eval_model_name> <dataset_name> [temperature] [sample_n] [--port PORT] [--pixel_budget N] [--inference_agent AGENT]"
    echo ""
    echo "Examples:"
    echo "  $0 'Qwen2.5-VL-32B-Instruct' 'vstar_bench'"
    echo "  $0 'Qwen2.5-VL-3B-Instruct' 'fixretina_sft' 0.3 4"
    echo "  $0 'Qwen2.5-VL-7B' 'vstar_bench' 0 1 --port 8080"
    echo "  $0 'Qwen2.5-VL-7B' 'vstar_bench' 0 1 --pixel_budget 12845056"
    echo "  $0 'Qwen2.5-VL-7B' 'vstar_bench' 0 1 --inference_agent deepeyes"
    echo ""
    echo "Available datasets: all, vstar_bench, fixretina_sft, hrbench4k, hrbench8k, mmerealworldlite, realworldqa, treebench, visualprobe_easy, visualprobe_medium, visualprobe_hard, etc."
    exit 1
fi

if [ "$dataset_name" = "all" ]; then
    DATASETS="hrbench4k hrbench8k mmerealworldlite realworldqa vstar_bench treebench visualprobe_easy visualprobe_medium visualprobe_hard"
elif [ "$dataset_name" = "visualprobe" ]; then
    DATASETS="visualprobe_easy visualprobe_medium visualprobe_hard"
else
    DATASETS="$dataset_name"
fi

# Load environment
source $(dirname $0)/.env

# Per pixel-budget sweep (must be after .env which sets SAVE_DIR)
SAVE_DIR="/map-vepfs/haozhe/yhwu/vlmpaper/data/eval_w_tool_bc${pixel_budget_per_image}_masked" # exp: masked return region
echo "Using SAVE_DIR=${SAVE_DIR}"

# Set up directories
base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir

# Activate virtual environment
source $VENV_DIR/bin/activate


for dataset_name in $DATASETS; do
    echo ""
    echo "=========================================="
    echo "INFERENCE + JUDGE PIPELINE"
    echo "=========================================="
    echo "Model:       $eval_model_name"
    echo "Dataset:     $dataset_name"
    echo "Temperature: $temperature"
    echo "Sample N:    $sample_n"
    echo "Pixel Budget: $pixel_budget_per_image"
    echo "Inference Agent: $inference_agent"
    echo "API URL:     $api_url"
    echo "Data Dir:    $DATA_DIR"
    echo "Save Dir:    $SAVE_DIR"
    echo "=========================================="

    # Log suffix to avoid overwriting across runs
    log_suffix="temp${temperature}n${sample_n}"

    # ============================================
    # STEP 1: INFERENCE
    # ============================================
    echo ""
    echo "=========================================="
    echo "STEP 1/2: Running Inference ($dataset_name)"
    echo "=========================================="

    # Inference configuration
    api_key="none"
    if [ -n "$port" ]; then
        api_url="http://localhost:${port}/v1"
    else
        api_url="http://localhost:9753/v1"
    fi

    # Determine dataset path based on dataset name
    case "$dataset_name" in
        "fixretina_sft")
            dataset_path="$DATA_DIR/FixRetina_SFT/sft"
            ;;
        "vstar_bench")
            dataset_path="$DATA_DIR/vstar_bench"
            ;;
        "hrbench4k")
            dataset_path="$DATA_DIR/hrbench/hr_bench_4k.tsv"
            ;;
        "hrbench8k")
            dataset_path="$DATA_DIR/hrbench/hr_bench_8k.tsv"
            ;;
        "mmerealworld")
            dataset_path="$DATA_DIR/mmerealworld"
            ;;
        "mmerealworldlite")
            dataset_path="$DATA_DIR/mmerealworldlite"
            ;;
        "realworldqa")
            dataset_path="$DATA_DIR/realworldqa"
            ;;
        "treebench")
            dataset_path="$DATA_DIR/TreeBench"
            ;;
        "visualprobe_easy")
            VISUALPROBE_BASE="$DATA_DIR/../rl/VisualProbe"
            dataset_path="$VISUALPROBE_BASE/VisualProbe_Easy"
            ;;
        "visualprobe_medium")
            VISUALPROBE_BASE="$DATA_DIR/../rl/VisualProbe"
            dataset_path="$VISUALPROBE_BASE/VisualProbe_Medium"
            ;;
        "visualprobe_hard")
            VISUALPROBE_BASE="$DATA_DIR/../rl/VisualProbe"
            dataset_path="$VISUALPROBE_BASE/VisualProbe_Hard"
            ;;
        *)
            # Default: assume dataset_name is a subdirectory in DATA_DIR
            dataset_path="$DATA_DIR/$dataset_name"
            ;;
    esac

    # Create output directory
    mkdir -p "$SAVE_DIR/$eval_model_name/$dataset_name/"

    # Run inference
    python run_inference.py \
        --dataset_path "$dataset_path" \
        --dataset_name "$dataset_name" \
        --start_index 0 \
        --end_index -1 \
        --save_path "$SAVE_DIR/$eval_model_name/$dataset_name/" \
        --inference_agent "$inference_agent" \
        --pixel_budget_per_image "$pixel_budget_per_image" \
        --save_step 10 \
        --eval_model_name "$eval_model_name" \
        --model_name "$eval_model_name" \
        --temperature "$temperature" \
        --sample_n "$sample_n" \
        --api_key "$api_key" \
        --api_url "$api_url" \
        2>&1 | tee "$SAVE_DIR/$eval_model_name/$dataset_name/inference_${log_suffix}.log"

    # Check if inference was successful
    if [ $? -ne 0 ]; then
        echo "Error: Inference failed. Check logs at:"
        echo "  $SAVE_DIR/$eval_model_name/$dataset_name/inference_${log_suffix}.log"
        exit 1
    fi

    echo ""
    echo "✓ Inference complete"

    # ============================================
    # STEP 2: JUDGE
    # ============================================
    echo ""
    echo "=========================================="
    echo "STEP 2/2: Running Evaluation/Judge"
    echo "=========================================="

    # Judge configuration
    judge_model_name="gpt-4o-mini"
    judge_api_key="${OPENAI_API_KEY}"
    judge_api_url="https://aigc.x-see.cn/v1"
    bon_n="$sample_n" 

    # Construct inference results path
    results_path="$SAVE_DIR/${eval_model_name}/${dataset_name}/${eval_model_name}/${dataset_name}/${dataset_name}_${eval_model_name}_n${sample_n}_temperature${temperature}.jsonl"
    output_path="${results_path%.jsonl}_judged.jsonl"
    results_dir=$(dirname "$results_path")

    # Check if results file exists
    if [ ! -f "$results_path" ]; then
        echo "Error: Inference results file not found: $results_path"
        echo "Expected location might be different. Searching..."
        
        # Try alternative path patterns
        alt_results_path="$SAVE_DIR/${eval_model_name}/${dataset_name}/${dataset_name}_${eval_model_name}_n${sample_n}_temperature${temperature}.0.jsonl"
        
        if [ -f "$alt_results_path" ]; then
            results_path="$alt_results_path"
            output_path="${results_path%.jsonl}_judged.jsonl"
            echo "Found results at: $results_path"
        else
            echo "Cannot find inference results. Please check the save path."
            exit 1
        fi
    fi

    echo "Evaluating results from: $results_path"
    echo "Output path: $output_path"

    # Run evaluation with optional filter_good_traj for fixretina_sft
    if [ "$dataset_name" = "fixretina_sft" ]; then
        filter_flag="--filter_good_traj"
    else
        filter_flag=""
    fi

    # For visualprobe datasets, use "visualprobe" as dataset_name for evaluation
    eval_dataset_name="$dataset_name"
    if [[ "$dataset_name" =~ ^visualprobe_(easy|medium|hard)$ ]]; then
        eval_dataset_name="visualprobe"
    fi

    python run_evaluation.py \
        --results_path "$results_path" \
        --output_path "$output_path" \
        --dataset_name "$eval_dataset_name" \
        --n "$bon_n" \
        --count_visual_tokens \
        $filter_flag \
        --base_url "$judge_api_url" \
        --openai_api_key "$judge_api_key" \
        --judge_model_name "$judge_model_name" \
        2>&1 | tee "$results_dir/evaluation_${log_suffix}.log"

    # Check if evaluation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed. Check logs at:"
        echo "  $results_dir/evaluation_${log_suffix}.log"
        exit 1
    fi

    echo ""
    echo "✓ Evaluation complete ($dataset_name)"

    # ============================================
    # SUMMARY (per dataset)
    # ============================================
    echo ""
    echo "=========================================="
    echo "PIPELINE COMPLETE: $dataset_name"
    echo "=========================================="
    echo "Model:             $eval_model_name"
    echo "Dataset:           $dataset_name"
    echo "Inference results: $results_path"
    echo "Judged results:    $output_path"
    echo "Inference log:     $SAVE_DIR/$eval_model_name/$dataset_name/inference_${log_suffix}.log"
    echo "Evaluation log:    $results_dir/evaluation_${log_suffix}.log"
    echo "=========================================="

done