#!/bin/bash

# Run inference + judge for visualprobe_easy, visualprobe_medium, visualprobe_hard
# Usage: ./visualprobe_easy_medium_hard_infer_and_judge.sh <eval_model_name> [temperature] [sample_n] [--port PORT] [--pixel_budget N] [--inference_agent AGENT]

eval_model_name=""
temperature=""
sample_n=""
port=""
pixel_budget_per_image=""
inference_agent=""
pos_arg_count=0

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
                1) eval_model_name="$1" ;;
                2) temperature="$1" ;;
                3) sample_n="$1" ;;
            esac
            shift
            ;;
    esac
done

temperature="${temperature:-0.0}"
sample_n="${sample_n:-1}"
pixel_budget_per_image="${pixel_budget_per_image:-$((16384 * 28 * 28))}"
inference_agent="${inference_agent:-fixretina}"

if [ -z "$eval_model_name" ]; then
    echo "Usage: $0 <eval_model_name> [temperature] [sample_n] [--port PORT] [--pixel_budget N] [--inference_agent AGENT]"
    echo "Example: $0 'Qwen2.5-VL-7B' 0 1"
    echo "Example: $0 'Qwen2.5-VL-7B' 0 1 --pixel_budget 12845056"
    echo "Example: $0 'Qwen2.5-VL-7B' 0 1 --inference_agent deepeyes"
    exit 1
fi

# Load environment
source $(dirname $0)/.env

# Override SAVE_DIR when pixel_budget_per_image is 200704 (must be after .env which sets SAVE_DIR)
if [ "$pixel_budget_per_image" = "200704" ]; then
    SAVE_DIR="/map-vepfs/haozhe/yhwu/vlmpaper/data/eval_w_tool_budget_constrain_result"
    echo "Using pixel budget-constrained result dir"
fi

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
source $VENV_DIR/bin/activate

VISUALPROBE_BASE="$DATA_DIR/../rl/VisualProbe"
DATASETS="visualprobe_easy visualprobe_medium visualprobe_hard"

if [ -n "$port" ]; then
    api_url="http://localhost:${port}/v1"
else
    api_url="http://localhost:9753/v1"
fi
api_key="none"

for dataset_name in $DATASETS; do
    echo ""
    echo "=========================================="
    echo "INFERENCE + JUDGE: $dataset_name"
    echo "=========================================="
    echo "Model:       $eval_model_name"
    echo "Temperature: $temperature"
    echo "Sample N:    $sample_n"
    echo "Pixel Budget: $pixel_budget_per_image"
    echo "Inference Agent: $inference_agent"
    echo "API URL:     $api_url"
    echo "Save Dir:    $SAVE_DIR"
    echo "=========================================="

    case "$dataset_name" in
        "visualprobe_easy")  dataset_path="$VISUALPROBE_BASE/VisualProbe_Easy" ;;
        "visualprobe_medium") dataset_path="$VISUALPROBE_BASE/VisualProbe_Medium" ;;
        "visualprobe_hard")   dataset_path="$VISUALPROBE_BASE/VisualProbe_Hard" ;;
        *) echo "Unknown dataset: $dataset_name"; exit 1 ;;
    esac

    mkdir -p "$SAVE_DIR/$eval_model_name/$dataset_name/"

    # Inference
    python run_inference.py \
        --dataset_path "$dataset_path" \
        --dataset_name "$dataset_name" \
        --start_index 0 \
        --end_index -1 \
        --save_path "$SAVE_DIR/$eval_model_name/$dataset_name/" \
        --inference_agent "$inference_agent" \
        --pixel_budget_per_image "$pixel_budget_per_image" \
        --eval_model_name "$eval_model_name" \
        --model_name "$eval_model_name" \
        --temperature "$temperature" \
        --sample_n "$sample_n" \
        --api_key "$api_key" \
        --api_url "$api_url" \
        --save_step 10 \
        2>&1 | tee "$SAVE_DIR/$eval_model_name/$dataset_name/inference.log"

    if [ $? -ne 0 ]; then
        echo "Error: Inference failed for $dataset_name"
        exit 1
    fi

    # Judge (use visualprobe evaluator for all three)
    results_path="$SAVE_DIR/${eval_model_name}/${dataset_name}/${eval_model_name}/${dataset_name}/${dataset_name}_${eval_model_name}_n${sample_n}_temperature${temperature}.jsonl"
    output_path="${results_path%.jsonl}_judged.jsonl"
    results_dir=$(dirname "$results_path")

    if [ ! -f "$results_path" ]; then
        alt_results_path="$SAVE_DIR/${eval_model_name}/${dataset_name}/${dataset_name}_${eval_model_name}_n${sample_n}_temperature${temperature}.0.jsonl"
        if [ -f "$alt_results_path" ]; then
            results_path="$alt_results_path"
            output_path="${results_path%.jsonl}_judged.jsonl"
        else
            echo "Error: Inference results not found for $dataset_name"
            exit 1
        fi
    fi

    judge_model_name="gpt-4o-mini"
    judge_api_key=$OPENAI_API_KEY
    judge_api_url=$OPENAI_API_URL

    python run_evaluation.py \
        --results_path "$results_path" \
        --output_path "$output_path" \
        --dataset_name "visualprobe" \
        --n "$sample_n" \
        --count_visual_tokens \
        --base_url "$judge_api_url" \
        --openai_api_key "$judge_api_key" \
        --judge_model_name "$judge_model_name" \
        2>&1 | tee "$results_dir/evaluation.log"

    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for $dataset_name"
        exit 1
    fi

    echo "✓ Complete: $dataset_name"
done

echo ""
echo "=========================================="
echo "ALL DONE: visualprobe_easy, visualprobe_medium, visualprobe_hard"
echo "=========================================="
