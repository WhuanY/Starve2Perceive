source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name=vstar_bench
eval_model_name="AdaptVision-7B"
model_name="AdaptVision-7B"  # Used for result filename matching
sample_n=1
temperature=0

# Judge configuration (optional, for LLM-based judging)
judge_model_name="gpt-4o-mini"
judge_api_key=$OPENAI_API_KEY
judge_api_url=$OPENAI_API_URL

# Best-of-N evaluation
bon_n=1

# Filter good trajectories (optional)
# filter_good_traj="--filter_good_traj"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

# Construct inference results path (matching the pattern from run_inference.py)
results_path=$SAVE_DIR/${model_name}/${dataset_name}/${model_name}/${dataset_name}/${dataset_name}_${model_name}_n1_temperature0.0.jsonl


output_path="${results_path%.jsonl}_judged.jsonl"

# Check if results file exists
if [ ! -f "$results_path" ]; then
    echo "Error: Inference results file not found: $results_path"
    echo "Please run inference first using vstar_infer_adaptvision.sh"
    exit 1
fi

echo "Evaluating results from: $results_path"
results_dir=$(dirname $results_path)
mkdir -p $results_dir
# Run evaluation
python run_evaluation.py \
    --results_path "$results_path" \
    --output_path "$output_path" \
    --dataset_name "$dataset_name" \
    --n $bon_n \
    --count_visual_tokens \
    ${filter_good_traj:-} \
    ${judge_base_url:+--base_url "$judge_base_url"} \
    ${judge_api_key:+--openai_api_key "$judge_api_key"} \
    ${judge_model_name:+--judge_model_name "$judge_model_name"} \
    2>&1 | tee $results_dir/evaluation.log

echo "Evaluation complete. Results saved to: $output_path"
echo "Log saved to: $results_dir/evaluation.log"

