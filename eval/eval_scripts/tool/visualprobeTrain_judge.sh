source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name=visualprobe
eval_model_name="qwen25vl7b_fixretina_v1_sft555"
model_name="qwen25vl7b_fixretina_v1_sft555"  # Used for result filename matching
sample_n=8
temperature=1.0

# Judge configuration (optional, for LLM-based judging)
judge_model_name="judge"
judge_api_key="none"
judge_api_url="http://localhost:8000/v1"

# Best-of-N evaluation
bon_n=8

# Filter good trajectories (optional)
# filter_good_traj="--filter_good_traj"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

# Construct inference results path (matching the pattern from run_inference.py)
results_path=$SAVE_DIR/${model_name}/${dataset_name}/${model_name}/${dataset_name}/${dataset_name}_${model_name}_n${sample_n}_temperature${temperature}.jsonl
output_path="${results_path%.jsonl}_judged.jsonl"
results_dir=$(dirname $results_path)

# Check if results file exists
if [ ! -f "$results_path" ]; then
    echo "Error: Inference results file not found: $results_path"
    echo "Please run inference first using visualprobeTrain_infer.sh"
    exit 1
fi

echo "Evaluating results from: $results_path"


# Run evaluation
python run_evaluation.py \
    --results_path "$results_path" \
    --output_path "$output_path" \
    --dataset_name "$dataset_name" \
    --n $bon_n \
    --count_visual_tokens \
    ${filter_good_traj:-} \
    ${judge_api_url:+--base_url "$judge_api_url"} \
    ${judge_api_key:+--openai_api_key "$judge_api_key"} \
    ${judge_model_name:+--judge_model_name "$judge_model_name"} \
    2>&1 | tee $results_dir/evaluation.log

echo "Evaluation complete. Results saved to: $output_path"
echo "Log saved to: $results_dir/evaluation.log"
