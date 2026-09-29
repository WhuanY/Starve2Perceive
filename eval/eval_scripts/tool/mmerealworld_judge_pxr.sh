source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name=mme_realworld_lite
# eval_model_name="Qwen2.5-VL-3B-Instruct"
# model_name="Qwen2.5-VL-3B-Instruct"
eval_model_name="PixelReasoner-RL-v1"
model_name="PixelReasoner-RL-v1"
sample_n=1
temperature=0

# Judge configuration (optional, for LLM-based judging)
judge_model_name="gpt-4o-mini"
judge_api_key=${OPENAI_API_KEY}
judge_api_url=${OPENAI_API_URL}

# Best-of-N evaluation
bon_n=1

# Filter good trajectories (optional)
# filter_good_traj="--filter_good_traj"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

results_path=$SAVE_DIR/${model_name}/${dataset_name}/${model_name}/${dataset_name}/${dataset_name}_${model_name}_n1_temperature0.0.jsonl
output_path="${results_path%.jsonl}_judged.jsonl"

# Check if results file exists
if [ ! -f "$results_path" ]; then
    echo "Error: Inference results file not found: $results_path"
    exit 1
fi

echo "Evaluating results from: $results_path"

output_path=$results_path  # Overwrite by default

# Run evaluation
python run_evaluation.py \
    --results_path "$results_path" \
    --output_path "${output_path}" \
    --dataset_name "$dataset_name" \
    --n $bon_n \
    --count_visual_tokens \
    ${filter_good_traj:-} \
    2>&1 | tee $output_path/evaluation.log

echo "Evaluation complete. Results saved to: $output_path"
echo "Log saved to: $output_path/evaluation.log"