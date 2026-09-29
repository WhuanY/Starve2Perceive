source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name=fixretina_sft
eval_model_name="Qwen2.5-VL-3B-Instruct"
model_name="Qwen2.5-VL-3B-Instruct"  # Used for result filename matching
sample_n=1
temperature=0.0

# Judge configuration (optional, for LLM-based judging)
judge_model_name="gpt-4o-mini"
judge_api_key="${OPENAI_API_KEY}"
judge_api_url="https://open.xiaojingai.com/v1"

# Best-of-N evaluation
bon_n=1

# Filter good trajectories (optional)
# filter_good_traj="--filter_good_traj"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

result_dir=/home/ywuit/vlmpaper/data/eval_w_tool_result/$eval_model_name/$dataset_name
results_path=${result_dir}/${eval_model_name}/${dataset_name}/${dataset_name}_${eval_model_name}_n${sample_n}_temperature${temperature}.jsonl

# Check if results file exists
if [ ! -f "$results_path" ]; then
    echo "Error: Inference results file not found: $results_path"
    echo "Please run inference first using fixretina_sft_infer.sh"
    exit 1
fi

echo "Evaluating results from: $results_path"

# Output path: replace .jsonl with _judged.jsonl
# output_path="${results_path%.jsonl}_judged.jsonl"
results_path="/home/ywuit/vlmpaper/data/eval_w_tool_result/Qwen2.5-VL-32B-Instruct/fixretina_sft/Qwen2.5-VL-32B-Instruct/fixretina_sft/fixretina_sft_Qwen2.5-VL-32B-Instruct_n4_temperature0.3_judged.jsonl"
output_path="/home/ywuit/vlmpaper/data/eval_w_tool_result/Qwen2.5-VL-32B-Instruct/fixretina_sft/Qwen2.5-VL-32B-Instruct/fixretina_sft/fixretina_sft_Qwen2.5-VL-32B-Instruct_n4_temperature0.3_judged.jsonl"
echo "Output path: $output_path"
# Run evaluation
python run_evaluation.py \
    --results_path "$results_path" \
    --output_path "$output_path" \
    --dataset_name "$dataset_name" \
    --n $bon_n \
    --filter_good_traj \
    --base_url "$judge_api_url" \
    --openai_api_key "$judge_api_key" \
    --judge_model_name "$judge_model_name" \
    2>&1 | tee ${results_dir}/evaluation.log

echo "Evaluation complete. Results saved to: ${output_path}"
echo "Log saved to: ${results_dir}/evaluation.log"
