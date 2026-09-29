source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config: must match treevgr_infer.sh (model_name, dataset_name, sample_n, temperature)
dataset_name="treevgr"
eval_model_name="qwen25vl7b_fixretina_v1_sft555"
model_name="$eval_model_name"
sample_n=8
temperature=1.0

# Judge configuration
judge_model_name="judge"
judge_api_url="http://localhost:8000/v1"
judge_api_key="none"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

# Inference results path (same pattern as run_inference.py output)
results_path=$SAVE_DIR/${model_name}/${dataset_name}/${model_name}/${dataset_name}/${dataset_name}_${model_name}_n${sample_n}_temperature${temperature}.jsonl
output_path="${results_path%.jsonl}_judged.jsonl"
results_dir=$(dirname $results_path)

if [ ! -f "$results_path" ]; then
    echo "Error: Inference results file not found: $results_path"
    echo "Please run inference first using treevgr_infer.sh"
    exit 1
fi

echo "Evaluating results from: $results_path"

python run_evaluation.py \
    --results_path "$results_path" \
    --output_path "$output_path" \
    --dataset_name "$dataset_name" \
    --n $sample_n \
    --count_visual_tokens \
    --base_url "$judge_api_url" \
    --openai_api_key "$judge_api_key" \
    --judge_model_name "$judge_model_name" \
    2>&1 | tee $results_dir/evaluation.log

echo "Evaluation complete. Results saved to: $output_path"
echo "Log saved to: $results_dir/evaluation.log"
