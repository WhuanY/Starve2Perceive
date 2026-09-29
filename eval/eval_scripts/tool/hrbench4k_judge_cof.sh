source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name=HRBench4K
eval_model_name="CoF-rl-model-7b"
model_name="CoF-rl-model-7b"
sample_n=1
temperature=0

# Best-of-N evaluation
bon_n=1

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