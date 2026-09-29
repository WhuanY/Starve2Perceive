source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name=fixretina_sft
# eval_model_name="gemini-2.5-flash"
# api_key="${OPENAI_API_KEY}"
# api_url="https://open.xiaojingai.com/v1"
eval_model_name="Qwen2.5-VL-32B-Instruct"
api_key="none"
api_url="http://localhost:8000/v1"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

SAVE_DIR="/map-vepfs/haozhe/yhwu/vlmpaper/data/eval_w_tool_result"
mkdir -p $SAVE_DIR/$eval_model_name/$dataset_name/
python run_inference.py \
    --dataset_path /map-vepfs/haozhe/yhwu/vlmpaper/tmp/BudgetConstrainInference/seed_sft_qas/ \
    --dataset_name $dataset_name \
    --start_index 0 \
    --end_index -1 \
    --pixel_budget_per_image 12845056 \
    --save_path $SAVE_DIR/$eval_model_name/$dataset_name/ \
    --save_step 10 \
    --eval_model_name $eval_model_name \
    --model_name $eval_model_name \
    --temperature 0.3 \
    --sample_n  \
    --api_key $api_key \
    --api_url $api_url 2>&1 | tee $SAVE_DIR/$eval_model_name/$dataset_name/logging.log