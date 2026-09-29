source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name="treebench"
eval_model_name="qwen25vl7b_fixretina_v1_sft555"
api_key="none"
api_url="http://localhost:9752/v1"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

mkdir -p $SAVE_DIR/$eval_model_name/$dataset_name/
python run_inference.py \
    --dataset_path $DATA_DIR/TreeBench \
    --dataset_name $dataset_name \
    --start_index 0 \
    --end_index -1 \
    --pixel_budget_per_image $((16384 * 28 * 28)) \
    --save_path $SAVE_DIR/$eval_model_name/$dataset_name/ \
    --eval_model_name $eval_model_name \
    --model_name $eval_model_name \
    --temperature 0.0 \
    --sample_n 1 \
    --api_key $api_key \
    --api_url $api_url 2>&1 | tee $SAVE_DIR/$eval_model_name/$dataset_name/logging.log