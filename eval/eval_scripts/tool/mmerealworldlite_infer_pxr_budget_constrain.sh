source $(dirname $0)/.env  # 全局信息

base_dir=$(dirname $0)/../../eval_w_tool
cd $base_dir
echo $DATA_DIR

# config 
dataset_name=mme_realworld_lite
eval_model_name="PixelReasoner-RL-v1"
api_key="none"
api_url="http://localhost:8000/v1"

source $VENV_DIR/bin/activate
echo $VENV_DIR/bin/activate

mkdir -p $SAVE_DIR/$eval_model_name/$dataset_name/
python run_inference.py \
    --dataset_path $DATA_DIR/mmerealworldlite/ \
    --inference_agent pixelreasoner \
    --pixel_budget_per_image 200704 \
    --dataset_name $dataset_name \
    --start_index 0 \
    --end_index -1 \
    --save_path $SAVE_DIR/$eval_model_name/$dataset_name/ \
    --eval_model_name $eval_model_name \
    --model_name $eval_model_name \
    --temperature 0 \
    --sample_n 1 \
    --api_key $api_key \
    --api_url $api_url 2>&1 | tee $SAVE_DIR/$eval_model_name/$dataset_name/logging.log

# 200704 = 28 * 16 ** 2

