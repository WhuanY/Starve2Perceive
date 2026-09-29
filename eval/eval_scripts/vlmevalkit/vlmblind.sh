# 文件位置/../../eval/VLMEvalKit/
VLMEVALKIT_ROOT_DIR=$(dirname $0)/../../../eval/VLMEvalKit
if [ ! -d "$VLMEVALKIT_ROOT_DIR" ]; then
    echo "VLMEVALKIT_ROOT_DIR does not exist"
    exit 1
fi

# config 
register_modelname="my_vllm_qwen"
data_name="VLMBlind"
concurrent_num=50

# running script
cd $VLMEVALKIT_ROOT_DIR
TIMESTAMP=$(date +%Y%m%d)
RANDOM_NUM=$(openssl rand -hex 4)
LOG_FILE_NAME="$T{TIMESTAMP}_${RANDOM_NUM}.log"
LOG_FILE_PATH="logs/${register_modelname}/${data_name}/${LOG_FILE_NAME}" 

# torchrun --nproc-per-node=1 run.py --data VLMBlind --model /home/ywuit/vlmpaper/models/Qwen2.5-VL-3B-Instruct
python run.py --data $data_name \
    --model $register_modelname \
    --api-nproc $concurrent_num \
    --reuse \
    --verbose 2>&1 | tee $LOG_FILE_PATH

