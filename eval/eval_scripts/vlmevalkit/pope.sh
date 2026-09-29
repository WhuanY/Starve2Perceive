# 文件位置/../../eval/VLMEvalKit/
VLMEVALKIT_ROOT_DIR=$(dirname $0)/../../../eval/VLMEvalKit
if [ ! -d "$VLMEVALKIT_ROOT_DIR" ]; then
    echo "VLMEVALKIT_ROOT_DIR does not exist"
    exit 1
fi

# config 
register_modelname="my_vllm_qwen25vl3b"
data_name="POPE"
concurrent_num=20 
mode="all" # "infer" or "eval" or "all"

# running script

cd $VLMEVALKIT_ROOT_DIR # cd to the vlmevalkit root directory
source $VLMEVALKIT_ROOT_DIR/.venv/bin/activate # activate the venv

TIMESTAMP=$(date +%Y%m%d)
RANDOM_NUM=$(openssl rand -hex 4)
LOG_FILE_NAME="${TIMESTAMP}_${RANDOM_NUM}.log"
mkdir -p logs/${register_modelname}/${data_name}
LOG_FILE_PATH="logs/${register_modelname}/${data_name}/${LOG_FILE_NAME}" 

# torchrun --nproc-per-node=1 run.py --data VLMBlind --model /home/ywuit/vlmpaper/models/Qwen2.5-VL-3B-Instruct
python run.py --data $data_name \
    --model $register_modelname \
    --api-nproc $concurrent_num \
    --mode $mode \
    --reuse \
    --verbose 2>&1 | tee $LOG_FILE_PATH

