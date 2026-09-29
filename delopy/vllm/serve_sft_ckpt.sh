#!/bin/bash
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo $SCRIPT_DIR

# 加载 .env 文件
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# config
ckpt_step=321
model_name="qwen25vl7b_fixretina_v2_sft${ckpt_step}"
LOG_FILE_NAME="serve_ckpt_${model_name}_$(date +'%Y%m%d_%H%M%S').log"
export CUDA_VISIBLE_DEVICES="0,1"

# exec
cd $SCRIPT_DIR
source $VLLM_ENV_DIR/bin/activate
vllm serve ${MY_MODELS_DIR}/sft/qwen25vl-7b/fixretina-v2_merged/checkpoint-${ckpt_step} \
    --served-model-name ${model_name} \
    --port 9752 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max_model_len ${MAX_MODEL_LEN} \
    --limit-mm-per-prompt image=10 \
    --disable-log-requests 2>&1 | tee -a ${LOG_FILE_NAME}



