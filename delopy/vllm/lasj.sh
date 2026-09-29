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
model_name="Qwen2.5-72B-Instruct-AWQ"
LOG_FILE_NAME="lasj_$(date +'%Y%m%d_%H%M%S').log"

# exec
cd $SCRIPT_DIR
export CUDA_VISIBLE_DEVICES="6,7"
# export CUDA_VISIBLE_DEVICES="3"
source $VLLM_ENV_DIR/bin/activate
vllm serve ${BASE_MODEL_DIR}/${model_name} \
    --served-model-name "judge" \
    --port ${PORT} \
    --max_model_len 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --disable-log-requests 2>&1 | tee -a ${LOG_FILE_NAME}

