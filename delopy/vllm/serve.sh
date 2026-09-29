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
model_name="Qwen2.5-VL-32B-Instruct"

# exec
cd $SCRIPT_DIR
source $VLLM_ENV_DIR/bin/activate
vllm serve /map-vepfs/models/Qwen/Qwen2.5-VL-32B-Instruct \
    --served-model-name ${model_name} \
    --port ${PORT} \
    --max_model_len 40000 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} 

