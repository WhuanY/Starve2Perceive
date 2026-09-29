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

export CUDA_VISIBLE_DEVICES="0,1"

cd $SCRIPT_DIR
source $VLLM_ENV_DIR/bin/activate
vllm serve /map-vepfs/haozhe/yhwu/vlmpaper/models/rl/qwen25vl7b/PixelReasoner-RL-v1/ \
    --served-model-name PixelReasoner-RL-v1 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt image=10 \
    --max_model_len 30000 \
    --port 9753
