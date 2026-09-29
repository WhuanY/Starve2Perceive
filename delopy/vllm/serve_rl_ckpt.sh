



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

export CUDA_VISIBLE_DEVICES="2,3"


# 20260228025509_v2_noBC  /map-vepfs/haozhe/yhwu/vlmpaper/models/rl/qwen25vl7b/20260228025509_v2_noBC
# v2_ratioBC_constBC04_dm1_s40    /map-vepfs/haozhe/yhwu/vlmpaper/models/rl/qwen25vl7b/20260227130337_v2_ratioBC_constBC04_s40


cd $SCRIPT_DIR
source $VLLM_ENV_DIR/bin/activate
vllm serve /map-vepfs/haozhe/yhwu/vlmpaper/models/rl/qwen25vl7b/20260227130337_v2_ratioBC_constBC04_s40 \
    --served-model-name v2_ratioBC_constBC04_dm1_s40  \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt image=10 \
    --max_model_len 32768 \
    --port 9753

       