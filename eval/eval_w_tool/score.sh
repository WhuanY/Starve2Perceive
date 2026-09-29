# python /mnt/bn/luoruipu-disk-2/minyingqian/DeepEyes/eval/judge_result.py \
#     --model_name qwen_base-vstar_trm-260  \
#     --api_url "https://aigc.x-see.cn/v1" \
#     --api_key "${OPENAI_API_KEY}" \
#     --vstar_bench_path /mnt/bn/luoruipu-disk-2/miqnyingqian/data/vstar_bench \
#     --save_path /mnt/bn/luoruipu-disk-2/minyingqian/DeepEyes/eval \
#     --eval_model_name gpt-4o-mini \
#     --num_workers 32

#     # --eval_model_name qwen \
#     # --api_url "https://aigc.x-see.cn/v1" \
#     # --api_key "${OPENAI_API_KEY}" \
#     # --api_key myq \
#     # --api_url http://0.0.0.0:1234/v1 \
# # unset no_proxy
# # python /mnt/bn/luoruipu-disk-2/minyingqian/DeepEyes/eval/judge_result_deepeyes_train_data.py \
# #     --model_name qwen7b-0_expanded \
# #     --api_key myq \
# #     --api_url http://0.0.0.0:8000/v1 \
# #     --vstar_bench_path /mnt/bn/luoruipu-disk-2/minyingqian/EasyR1-main/infer_results/deepeyes/qwen7bvl-0_expanded.jsonl \
# #     --save_path /mnt/bn/luoruipu-disk-2/minyingqian/DeepEyes/eval \
# #     --eval_model_name qwen \
# #     --num_workers 16

#!/bin/bash
# Usage: ./score.sh <MODEL_NAME>

if [ -z "$1" ]; then
    echo "错误: 缺少参数。"
    echo "用法: $0 <MODEL_NAME>"
    exit 1
fi

MODEL_NAME=$1

echo "Running scoring for $MODEL_NAME..."

python /mnt/bn/luoruipu-disk-2/minyingqian/DeepEyes/eval/judge_result.py \
    --model_name $MODEL_NAME  \
    --api_url "https://aigc.x-see.cn/v1" \
    --api_key "${OPENAI_API_KEY}" \
    --vstar_bench_path /mnt/bn/luoruipu-disk-2/miqnyingqian/data/vstar_bench \
    --save_path /mnt/bn/luoruipu-disk-2/minyingqian/DeepEyes/eval/eval_results \
    --eval_model_name gpt-4o-mini \
    --num_workers 32