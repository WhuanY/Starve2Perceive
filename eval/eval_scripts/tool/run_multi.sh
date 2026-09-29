#!/bin/bash
set -x

# 加载环境变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Checkpoint列表
CKPT_STEPS=(500 550 600 650 700)

# 健康检查配置
HEALTH_CHECK_URL="http://localhost:9753/health"
TIMEOUT_SECONDS=1200
CHECK_INTERVAL=30
MAX_ATTEMPTS=$((TIMEOUT_SECONDS / CHECK_INTERVAL))

# Serve脚本路径
SERVE_SCRIPT="/home/ywuit/vlmpaper/deploy/vllm/serve_ckpt.sh"
INFER_SCRIPT_DIR="$(dirname $0)"

# 错误处理
trap 'echo "Error occurred in checkpoint processing"; kill $(jobs -p) 2>/dev/null; exit 1' ERR

echo "=== Starting Multi-Checkpoint Inference ==="
echo "Checkpoints to process: ${CKPT_STEPS[@]}"

# 遍历每个checkpoint
for ckpt_step in "${CKPT_STEPS[@]}"; do
    echo ""
    echo "=========================================="
    echo "=== Processing Checkpoint: $ckpt_step ==="
    echo "=========================================="
    
    # ============================================================================
    # Step 1: Start vLLM server in background
    # ============================================================================
    echo "=== Step 1: Starting vLLM server for checkpoint $ckpt_step ==="
    
    # 修改serve_ckpt.sh中的ckpt_step，或者直接调用vllm serve
    # 这里我们直接调用vllm serve命令，参考serve_ckpt.sh的结构
    if [ -f "$SERVE_SCRIPT" ]; then
        # 如果serve_ckpt.sh存在，我们需要修改它或者直接调用vllm
        # 为了灵活性，我们直接调用vllm serve命令
        cd "$(dirname $SERVE_SCRIPT)"
        
        # 加载serve脚本的环境变量
        if [ -f "$(dirname $SERVE_SCRIPT)/.env" ]; then
            set -a
            source "$(dirname $SERVE_SCRIPT)/.env"
            set +a
        fi
        
        model_name="qwen25vl3b_sft${ckpt_step}"
        source $VLLM_ENV_DIR/bin/activate
        
        # 启动vllm serve在后台
        vllm serve /home/ywuit/vlmpaper/models/sft/qwen25vl-3b/fixretina_merged/checkpoint-${ckpt_step} \
            --served-model-name ${model_name} \
            --port ${PORT} \
            --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
            --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
            --max_model_len ${MAX_MODEL_LEN} \
            --limit-mm-per-prompt image=10 &
        VLLM_PID=$!
    else
        echo "ERROR: Serve script not found at $SERVE_SCRIPT"
        exit 1
    fi
    
    echo "vLLM server started with PID: $VLLM_PID for checkpoint $ckpt_step"
    sleep 5  # Give the server a moment to initialize
    
    # ============================================================================
    # Step 2: Health check
    # ============================================================================
    echo "=== Step 2: Performing health checks for checkpoint $ckpt_step ==="
    ATTEMPTS=0
    HEALTH_PASSED=false
    
    while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
        ATTEMPTS=$((ATTEMPTS + 1))
        echo "Health check attempt $ATTEMPTS/$MAX_ATTEMPTS ($(($ATTEMPTS * $CHECK_INTERVAL))s elapsed)..."
        
        # Try to reach health endpoint
        if curl -s -f "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
            echo "✓ Health check PASSED for checkpoint $ckpt_step!"
            HEALTH_PASSED=true
            break
        else
            echo "✗ Health check failed. Waiting ${CHECK_INTERVAL}s before retry..."
            sleep $CHECK_INTERVAL
        fi
    done
    
    # 检查健康检查是否通过
    if [ "$HEALTH_PASSED" = false ]; then
        echo "CRITICAL: Health check failed after ${TIMEOUT_SECONDS}s for checkpoint $ckpt_step."
        echo "Killing vLLM server and skipping to next checkpoint."
        kill $VLLM_PID 2>/dev/null || true
        continue  # 跳过这个checkpoint，继续下一个
    fi
    
    # ============================================================================
    # Step 3: Run inference
    # ============================================================================
    echo "=== Step 3: Running inference for checkpoint $ckpt_step ==="
    
    # 创建临时推理脚本，设置正确的ckpt_step
    TEMP_INFER_SCRIPT=$(mktemp)
    cat > "$TEMP_INFER_SCRIPT" << EOF
source $INFER_SCRIPT_DIR/.env  # 全局信息

base_dir=\$(dirname \$0)/../../eval_w_tool
cd \$base_dir
echo \$DATA_DIR

# config 
dataset_name=generalvqa
ckpt_step=${ckpt_step}
eval_model_name="qwen25vl3b_sft\${ckpt_step}"
api_key="none"
api_url="http://localhost:9753/v1"

source \$VENV_DIR/bin/activate
echo \$VENV_DIR/bin/activate

mkdir -p \$SAVE_DIR/\$eval_model_name/\$dataset_name/
python run_inference.py \\
    --dataset_path \$DATA_DIR/FixRetina_RL \\
    --dataset_name \$dataset_name \\
    --start_index 0 \\
    --end_index -1 \\
    --save_path \$SAVE_DIR/\$eval_model_name/\$dataset_name/ \\
    --eval_model_name \$eval_model_name \\
    --model_name \$eval_model_name \\
    --temperature 0 \\
    --sample_n 1 \\
    --api_key \$api_key \\
    --api_url \$api_url 2>&1 | tee \$SAVE_DIR/\$eval_model_name/\$dataset_name/logging.log
EOF
    
    bash "$TEMP_INFER_SCRIPT"
    INFERENCE_EXIT_CODE=$?
    rm -f "$TEMP_INFER_SCRIPT"
    
    echo "=== Inference completed for checkpoint $ckpt_step with exit code: $INFERENCE_EXIT_CODE ==="
    
    # ============================================================================
    # Step 4: Cleanup - Gracefully shutdown vLLM server
    # ============================================================================
    echo "=== Step 4: Gracefully shutting down vLLM server for checkpoint $ckpt_step ==="
    if kill -0 $VLLM_PID 2>/dev/null; then
        echo "Sending SIGTERM to vLLM server (PID: $VLLM_PID)..."
        kill -TERM $VLLM_PID
        
        # Wait up to 30 seconds for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 $VLLM_PID 2>/dev/null; then
                echo "vLLM server shut down gracefully for checkpoint $ckpt_step"
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 $VLLM_PID 2>/dev/null; then
            echo "Force killing vLLM server for checkpoint $ckpt_step..."
            kill -9 $VLLM_PID 2>/dev/null || true
        fi
    fi
    
    # 等待一段时间确保端口释放
    sleep 5
    
    echo "=== Checkpoint $ckpt_step processing complete ==="
done

echo ""
echo "=== All Checkpoints Processing Complete ==="