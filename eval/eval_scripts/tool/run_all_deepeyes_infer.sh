#!/bin/bash
# Script to run all AdaptVision inference scripts sequentially and shutdown the machine

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# List of AdaptVision inference scripts to run (in order)
INFERENCE_SCRIPTS=(
    "hrbench4k_infer_deepeyes.sh"
    "hrbench8k_infer_deepeyes.sh"
    "vstar_infer_deepeyes.sh"
    "mmerealworldlite_infer_deepeyes.sh"
)

# Log file
LOG_FILE="${SCRIPT_DIR}/run_all_adaptvision_infer.log"
START_TIME_EPOCH=$(date +%s)
START_TIME=$(date +%Y-%m-%d_%H-%M-%S)

echo "=========================================="
echo "AdaptVision Inference Batch Runner"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Track failures
FAILED_SCRIPTS=()
SUCCESSFUL_SCRIPTS=()

# Run each inference script sequentially
for script in "${INFERENCE_SCRIPTS[@]}"; do
    script_path="${SCRIPT_DIR}/${script}"
    
    if [ ! -f "$script_path" ]; then
        log_message "ERROR: Script not found: $script_path"
        FAILED_SCRIPTS+=("$script (NOT FOUND)")
        continue
    fi
    
    log_message "=========================================="
    log_message "Starting: $script"
    log_message "=========================================="
    
    # Run the script and capture exit code
    script_start_time=$(date +%s)
    bash "$script_path" 2>&1 | tee -a "$LOG_FILE"
    script_exit_code=${PIPESTATUS[0]}
    script_end_time=$(date +%s)
    script_duration=$((script_end_time - script_start_time))
    
    if [ $script_exit_code -eq 0 ]; then
        log_message "✓ SUCCESS: $script completed in ${script_duration} seconds"
        SUCCESSFUL_SCRIPTS+=("$script")
    else
        log_message "✗ FAILED: $script exited with code $script_exit_code after ${script_duration} seconds"
        FAILED_SCRIPTS+=("$script (exit code: $script_exit_code)")
    fi
    
    log_message ""
done

# Summary
END_TIME=$(date +%Y-%m-%d_%H-%M-%S)
END_TIME_EPOCH=$(date +%s)
TOTAL_DURATION=$((END_TIME_EPOCH - START_TIME_EPOCH))

log_message "=========================================="
log_message "SUMMARY"
log_message "=========================================="
log_message "Start time: $START_TIME"
log_message "End time: $END_TIME"
log_message "Total duration: ${TOTAL_DURATION} seconds ($(($TOTAL_DURATION / 60)) minutes)"
log_message ""
log_message "Successful scripts (${#SUCCESSFUL_SCRIPTS[@]}):"
for script in "${SUCCESSFUL_SCRIPTS[@]}"; do
    log_message "  ✓ $script"
done
log_message ""
log_message "Failed scripts (${#FAILED_SCRIPTS[@]}):"
if [ ${#FAILED_SCRIPTS[@]} -eq 0 ]; then
    log_message "  (none)"
else
    for script in "${FAILED_SCRIPTS[@]}"; do
        log_message "  ✗ $script"
    done
fi
log_message "=========================================="
log_message ""

# Check if we should shutdown
if [ ${#FAILED_SCRIPTS[@]} -eq 0 ]; then
    log_message "All inference scripts completed successfully."
else
    log_message "Some inference scripts failed, but proceeding with shutdown."
fi

log_message "Preparing to shutdown the machine in 30 seconds..."
log_message "Press Ctrl+C within 30 seconds to cancel shutdown."
sleep 30

log_message "Shutting down the machine now..."
log_message "Final log entry at: $(date)"
echo ""

# Shutdown the machine
/usr/bin/shutdown -h now "deepeyes inference batch completed. Shutting down."

