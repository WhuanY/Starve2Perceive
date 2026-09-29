set -x
export CUDA_VISIBLE_DEVICES="0"
# repo root is five levels up from train/rl/verl_starve/examples/agent/
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}/train/rl/verl_starve"
set -a
source .env
set +a

source /opt/conda/etc/profile.d/conda.sh
conda activate "$VENV_DIR"

mkdir -p $LOG_DIR
PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="debug_for_single_node_$(date +%Y%m%d%H%M%S)"


BASEDIR="${REPO_ROOT}/data/rl/fixretina_rl" # rl training data base dir
TRAIN_DATA_DIR="${BASEDIR}/train"

# VisualProbe is sharded; glob the parts and join with commas
TRAIN_DATASET_VISUAL_PROBE_PATHS=(${TRAIN_DATA_DIR}/visualprobe_crNone_res_262144toInf_fixretina_ap_part[0-9].parquet ${TRAIN_DATA_DIR}/visualprobe_crNone_res_262144toInf_fixretina_ap_part1[0-1].parquet)
IFS=',' TRAIN_DATASET_VPROBE="${TRAIN_DATASET_VISUAL_PROBE_PATHS[*]}"
TRAIN_DATASET_TREEVGR_PATH=${TRAIN_DATA_DIR}/treevgr_cs1to3_fixrl.parquet
TRAIN_DATASET_VIRL_PATH=${TRAIN_DATA_DIR}/virl8k_crNone_res_262144toNone_fixretina_ap.parquet
TRAIN_DATASET_CHART_PATH=${TRAIN_DATA_DIR}/dpys_chart_fixretina_262144toNone.parquet

EVAL_DATASET=${BASEDIR}/eval/heldout_50/all_heldout_50.parquet
WORLD_SIZE=1

# set QWEN25VL_3B to your local Qwen2.5-VL-3B-Instruct, or leave it to pull from HF
REF_MODEL_PATH="${QWEN25VL_3B:-Qwen/Qwen2.5-VL-3B-Instruct}"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files="[${TRAIN_DATASET_VPROBE},${TRAIN_DATASET_VIRL_PATH}]" \
    data.val_files=[${EVAL_DATASET}] \
    data.train_batch_size=24 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.custom_cls.path=verl.utils.dataset.rl_dataset_fixretina \
    data.custom_cls.name=RLHFDatasetFixRetina \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=8000 \
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=32 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    reward_model.reward_manager=prime \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=50 \
    trainer.test_freq=10000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=32 2>&1 | tee ${LOG_DIR}/${EXPERIMENT_NAME}.log
