set -x
export CUDA_VISIBLE_DEVICES="0,1"
cd /map-vepfs/haozhe/yhwu/vlmpaper/train/rl/verl_starve
set -a 
source .env 
set +a 

source $VENV_DIR/bin/activate

mkdir -p $LOG_DIR
PROJECT_NAME="agent_vlagent_debug"
EXPERIMENT_NAME="debug_chatml_format_reward_lasj_acc_vprobeconvergetest$(date +%Y%m%d%H%M%S)"
TRAIN_DATASET_VPROBE="$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part0.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part1.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part2.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part3.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part4.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part5.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part6.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part7.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part8.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part9.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part10.parquet,$TRAIN_DATA_DIR/visualprobe_crNone_res_262144toInf_fixretina_ap_part11.parquet"
DEEPEYES_CHART_PATH="$TRAIN_DATA_DIR/dpys_chart_fixretina_262144toNone.parquet"
TREEVGR_PATH="$TRAIN_DATA_DIR/treevgr_cs1to3_fixrl.parquet"
VIRL_PATH="$TRAIN_DATA_DIR/virl8k_crNone_res_262144toNone_fixretina_ap.parquet"
PXR_PATH="$TRAIN_DATA_DIR/pxr_res1024toNone_fixretina_ap_part3.parquet"
TRAIN_FILES_LIST="[${VIRL_PATH},${TREEVGR_PATH},${DEEPEYES_CHART_PATH},${TRAIN_DATASET_VPROBE},${PXR_PATH}]"
# TRAIN_FILES_LIST="[${VIRL_PATH},${PXR_PATH}]"
# EVAL_DATASET="/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/eval/heldout_50/all_heldout_50.parquet"
EVAL_DATASET="/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/virl8k_crNone_res_262144toNone_fixretina_ap_100.parquet,/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/treevgr_cs1to3_fixrl_100.parquet,/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/dpys_chart_fixretina_262144toNone_100.parquet,/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/visualprobe_crNone_res_262144toInf_fixretina_ap_part5_100.parquet"
EVAL_FILE_LIST="[${EVAL_DATASET}]"
# SMALL_DATASET="/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/eval/heldout_50/visualprobe_hard_heldout_50.parquet"
SMALL_DATASET="/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/eval/heldout_50/vstar_heldout_50.parquet"


WORLD_SIZE=1

b_tkn_max=512
b_tkn_start=256 
b_tkn_end=8 
b_alpha=100
curriculum_controler_type="absolute"
DATASET_CLASS_PATH="verl.utils.dataset.rl_dataset_fixretina"
DATASET_CLASS_NAME="RLHFDatasetFixRetina"
# REF_MODEL_PATH="/map-vepfs/haozhe/yhwu/vlmpaper/models/sft/qwen25vl-7b/fixretina-v2_merged/checkpoint-321"
# REF_MODEL_PATH="/map-vepfs/haozhe/yhwu/vlmpaper/models/sft/qwen25vl-3b/fxiretina-v1_merged/checkpoint-555"
# REF_MODEL_PATH="/map-vepfs/models/Qwen2.5-VL-3B-Instruct"
REF_MODEL_PATH="/map-vepfs/haozhe/yhwu/vlmpaper/models/sft/qwen25vl-7b/fixretina_distillv2_sc_qw25_s900"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files=/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/treevgr_cs1to3_fixrl_100.parquet \
    data.val_files=/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/fixretina_rl/train/treevgr_cs1to3_fixrl_100.parquet \
    data.train_batch_size=10 \
    data.max_prompt_length=20000 \
    data.max_response_length=30000 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.custom_cls.path=${DATASET_CLASS_PATH} \
    data.custom_cls.name=${DATASET_CLASS_NAME} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=1024 \
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=48 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    reward_model.reward_manager=prime \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10000 \
    trainer.test_freq=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    +trainer.curriculum.b_tkn_max=${b_tkn_start} \
    +trainer.curriculum.b_tkn_min=${b_tkn_end} \
    +trainer.curriculum.b_tkn_start=${b_tkn_start} \
    +trainer.curriculum.controler_type=${curriculum_controler_type} \
    +trainer.curriculum.b_alpha=${b_alpha} \
    +trainer.curriculum.warmup_steps=100 \
    trainer.total_training_steps=1 2>&1 | tee ${LOG_DIR}/${EXPERIMENT_NAME}.log
    # trainer.total_epochs=1 2>&1 | tee ${LOG_DIR}/${EXPERIMENT_NAME}.log
