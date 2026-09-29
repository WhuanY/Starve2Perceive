VISUAL_PROBE_TRAIN_BASE_DIR="/map-vepfs/haozhe/yhwu/vlmpaper/data/rl/VisualProbe/VisualProbe_train"
TRAIN_DATASET_VISUAL_PROBE=(${VISUAL_PROBE_TRAIN_BASE_DIR}/visualprobe_maxview448_highres1024_fixretina_part*.parquet)
IFS=',' TRAIN_DATASET="${TRAIN_DATASET_VISUAL_PROBE[*]}"
echo "TRAIN_DATASET=${TRAIN_DATASET}"
