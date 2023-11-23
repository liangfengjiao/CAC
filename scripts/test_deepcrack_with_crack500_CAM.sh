GPU_IDS=$1

DATAROOT=./datasets/AEL
NAME=crack500_CAM_NoMeta
MODEL=deepcrack
DATASET_MODE=crack500_CAM

BATCH_SIZE=1
NORM=batch

NUM_CLASSES=1
NUM_TEST=1124

python3 test.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --norm ${NORM} \
  --num_test ${NUM_TEST}\
  --display_sides 1