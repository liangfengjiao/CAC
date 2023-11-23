GPU_IDS=$1

DATAROOT=./datasets/crack500
NAME=crack500_CAM_proportion_c=0.5
MODEL=deepcrack
DATASET_MODE=crack500_CAM_meta

BATCH_SIZE=4
NUM_CLASSES=1
LOAD_WIDTH=256
LOAD_HEIGHT=256
LOSS_MODE=focal

NORM=batch
NITER=30
NITER_DECAY=10

python3 meta_train_with_crack500.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --norm ${NORM} \
  --lr_decay_iters 10 \
  --lr_policy step \
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} \
  --no_flip 0 \
  --display_id 0 \
  --loss_mode ${LOSS_MODE}
