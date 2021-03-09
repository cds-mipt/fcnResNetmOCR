# Usage:
# ./train.sh

python train.py \
    --dataset mapillary \
    --save_dir experiments/mapillary_1920_1080_resnet34_layer3 \
    --arch resnet34_layer3 \
    --focal_loss \
    --gamma_focal_loss 2 \
    --width 1920 \
    --height 1080 \
    --device cuda:0 \
    --batch-size 2 \
    --epochs 100 \
    --lr 0.01 \
    --print-freq 100
