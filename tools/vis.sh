# Usage:
# ./vis.sh

python train.py \
    --dataset mapillary \
    --save_dir experiments/mapillary_1920_1080_resnet34_layer3/val_mapillary \
    --arch resnet34_layer3 \
    --width 1920 \
    --height 1080 \
    --device cuda:0 \
    --resume experiments/mapillary_1920_1080_resnet34_layer3/weights/model_best.pth \
    --vis_only
