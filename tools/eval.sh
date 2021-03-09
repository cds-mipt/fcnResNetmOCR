# Usage:
# ./eval.sh

python train.py \
    --dataset mapillary \
    --ocr \
    --arch resnet34_base_oc_layer3 \
    --width 1920 \
    --height 1080 \
    --device cuda:0 \
    --resume pretrained_models/mapillary_resnet34_base_oc_layer3_pretrained_backbone_focal_2_batch_6/model_best.pth \
    --test_only
    
