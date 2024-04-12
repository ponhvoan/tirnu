#! /usr/bin/env bash

DATADIR=data/cifar
DATASET='cifar10'
CORRUPTION='snow gaussian_noise motion_blur elastic_transform brightness pixelate defocus_blur shot_noise impulse_noise glass_blur zoom_blur frost fog contrast jpeg_compression'

########## Parameters ##########
LEVEL=5
METHOD=tirnu
NSAMPLE=100000
LR=0.001
BS_SHOT=128
TRANSFORM=simclr
######################################
for DATA in $DATASET
do
    for CORRUPT in $CORRUPTION
    do
        python main.py \
            --dataset ${DATA} \
            --dataroot ${DATADIR} \
            --resume models/source_weights/${DATA}_resnet50 \
            --outf results/${DATA}_${METHOD}_resnet50 \
            --corruption ${CORRUPT} \
            --level ${LEVEL} \
            --workers 36 \
            --batch_size ${BS_SHOT} \
            --ks 10 \
            --lr ${LR} \
            --transform ${TRANSFORM} \
            --nepoch 10
            # --tsne
    done
done