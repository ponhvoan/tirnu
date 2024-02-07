#! /usr/bin/env bash

DATADIR=data/cifar
DATASET='cifar10 cifar100'
CORRUPTION='snow gaussian_noise motion_blur elastic_transform brightness pixelate defocus_blur shot_noise impulse_noise glass_blur zoom_blur frost fog contrast jpeg_compression cifar_mix5 cifar_mix10'

########## Parameters ##########
LEVEL=5
METHOD=tirnu
NSAMPLE=100000
LR=0.001
BS_SHOT=128
TRANSFORM=T1
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
            --queue_size 0 \
            --transform T1 \
            --ent_par 1.0 \
            --ent2_par 0.2 \
            --uncert_par 0.1 \
            --Izn_par 0.1 \
            --alpha 1.01 \
            --nepoch 10 \
            --transform T1
            # --tsne
    done
done