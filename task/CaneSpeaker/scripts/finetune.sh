#!/bin/bash
flag="--dataset R2R RxR REVERIEobj SOON Caption Rx2R

        --batch_size 4
        --accumulate_size 4
        --epoch 2
        --lr 8e-5 4e-5
        --decay 0.99

        --dropout 0.3

        --arch linear
        
        --val_dataset R2R 
        --val_split val_seen val_unseen
        --val_dataset_sample 1

        --save 2000
        --load_opt 1
        "

# train
TRANSFORMERS_OFFLINE=1 python ./tasks/CaneSpeaker/train.py $flag \
--resume # replace with the checkpoint from warmup.sh

