#!/bin/bash
flag="--dataset Caption

        --batch_size 8
        --accumulate_size 4
        --epoch 2
        --lr 1e-4 5e-5
        --decay 0.99

        --dropout 0.3

        --arch linear
        
        --val_dataset R2R
        --val_split val_seen val_unseen
        --val_dataset_sample 0.01

        --save 100000
        --load_opt 1
        "

# train
TRANSFORMERS_OFFLINE=1 python ./tasks/CaneSpeaker/train.py $flag \
--resume None