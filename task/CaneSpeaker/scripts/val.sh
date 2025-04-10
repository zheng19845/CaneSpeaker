#!/bin/bash
flag="  --arch linear
        
        --val_dataset R2R
        --val_split val_seen val_unseen

        "

# generate for val splits
TRANSFORMERS_OFFLINE=1 python ./tasks/CaneSpeaker/generate_set.py $flag \
--resume # replace with the checkpoint from finetune.sh

# calculate metrics
TRANSFORMERS_OFFLINE=1 python ./tasks/CaneSpeaker/metrics/metrics.py $flag
