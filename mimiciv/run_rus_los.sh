#!/usr/bin/env bash

# Usage: ./run_rus_los.sh <gpu>
GPU=${1:-0}
DATA_DIR=./data/los
TRAIN_DATA="$DATA_DIR/train_los-cxr-notes-missingInd-standardized_stays.pkl"

python mimiciv_rus_multimodal.py \
    --train_dataset_path "$TRAIN_DATA" \
    --task los \
    --seq_len 48 \
    --num_lags 8 \
    --sequence_pooling mean \
    --gpu "$GPU"
