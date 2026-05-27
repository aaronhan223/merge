#!/usr/bin/env bash

# Usage: ./run_rus_ihm.sh <gpu>
GPU=${1:-0}
DATA_DIR=./data/ihm
TRAIN_DATA="$DATA_DIR/train_ihm-48-cxr-notes-missingInd-standardized_stays.pkl"

python mimiciv_rus_multimodal.py \
    --train_dataset_path "$TRAIN_DATA" \
    --task ihm \
    --seq_len 48 \
    --num_lags 8 \
    --sequence_pooling mean \
    --gpu "$GPU"
