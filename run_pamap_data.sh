#!/bin/bash

# Set global variables for shared hyper-parameters
subject_ids="1-9"          # All PAMAP subjects: per-subject RUS values
train_subjects="1-6"       # Subjects used for training
val_subjects="7"           # Validation subject
test_subjects="8-9"        # Held-out test subjects (RUS for these is only used at inference)
method="batch"
max_lag=5
discrim_epochs=30
ce_epochs=20
n_batches=1
batch_size=512
seed=42
gpu=3

start_time=$(date +%s)

# Compute per-subject RUS values for every subject in $subject_ids
python pamap_rus_multimodal.py \
    --method $method \
    --subject_ids $subject_ids \
    --max_lag $max_lag \
    --dominance_threshold 0.4 \
    --dominance_percentage 0.9 \
    --gpu $gpu \
    --hidden_dim 64 \
    --layers 3 \
    --lr 0.001 \
    --discrim_epochs $discrim_epochs \
    --ce_epochs $ce_epochs \
    --activation relu \
    --embed_dim 20 \
    --batch_size $batch_size \
    --n_batches $n_batches \
    --seed $seed

# Train on $train_subjects, validate on $val_subjects, test on $test_subjects.
# Each subject is paired with its own RUS values — RUS for val/test subjects is
# only used at inference time, matching the raw-data split described in MERGE.
python train_pamap_multimodal.py \
    --train_subjects $train_subjects \
    --val_subjects $val_subjects \
    --test_subjects $test_subjects \
    --seq_len 100 \
    --window_step 50 \
    --rus_max_lag $max_lag \
    --d_model 128 \
    --nhead 4 \
    --d_ff 256 \
    --num_encoder_layers 6 \
    --num_moe_layers 3 \
    --dropout 0.1 \
    --modality_encoder_layers 2 \
    --moe_num_experts 8 \
    --moe_num_synergy_experts 2 \
    --moe_k 2 \
    --moe_expert_hidden_dim 128 \
    --moe_capacity_factor 1.25 \
    --moe_router_gru_hidden_dim 64 \
    --moe_router_token_processed_dim 64 \
    --moe_router_attn_key_dim 32 \
    --moe_router_attn_value_dim 32 \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --clip_grad_norm 1.0 \
    --use_lr_scheduler \
    --threshold_u 0.5 \
    --threshold_r 0.1 \
    --threshold_s 0.1 \
    --lambda_u 10 \
    --lambda_r 10 \
    --lambda_s 10 \
    --epsilon_loss 1e-8 \
    --cuda_device $gpu \
    --wandb_project pamap-multimodal-trus-moe \
    --rus_method $method \
    --rus_discrim_epochs $discrim_epochs \
    --rus_ce_epochs $ce_epochs \
    --rus_n_batches $n_batches \
    --rus_batch_size $batch_size \
    --rus_seed $seed

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total running time: $elapsed_time seconds"
