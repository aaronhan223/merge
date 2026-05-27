#! /bin/bash
# Best found: lambda_rus = 0.5, lambda_load = 0.02

# Check if correct number of arguments provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <lambda_rus> <lambda_load> <gpu> <seed>"
    echo "  lambda_rus: Value for lambda_u, lambda_r, and lambda_s (e.g., 0, 0.5, 1)"
    echo "  lambda_load: Value for lambda_load (e.g., 0.02, 0.05)"
    echo "  gpu: GPU device ID (e.g., 0, 1, 2)"
    echo "  seed: Random seed (e.g., 42, 123, 456)"
    echo ""
    echo "Example: $0 0.5 0.02 1 42"
    exit 1
fi

# Get arguments
LAMBDA_RUS=$1
LAMBDA_LOAD=$2
GPU=$3
SEED=$4

# Validate lambda_rus argument
if [[ ! "$LAMBDA_RUS" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Error: lambda_rus must be a number (got: $LAMBDA_RUS)"
    exit 1
fi

# Validate lambda_load argument
if [[ ! "$LAMBDA_LOAD" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Error: lambda_load must be a number (got: $LAMBDA_LOAD)"
    exit 1
fi

# Validate GPU argument
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "Error: gpu must be a non-negative integer (got: $GPU)"
    exit 1
fi

# Validate seed argument
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: seed must be a non-negative integer (got: $SEED)"
    exit 1
fi

# Create run name with hyperparameters
RUN_NAME="mimiciv_los_lambdarus${LAMBDA_RUS}_lambdaload${LAMBDA_LOAD}_seed${SEED}"

echo "Starting training with:"
echo "  lambda_u = lambda_r = lambda_s = $LAMBDA_RUS"
echo "  lambda_load = $LAMBDA_LOAD"
echo "  gpu = $GPU"
echo "  seed = $SEED"
echo "  wandb_run_name = $RUN_NAME"
echo ""

python train_mimiciv_multimodal.py \
    --train_data_path ./data/los/train_los-cxr-notes-missingInd-standardized_stays.pkl \
    --val_data_path ./data/los/val_los-cxr-notes-missingInd-standardized_stays.pkl \
    --rus_data_path ./results/los/rus_multimodal_all_seq48_lags8_meanpool.npy \
    --task los \
    --truncate_from_end \
    --seq_len 48 \
    --gpu $GPU \
    --use_wandb \
    --wandb_project mimiciv-multimodal-trus-moe \
    --plot_expert_activations \
    --lambda_u $LAMBDA_RUS \
    --lambda_r $LAMBDA_RUS \
    --lambda_s $LAMBDA_RUS \
    --lambda_load $LAMBDA_LOAD \
    --run_name $RUN_NAME \
    --seed $SEED \
    --lr 1e-3 \
    --epochs 20 \
    --output_dir ./results
