# Building Massively Multimodal Foundation Models with Dependency-Aware Experts

## Requirements
You can install them using:
```
pip install -r requirements.txt
```

## Datasets
Here we use the PAMAP2 dataset as an example. You can download the dataset from [here](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring).

## Run Experiments

Please run experiments using the following commands to obtain results for PAMAP2 dataset.
```
sh run_pamap_data.sh
```

## RUS Estimation Methods

The current implementation supports two methods for estimating RUS values for multimodal inputs:

1. **Step-wise Batch Estimator (`method="batch"`)**
   - Computes RUS values independently at each time step using the existing batch estimation method from [1].
   - Simple and straightforward implementation.
   - Recommended for shorter temporal sequences with fewer time steps.

2. **Multi-scale Batch Estimator (`method="multiscale_batch"`)**
   - Simultaneously estimates RUS values across all time steps.
   - Proposed in the MERGE paper.
   - More computationally efficient for longer temporal sequences (e.g., more than 10 time steps).

### Usage

Specify the estimator method in `run_pamap_data.sh`:

## Code Structure

The implementation is divided into two parts, (1) compute temporal RUS values, and (2) leverage the obtained RUS values to guide the training of multimodal MoE:
- `pamap_rus_multimodal.py`: Computes per-subject temporal RUS values. Pass `--subject_ids` as a single id (`1`), a comma list (`1,2,3`), or a range (`1-9`); RUS values are saved independently for each subject.
- `train_pamap_multimodal.py`: Trains the multimodal MoE on a cross-subject split. Subjects are partitioned via `--train_subjects` / `--val_subjects` / `--test_subjects` (defaults `1-6` / `7` / `8-9`, matching the MERGE paper). Each subject's windows are paired with that subject's RUS values, so RUS for val/test subjects is only consumed at inference.

Computing temporal RUS values can take some time, especially for high-dimensional datasets. But for each dataset, it only needs to be computed once: the obtained RUS values are stored in the `results` folder and can be reused later. You can comment out the `python pamap_rus_multimodal.py` command in the `run_pamap_data.sh` script if the desired RUS values have already been computed.