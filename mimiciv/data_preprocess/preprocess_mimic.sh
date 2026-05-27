#!/usr/bin/env bash
# Usage: bash data_preprocess/preprocess_mimic.sh \
#            <mimic_iv_dir> <mimic_notes_dir> <mimic_cxr_jpg_dir> [gpu]
#
# mimic_iv_dir:      path to MIMIC-IV 3.1 root (contains hosp/ and icu/)
# mimic_notes_dir:   path to MIMIC-IV-Note 2.2 note/ directory
#                    (the directory that directly contains radiology.csv.gz)
# mimic_cxr_jpg_dir: path to MIMIC-CXR-JPG 2.0.0 root
# gpu:               GPU device ID for embedding steps (default: 0)
#
# All intermediate and final files are written to ./data/
# Run from the mimiciv/ directory.

set -e

if [[ $# -lt 3 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: bash data_preprocess/preprocess_mimic.sh <mimic_iv_dir> <mimic_notes_dir> <mimic_cxr_jpg_dir> [gpu]"
    echo ""
    echo "  mimic_iv_dir:      MIMIC-IV 3.1 root (contains hosp/ and icu/)"
    echo "  mimic_notes_dir:   MIMIC-IV-Note 2.2 note/ dir (contains radiology.csv.gz)"
    echo "  mimic_cxr_jpg_dir: MIMIC-CXR-JPG 2.0.0 root"
    echo "  gpu:               GPU device ID (default: 0)"
    exit 1
fi

MIMIC_IV_DIR=$1
MIMIC_NOTES_DIR=$2
MIMIC_CXR_JPG_DIR=$3
GPU=${4:-0}
OUTPUT_DIR=./data

echo "=== Step 1/8: Irregular time series (labs + vitals) ==="
python data_preprocess/preprocess_irg_time_series.py \
    --mimic_iv_dir "$MIMIC_IV_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "=== Step 2/8: Imputed regular time series ==="
python data_preprocess/preprocess_imputed_time_series.py \
    --output_dir "$OUTPUT_DIR"

echo "=== Step 3/8: Radiology notes text ==="
python data_preprocess/preprocess_notes.py \
    --mimic_iv_dir "$MIMIC_IV_DIR" \
    --mimic_iv_notes_dir "$MIMIC_NOTES_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "=== Step 4/8: BioBERT note embeddings (GPU-intensive) ==="
python data_preprocess/preprocess_notes_embeddings.py \
    --output_dir "$OUTPUT_DIR" \
    --device_number "$GPU"

echo "=== Step 5/8: CXR metadata ==="
python data_preprocess/preprocess_cxr.py \
    --mimic_cxr_jpg_dir "$MIMIC_CXR_JPG_DIR" \
    --mimic_iv_dir "$MIMIC_IV_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "=== Step 6/8: DenseNet121 CXR embeddings (GPU-intensive) ==="
python data_preprocess/preprocess_cxr_embeddings.py \
    --mimic_cxr_jpg_dir "$MIMIC_CXR_JPG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device_number "$GPU"

echo "=== Step 7/8: Create IHM task (train/val/test pkl files) ==="
python data_preprocess/create_ihm_task.py \
    --mimic_iv_dir "$MIMIC_IV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --restrict_hours 48 \
    --include_notes \
    --include_cxr \
    --include_missing \
    --standardize_features \
    --seed 42

echo "=== Step 8/8: Create LOS task (train/val/test pkl files) ==="
python data_preprocess/create_los_task.py \
    --mimic_iv_dir "$MIMIC_IV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --include_notes \
    --include_cxr \
    --include_missing \
    --standardize_features \
    --seed 42

echo ""
echo "Preprocessing complete. Output files:"
echo "  IHM: $OUTPUT_DIR/ihm/{train,val,test}_ihm-48-cxr-notes-missingInd-standardized_stays.pkl"
echo "  LOS: $OUTPUT_DIR/los/{train,val,test}_los-cxr-notes-missingInd-standardized_stays.pkl"
