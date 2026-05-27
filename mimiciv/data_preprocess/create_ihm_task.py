import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle


def get_stay_list(stays, irg_labs_vitals_df, imputed_labs_vitals_df, notes_df, cxr_df, admissions_df, include_notes=False, include_cxr=False, use_raw_data=False):
    stays_list = []

    for curr_stay in tqdm(stays, desc="Processing stays"):
        curr_stay_irg = irg_labs_vitals_df[irg_labs_vitals_df['stay_id'] == curr_stay].copy()
        curr_stay_imputed = imputed_labs_vitals_df[imputed_labs_vitals_df['stay_id'] == curr_stay].copy()

        if len(curr_stay_irg) == 0:
            continue

        curr_stay_dict = {}
        curr_stay_dict['name'] = curr_stay_irg['subject_id'].iloc[0]
        curr_stay_dict['hadm_id'] = curr_stay_irg['hadm_id'].iloc[0]
        curr_stay_dict['stay_id'] = curr_stay
        curr_stay_dict['ts_tt'] = curr_stay_irg['icu_time_delta'].values

        curr_stay_irg.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'icu_time_delta', 'hosp_time_delta'], inplace=True)
        curr_stay_imputed.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'icu_time_delta'], inplace=True)
        
        # Verify that feature names are the same for both irregular and imputed data
        irg_feature_names = curr_stay_irg.columns.tolist()
        imputed_feature_names = curr_stay_imputed.columns.tolist()
        
        if irg_feature_names != imputed_feature_names:
            raise ValueError(f"Feature names mismatch between irregular and imputed data!\n"
                           f"Irregular: {irg_feature_names}\n"
                           f"Imputed: {imputed_feature_names}")
        
        # Store feature names (verified to be the same for both irregular and imputed data)
        curr_stay_dict['feature_names'] = irg_feature_names
        
        irg_ts_mask = curr_stay_irg.notnull()
        curr_stay_irg.fillna(0, inplace=True)
        curr_stay_dict['irg_ts'] = curr_stay_irg.values
        curr_stay_dict['irg_ts_mask'] = irg_ts_mask.values.astype(int)

        curr_stay_dict['reg_ts'] = curr_stay_imputed.values

        if include_notes:
            if notes_df is None:
                curr_stay_dict['text_data'] = []
                curr_stay_dict['text_time'] = []
                curr_stay_dict['text_embeddings'] = []
                curr_stay_dict['text_missing'] = 1
            else:
                curr_stay_notes = notes_df[notes_df['stay_id'] == curr_stay].copy()

                if len(curr_stay_notes) == 0:
                    curr_stay_dict['text_data'] = []
                    curr_stay_dict['text_time'] = []
                    curr_stay_dict['text_embeddings'] = []
                    curr_stay_dict['text_missing'] = 1
                else:
                    curr_stay_dict['text_data'] = curr_stay_notes['text'].tolist()
                    curr_stay_dict['text_time'] = curr_stay_notes['icu_time_delta'].values
                    if not use_raw_data:
                        curr_stay_dict['text_embeddings'] = [np.mean(chunk_embs, axis=0) for chunk_embs in curr_stay_notes['biobert_embeddings']] # average over chunk embeddings
                    else:
                        curr_stay_dict['text_embeddings'] = []
                    curr_stay_dict['text_missing'] = 0

        if include_cxr:
            if cxr_df is None:
                curr_stay_dict['cxr_feats'] = []
                curr_stay_dict['cxr_time'] = []
                curr_stay_dict['cxr_missing'] = 1
            else:
                curr_stay_cxr = cxr_df[cxr_df['stay_id'] == curr_stay].copy()
                
                if len(curr_stay_cxr) == 0:
                    curr_stay_dict['cxr_feats'] = []
                    curr_stay_dict['cxr_time'] = []
                    curr_stay_dict['cxr_missing'] = 1
                else:
                    curr_stay_dict['cxr_subject_ids'] = curr_stay_cxr['subject_id'].values
                    curr_stay_dict['cxr_study_ids'] = curr_stay_cxr['study_id'].values
                    curr_stay_dict['cxr_dicom_ids'] = curr_stay_cxr['dicom_id'].values
                    if not use_raw_data:
                        curr_stay_dict['cxr_feats'] = curr_stay_cxr['densefeatures'].tolist()
                    else:
                        curr_stay_dict['cxr_feats'] = []
                    curr_stay_dict['cxr_time'] = curr_stay_cxr['icu_time_delta'].values
                    curr_stay_dict['cxr_missing'] = 0

        curr_stay_dict['label'] = admissions_df[admissions_df['hadm_id'] == curr_stay_dict['hadm_id']]['died'].iloc[0]

        stays_list.append(curr_stay_dict)

    return stays_list


def main(args):
    print("Starting IHM task creation...")
    print(f"Configuration:")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Restrict hours: {args.restrict_hours}")
    print(f"  - Include notes: {args.include_notes}")
    print(f"  - Include CXR: {args.include_cxr}")
    print(f"  - Include missing modalities: {args.include_missing}")
    print(f"  - Standardize features: {args.standardize_features}")
    print(f"  - Use raw data: {args.use_raw_data}")
    print(f"  - Random seed: {args.seed}")
    print()
    
    print("Loading lab and vital sign data...")
    irg_labs_vitals_df = pd.read_parquet(os.path.join(args.output_dir, "ts_labs_vitals.parquet"))
    imputed_labs_vitals_df = pd.read_parquet(os.path.join(args.output_dir, "imputed_ts_labs_vitals.parquet"))
    print(f"  - Loaded {len(irg_labs_vitals_df)} irregular time series records")
    print(f"  - Loaded {len(imputed_labs_vitals_df)} imputed time series records")

    print("Filtering data to ICU admission time and onwards...")
    irg_labs_vitals_df = irg_labs_vitals_df[irg_labs_vitals_df['icu_time_delta'] >= 0]
    imputed_labs_vitals_df = imputed_labs_vitals_df[imputed_labs_vitals_df['icu_time_delta'] >= 0]
    print(f"  - After filtering: {len(irg_labs_vitals_df)} irregular records, {len(imputed_labs_vitals_df)} imputed records")

    if args.restrict_hours is not None:
        print(f"Restricting data to first {args.restrict_hours} hours of ICU stay...")
        irg_labs_vitals_df = irg_labs_vitals_df[irg_labs_vitals_df['icu_time_delta'] <= args.restrict_hours]
        imputed_labs_vitals_df = imputed_labs_vitals_df[imputed_labs_vitals_df['icu_time_delta'] <= args.restrict_hours]
        print(f"  - After {args.restrict_hours}h restriction: {len(irg_labs_vitals_df)} irregular records, {len(imputed_labs_vitals_df)} imputed records")

    # Initialize notes_df and cxr_df
    notes_df = None
    cxr_df = None

    if args.include_notes:
        print("Loading radiology notes and text embeddings...")
        notes_df = pd.read_parquet(os.path.join(args.output_dir, "rad_notes_text_embeddings.parquet"))
        notes_df = notes_df[notes_df['stay_id'].notnull()]
        print(f"  - Loaded {len(notes_df)} note records")

        notes_df = notes_df[notes_df['icu_time_delta'] >= 0]
        if args.restrict_hours is not None:
            notes_df = notes_df[notes_df['icu_time_delta'] <= args.restrict_hours]
            print(f"  - After filtering: {len(notes_df)} note records")

    if args.include_cxr:
        print("Loading chest X-ray embeddings...")
        cxr_df = pd.read_parquet(os.path.join(args.output_dir, "mimic_cxr_embeddings.parquet"))
        cxr_df = cxr_df[cxr_df['icu_time_delta'] >= 0]
        print(f"  - Loaded {len(cxr_df)} CXR records")
        if args.restrict_hours is not None:
            cxr_df = cxr_df[cxr_df['icu_time_delta'] <= args.restrict_hours]
            print(f"  - After filtering: {len(cxr_df)} CXR records")

    print("Loading ICU stays data...")
    icustays_df = pd.read_csv(os.path.join(args.mimic_iv_dir, "icu", "icustays.csv.gz"))
    icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
    icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'])
    print(f"  - Loaded {len(icustays_df)} ICU stays")

    if args.restrict_hours is not None:
        min_los_days = args.restrict_hours / 24.0
        print(f"Filtering ICU stays to those with LOS >= {min_los_days} days...")
        icustays_df = icustays_df[icustays_df['los'] >= min_los_days]
        print(f"  - After LOS filtering: {len(icustays_df)} ICU stays")

    valid_stay_ids = icustays_df['stay_id'].unique()
    print(f"Valid stay IDs: {len(valid_stay_ids)}")

    print("Filtering all datasets to valid stay IDs...")
    irg_labs_vitals_df = irg_labs_vitals_df[irg_labs_vitals_df['stay_id'].isin(valid_stay_ids)]
    imputed_labs_vitals_df = imputed_labs_vitals_df[imputed_labs_vitals_df['stay_id'].isin(valid_stay_ids)]

    if args.include_notes:
        if notes_df is not None:
            notes_df = notes_df[notes_df['stay_id'].isin(valid_stay_ids)]
            if args.min_cxr_note_observation is not None:
                # Count observations per stay_id
                stay_counts = notes_df.groupby('stay_id').size()
                # Get stay_ids that meet the minimum observation requirement
                valid_stays_for_notes = stay_counts[stay_counts >= args.min_cxr_note_observation].index
                # Filter notes_df to only include valid stay_ids
                notes_df = notes_df[notes_df['stay_id'].isin(valid_stays_for_notes)]

    if args.include_cxr:
        if cxr_df is not None:
            cxr_df = cxr_df[cxr_df['stay_id'].isin(valid_stay_ids)]
            if args.min_cxr_note_observation is not None:
                # Count observations per stay_id
                stay_counts = cxr_df.groupby('stay_id').size()
                # Get stay_ids that meet the minimum observation requirement
                valid_stays_for_cxr = stay_counts[stay_counts >= args.min_cxr_note_observation].index
                # Filter cxr_df to only include valid stay_ids
                cxr_df = cxr_df[cxr_df['stay_id'].isin(valid_stays_for_cxr)]

    print("Loading admissions data for mortality labels...")
    admissions_df = pd.read_csv(os.path.join(args.mimic_iv_dir, "hosp", "admissions.csv.gz"))
    admissions_df = admissions_df.rename(columns={"hospital_expire_flag": "died"})
    admissions_df = admissions_df[["subject_id", "hadm_id", "died"]]
    print(f"  - Loaded {len(admissions_df)} admission records")

    print("Determining final set of stays based on modality requirements...")
    if not args.include_missing:
        unique_stays = irg_labs_vitals_df['stay_id'].unique()
        if args.include_notes and notes_df is not None:
            unique_stays = np.intersect1d(unique_stays, notes_df['stay_id'].unique())
        if args.include_cxr and cxr_df is not None:
            unique_stays = np.intersect1d(unique_stays, cxr_df['stay_id'].unique())

        print(f"Number of stays with all required modalities: {len(unique_stays)}")
    else:
        unique_stays = irg_labs_vitals_df['stay_id'].unique()
        if args.include_notes and notes_df is not None:
            unique_stays = np.union1d(unique_stays, notes_df['stay_id'].unique())
        if args.include_cxr and cxr_df is not None:
            unique_stays = np.union1d(unique_stays, cxr_df['stay_id'].unique())

        print(f"Number of stays with any available modality: {len(unique_stays)}")

    print("Creating train/validation/test splits...")
    np.random.seed(args.seed)
    np.random.shuffle(unique_stays)
    train_num = int(len(unique_stays) * 0.7)
    val_num = int(len(unique_stays) * 0.15)

    train_stays = unique_stays[:train_num]
    val_stays = unique_stays[train_num:train_num+val_num]
    test_stays = unique_stays[train_num+val_num:]
    print(f"  - Train: {len(train_stays)} stays (70%)")
    print(f"  - Validation: {len(val_stays)} stays (15%)")
    print(f"  - Test: {len(test_stays)} stays (15%)")

    train_irg_labs_vitals_df = irg_labs_vitals_df[irg_labs_vitals_df['stay_id'].isin(train_stays)]
    train_imputed_labs_vitals_df = imputed_labs_vitals_df[imputed_labs_vitals_df['stay_id'].isin(train_stays)]

    cols = train_irg_labs_vitals_df.columns.tolist()
    numeric_cols = [col for col in cols if col not in ['stay_id', 'subject_id', 'hadm_id', 'icu_time_delta', 'hosp_time_delta']]

    # Apply standardization if requested
    irg_scaler = None
    imputed_scaler = None
    
    if args.standardize_features:
        print("Fitting and applying feature standardization...")
        irg_scaler = StandardScaler()
        irg_scaler.fit(train_irg_labs_vitals_df[numeric_cols])
        
        imputed_scaler = StandardScaler()
        imputed_scaler.fit(train_imputed_labs_vitals_df[numeric_cols])
        
        # Apply scaling to all splits using training scaler
        print("  - Standardizing irregular time series features...")
        irg_labs_vitals_df[numeric_cols] = irg_scaler.transform(irg_labs_vitals_df[numeric_cols])
        print("  - Standardizing imputed time series features...")
        imputed_labs_vitals_df[numeric_cols] = imputed_scaler.transform(imputed_labs_vitals_df[numeric_cols])
        print("  - Feature standardization complete")

    # Create stay lists for all splits
    print("Processing stays data for each split...")
    print("  - Processing training stays...")
    train_stays_list = get_stay_list(train_stays, irg_labs_vitals_df, imputed_labs_vitals_df, notes_df, cxr_df, admissions_df, args.include_notes, args.include_cxr, args.use_raw_data)
    print("  - Processing validation stays...")
    val_stays_list = get_stay_list(val_stays, irg_labs_vitals_df, imputed_labs_vitals_df, notes_df, cxr_df, admissions_df, args.include_notes, args.include_cxr, args.use_raw_data)
    print("  - Processing test stays...")
    test_stays_list = get_stay_list(test_stays, irg_labs_vitals_df, imputed_labs_vitals_df, notes_df, cxr_df, admissions_df, args.include_notes, args.include_cxr, args.use_raw_data)

    # Save the data
    print("Generating output filename...")
    base_name = "ihm"
    if args.restrict_hours is not None:
        base_name += f"-{args.restrict_hours}"
    else:
        base_name += "-all"

    if args.include_cxr:
        if args.include_notes:
            base_name += "-cxr-notes"
        else:
            base_name += "-cxr"
    elif args.include_notes:
        base_name += "-notes"

    if args.include_missing:
        base_name += "-missingInd"

    if args.min_cxr_note_observation is not None:
        base_name += f"-min{args.min_cxr_note_observation}cxrnotes"

    if args.standardize_features:
        base_name += "-standardized"

    if args.use_raw_data:
        base_name += "-raw"

    print(f"Output filename base: {base_name}")

    print("Saving processed data...")
    task_dir = os.path.join(args.output_dir, "ihm")
    os.makedirs(task_dir, exist_ok=True)
    # Save train data
    f_path = os.path.join(task_dir, f"train_{base_name}_stays.pkl")
    with open(f_path, 'wb') as f:
        print(f"Saving train stays to {f_path}")
        pickle.dump(train_stays_list, f)

    # Save validation data
    f_path = os.path.join(task_dir, f"val_{base_name}_stays.pkl")
    with open(f_path, 'wb') as f:
        print(f"Saving val stays to {f_path}")
        pickle.dump(val_stays_list, f)

    # Save test data
    f_path = os.path.join(task_dir, f"test_{base_name}_stays.pkl")
    with open(f_path, 'wb') as f:
        print(f"Saving test stays to {f_path}")
        pickle.dump(test_stays_list, f)

    # Save scalers if standardization was applied
    if args.standardize_features:
        print("Saving feature standardization scalers...")
        scaler_path = os.path.join(task_dir, f"{base_name}_irg_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            print(f"Saving irregular data scaler to {scaler_path}")
            pickle.dump(irg_scaler, f)
            
        scaler_path = os.path.join(task_dir, f"{base_name}_imputed_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            print(f"Saving imputed data scaler to {scaler_path}")
            pickle.dump(imputed_scaler, f)

    print()
    print("=" * 50)
    print("IHM TASK CREATION COMPLETE!")
    print("=" * 50)
    print(f"Data saved successfully!")
    print(f"Train: {len(train_stays_list)} stays")
    print(f"Val: {len(val_stays_list)} stays") 
    print(f"Test: {len(test_stays_list)} stays")
    print(f"Output directory: {task_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_iv_dir", type=str, required=True, help='Path to mimic-iv data directory (e.g. mimiciv/3.1/)')
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    parser.add_argument("--restrict_hours", type=int, choices=[24, 48], default=None, help='Restrict to specified hours of data (24 or 48). If not specified, use all available data.')
    parser.add_argument("--include_notes", action='store_true', help='Include notes in the task')
    parser.add_argument("--include_cxr", action='store_true', help='Include chest X-rays in the task')
    parser.add_argument("--include_missing", action='store_true', help='Include stays with missing modalities')
    parser.add_argument("--min_cxr_note_observation", type=int, default=None, help='Minimum number of observations of chest x-rays and clinical notes to include in the task')
    parser.add_argument("--standardize_features", action='store_true', help='Standardize features')
    parser.add_argument("--use_raw_data", action='store_true', help='If true, not include embeddings for notes and CXR. For notes, use raw text. For CXR, use subject_id, study_id, and dicom_id to specify the image. If false, also include notes and CXR embeddings.')
    parser.add_argument("--seed", type=int, default=42, help='Random seed')
    args = parser.parse_args()
    if args.min_cxr_note_observation is not None:
        assert args.min_cxr_note_observation > 0, "Minimum number of observations per modality must be greater than 0"
        assert not args.include_missing, "Cannot include missing modalities if minimum number of observations per modality is specified"
    
    main(args)
