import os
import argparse
import pandas as pd


def add_time_delta_cxr_vectorized(metadata_df, admissions_df, icustays_df):
    """
    Add event time with respect to hospital admission time and ICU stay time to the CXR metadata dataframe.
    Args:
        metadata_df: CXR metadata DataFrame with study_datetime
        admissions_df: admissions DataFrame with hospital admission time
        icustays_df: icustays DataFrame with ICU stay time
    Returns:
        df: DataFrame with event times with respect to hospital admission time and ICU stay time
    """
    df = metadata_df.copy()
    
    # Initialize new columns
    df['hadm_id'] = None
    df['stay_id'] = None
    
    # Step 1: Handle hospital admissions
    # Create cartesian product of CXR studies and admissions for the same subject
    df_with_idx = df.reset_index().rename(columns={'index': 'original_idx'})
    
    # Merge with admissions data
    admissions_subset = admissions_df[['subject_id', 'hadm_id', 'admittime', 'dischtime']].copy()
    merged_adm = df_with_idx.merge(admissions_subset, on='subject_id', how='left', suffixes=('_orig', '_adm'))
    
    # Filter to CXR studies that fall within admission time windows
    mask_adm = (merged_adm['study_datetime'] >= merged_adm['admittime']) & (merged_adm['study_datetime'] <= merged_adm['dischtime'])
    valid_adm_matches = merged_adm[mask_adm].copy()
    
    if not valid_adm_matches.empty:
        # Calculate hospital time delta for valid matches
        valid_adm_matches['hosp_time_delta'] = (valid_adm_matches['study_datetime'] - valid_adm_matches['admittime']).dt.total_seconds() / 3600
        
        # Handle multiple admissions for same CXR study (take the first match by admit time)
        valid_adm_matches = valid_adm_matches.sort_values(['original_idx', 'admittime'])
        valid_adm_matches = valid_adm_matches.drop_duplicates('original_idx', keep='first')
        
        # Merge back the hadm_id and hosp_time_delta
        df = df_with_idx.merge(
            valid_adm_matches[['original_idx', 'hadm_id_adm', 'hosp_time_delta']], 
            on='original_idx', 
            how='left'
        )
        df['hadm_id'] = df['hadm_id_adm']
        df = df.drop(['hadm_id_adm'], axis=1)
    else:
        df = df_with_idx
    
    # Step 2: Handle ICU stays
    # Create cartesian product of CXR studies and ICU stays for the same subject
    icu_stays_subset = icustays_df[['subject_id', 'stay_id', 'intime', 'outtime']].copy()
    merged_icu = df.merge(icu_stays_subset, on='subject_id', how='left', suffixes=('_orig', '_icu'))
    
    # Filter to CXR studies that fall within ICU stay time windows
    mask_icu = (merged_icu['study_datetime'] >= merged_icu['intime']) & (merged_icu['study_datetime'] <= merged_icu['outtime'])
    valid_icu_matches = merged_icu[mask_icu].copy()
    
    if not valid_icu_matches.empty:
        # Calculate ICU time delta for valid matches
        valid_icu_matches['icu_time_delta'] = (valid_icu_matches['study_datetime'] - valid_icu_matches['intime']).dt.total_seconds() / 3600
        
        # Handle multiple ICU stays for same CXR study (take the first match by intime)
        valid_icu_matches = valid_icu_matches.sort_values(['original_idx', 'intime'])
        valid_icu_matches = valid_icu_matches.drop_duplicates('original_idx', keep='first')
        
        # Update the stay_id and icu_time_delta columns
        df = df.merge(valid_icu_matches[['original_idx', 'stay_id_icu', 'icu_time_delta']], on='original_idx', how='left')
        df['stay_id'] = df['stay_id_icu']
        df = df.drop(['stay_id_icu'], axis=1)

    # Clean up temporary columns
    df = df.drop(['original_idx'], axis=1, errors='ignore')
    
    # Sort the DataFrame
    df = df.sort_values(by=['subject_id', 'hadm_id', 'hosp_time_delta', 'stay_id', 'icu_time_delta'])
    
    return df


def main(args):
    print('Loading MMIC CXR metadata...')
    metadata_df = pd.read_csv(os.path.join(args.mimic_cxr_jpg_dir, "mimic-cxr-2.0.0-metadata.csv.gz"))
    # normalize to consistent datetime
    metadata_df["StudyDate"] = metadata_df["StudyDate"].astype(str).str.zfill(8)                     # YYYYMMDD
    metadata_df["StudyTime"] = (metadata_df["StudyTime"].fillna("000000").astype(str)
                        .str.split(".").str[0].str.zfill(6))                          # HHMMSS
    metadata_df["study_datetime"] = pd.to_datetime(metadata_df["StudyDate"] + metadata_df["StudyTime"],
                                        format="%Y%m%d%H%M%S", errors="coerce")
    metadata_df = metadata_df.dropna(subset=["study_datetime"])

    print('Loading icustays...')
    icustays_df = pd.read_csv(os.path.join(args.mimic_iv_dir, "icu", "icustays.csv.gz"))
    icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
    icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'])

    print('Loading admissions...')
    admissions_df = pd.read_csv(os.path.join(args.mimic_iv_dir, "hosp", "admissions.csv.gz"))
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

    print('Adding time delta...')
    metadata_df = add_time_delta_cxr_vectorized(metadata_df, admissions_df, icustays_df)
    
    print('Saving data...')
    metadata_df.to_parquet(os.path.join(args.output_dir, "cxr_metadata_with_time_delta.parquet"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_cxr_jpg_dir", type=str, required=True, help='Path to mimic-cxr-jpg v2.0.0 directory (e.g. mimic-cxr-jpg/2.0.0/)')
    parser.add_argument("--mimic_iv_dir", type=str, required=True, help='Path to mimic-iv data directory (e.g. mimiciv/3.1/)')
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)