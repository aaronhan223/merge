import os
import argparse
import pandas as pd
from tqdm import tqdm

def impute_time_series(df):
    interval_length = 1
    global_means = df.mean()
    grouped = df.groupby('stay_id')

    all_stays_imputed = []

    for stay_id, group in tqdm(grouped):
        group = group.copy()
        
        group['time_interval'] = (group['icu_time_delta'] / interval_length).astype(int)

        # Use mean instead of last - this naturally skips NaN values
        curr_stay_imputed = group.groupby('time_interval').mean(numeric_only=True)
        
        # Keep the ID columns
        id_cols = ['stay_id', 'subject_id', 'hadm_id']
        for col in id_cols:
            if col in group.columns:
                curr_stay_imputed[col] = group.groupby('time_interval')[col].last()

        curr_stay_imputed = curr_stay_imputed.reindex(range(curr_stay_imputed.index.max() + 1))
        curr_stay_imputed['icu_time_delta'] = curr_stay_imputed.index * interval_length
        
        # Apply interpolation to non-id columns
        measurement_cols = [col for col in curr_stay_imputed.columns 
                          if col not in ['stay_id', 'subject_id', 'hadm_id', 'timedelta']]
        
        curr_stay_imputed[measurement_cols] = curr_stay_imputed[measurement_cols].interpolate(
            method='linear', 
            limit_direction='both'
        )
        
        # Final fallback: global means for any remaining NaN
        curr_stay_imputed.fillna(global_means, inplace=True)

        all_stays_imputed.append(curr_stay_imputed)

    imputed_df = pd.concat(all_stays_imputed, axis=0, ignore_index=True)
    return imputed_df

def main(args):
    labs_vitals_ts_df = pd.read_parquet(os.path.join(args.output_dir, "ts_labs_vitals.parquet"))
    labs_vitals_ts_df.drop(columns=['hosp_time_delta'], inplace=True)
    # labs_vitals_ts_df.rename(columns={'icu_time_delta': 'timedelta'}, inplace=True)

    print('Imputing time series...')
    labs_vitals_ts_df = impute_time_series(labs_vitals_ts_df)

    print('Saving imputed time series...')
    labs_vitals_ts_df.to_parquet(os.path.join(args.output_dir, "imputed_ts_labs_vitals.parquet"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    args = parser.parse_args()
    main(args)