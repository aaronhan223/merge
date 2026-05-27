import os
import argparse
import pandas as pd


def get_labs_of_interest(labevents_df, d_lab_items_df):
    df = labevents_df.copy()
    event_list = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
           'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
           'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
           'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW', 
           'Platelet Count', 'Neutrophils', 'Vancomycin']
    
    event_id_df = pd.DataFrame()
    for event in event_list:
        event_item_id = d_lab_items_df[d_lab_items_df['label'] == event]['itemid'].values[0]
        event_id_df = pd.concat([event_id_df, pd.DataFrame({'itemid': event_item_id, 'event': event}, index=[0])], axis=0, ignore_index=True)

    df = df[df['itemid'].isin(event_id_df['itemid'])]
    df = df.merge(event_id_df, on='itemid', how='left')
    # df.drop(columns=['itemid'], inplace=True)
    return df

def get_vitals_of_interest(chartevents_df, d_items_df):
    df = chartevents_df.copy()
    event_list = ['Heart Rate','Non Invasive Blood Pressure systolic',
                'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean', 
                'Respiratory Rate','O2 saturation pulseoxymetry', 
                'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response']

    event_id_df = pd.DataFrame()
    for event in event_list:
        print(event)
        event_item_id = d_items_df[d_items_df['label'] == event]['itemid'].values[0]
        event_id_df = pd.concat([event_id_df, pd.DataFrame({'itemid': event_item_id, 'event': event}, index=[0])], axis=0, ignore_index=True)

    df = df[df['itemid'].isin(event_id_df['itemid'])]
    df = df.merge(event_id_df, on='itemid', how='left')
    # df.drop(columns=['itemid'], inplace=True)
    rename_dict = {
        'Non Invasive Blood Pressure systolic': 'Systolic BP',
        'Non Invasive Blood Pressure diastolic': 'Diastolic BP',
        'Non Invasive Blood Pressure mean': 'Mean BP',
        'O2 saturation pulseoxymetry': 'O2 Saturation'
    }
    df['event'] = df['event'].replace(rename_dict)
    return df

def add_time_delta_vectorized(df, admissions_df, icustays_df):
    """
    Add event time with respect to hospital admission time and ICU stay time.
    Args:
        df: labevents or chartevents DataFrame
        admissions_df: admissions DataFrame with hospital admission time
        icustays_df: icustays DataFrame with ICU stay time
    Returns:
        df: DataFrame with event times with respect to hospital admission time and ICU stay time
    """

    df = df.copy()
    
    # Determine reference time column
    if 'charttime' in df.columns:
        ref_time_col = 'charttime'
    elif 'storetime' in df.columns:
        ref_time_col = 'storetime'
    else:
        raise ValueError('DataFrame must contain either charttime or storetime column')
    
    # Check if stay_id exists
    stay_id_in_cols = 'stay_id' in df.columns
    
    # Merge with admissions to get hospital admission times
    admission_cols = ['subject_id', 'hadm_id', 'admittime']
    df = df.merge(admissions_df[admission_cols], on=['subject_id', 'hadm_id'], how='left')
    
    # Calculate hospital time delta vectorized
    df['hosp_time_delta'] = (df[ref_time_col] - df['admittime']).dt.total_seconds() / 3600
    
    # Handle ICU time delta
    if stay_id_in_cols:
        # If stay_id exists, merge directly with ICU stays
        icu_cols = ['subject_id', 'stay_id', 'intime']
        df = df.merge(icustays_df[icu_cols], on=['subject_id', 'stay_id'], how='left')
        df['icu_time_delta'] = (df[ref_time_col] - df['intime']).dt.total_seconds() / 3600
    else:
        # If no stay_id, need to find which ICU stay each event belongs to
        
        # Prepare ICU stays data
        icu_stays = icustays_df[['subject_id', 'stay_id', 'intime', 'outtime']].copy()
        
        # Create a cartesian product of events and ICU stays for the same subject
        df_with_idx = df.reset_index().rename(columns={'index': 'original_idx'})
        merged = df_with_idx.merge(icu_stays, on='subject_id', how='left')
        
        # Filter to events that fall within ICU stay time windows
        mask = (merged[ref_time_col] >= merged['intime']) & (merged[ref_time_col] <= merged['outtime'])
        valid_matches = merged[mask].copy()
        
        # Calculate ICU time delta for valid matches
        valid_matches['icu_time_delta'] = (valid_matches[ref_time_col] - valid_matches['intime']).dt.total_seconds() / 3600
        
        # Handle multiple ICU stays for same event (take the first match)
        # Sort by intime to ensure consistent selection
        valid_matches = valid_matches.sort_values(['original_idx', 'intime'])
        valid_matches = valid_matches.drop_duplicates('original_idx', keep='first')
        # Merge back the stay_id and icu_time_delta
        if not valid_matches.empty:
            df = df_with_idx.merge(
                valid_matches[['original_idx', 'stay_id', 'icu_time_delta']], 
                on='original_idx', 
                how='left'
            )
            df = df.drop(['original_idx'], axis=1)
        else:
            # If no valid matches, just add stay_id and icu_time_delta columns with None
            df = df_with_idx
            df['stay_id'] = None
            df['icu_time_delta'] = None
            df = df.drop(['original_idx'], axis=1)
    
    # Clean up temporary columns
    df = df.drop(['admittime'], axis=1, errors='ignore')
    if 'intime' in df.columns:
        df = df.drop(['intime'], axis=1)
    
    # Sort the DataFrame
    df = df.sort_values(by=['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta'])
    
    return df

def convert_events_table_to_ts_array(df):
    # Ensure 'valuenum' or 'value' columns exist
    value_column = 'valuenum' if 'valuenum' in df.columns else 'value'

    # Create a pivot table
    pivot_df = df.pivot_table(index=['hadm_id', 'hosp_time_delta'], 
                              columns='event', 
                              values=value_column, 
                              aggfunc='first').reset_index()

    # Join with the original DataFrame to get other required columns
    keys = ['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta', 'icu_time_delta']
    merged_df = pd.merge(df[keys].drop_duplicates(), pivot_df, on=['hadm_id', 'hosp_time_delta'])

    # Reorder the columns
    cols = merged_df.columns.tolist()
    cols = [col for col in keys if col in cols] + [col for col in cols if col not in keys]
    merged_df = merged_df[cols]

    # Sort the DataFrame
    merged_df.sort_values(by=['subject_id', 'hadm_id', 'stay_id', 'hosp_time_delta'], inplace=True)

    return merged_df

def create_event_uom_map(df):
    """
    Create a pd.Dataframe with event, itemid, and valueuom from labevents_df or vitals_df
    """
    df = df.copy()
    df = df[['event', 'itemid', 'valueuom']]
    df.drop_duplicates(inplace=True)
    return df

def main(args):
    # load mimic-iv admissions table
    print('Loading admissions table...')
    admissions_df = pd.read_csv(os.path.join(args.mimic_iv_dir, 'hosp', 'admissions.csv.gz'))
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

    # load mimic-iv icustays table
    print('Loading icustays table...')
    icustays_df = pd.read_csv(os.path.join(args.mimic_iv_dir, 'icu', 'icustays.csv.gz'))
    icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
    icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'])

    # load mimic-iv chartevents table
    print('Loading chartevents table...')
    chartevents_df = pd.read_csv(os.path.join(args.mimic_iv_dir, 'icu', 'chartevents.csv.gz'))
    chartevents_df['charttime'] = pd.to_datetime(chartevents_df['charttime'])
    chartevents_df['storetime'] = pd.to_datetime(chartevents_df['storetime'])

    # load mimic-iv labevents table
    print('Loading labevents table...')
    hosp_lab_events = pd.read_csv(os.path.join(args.mimic_iv_dir, 'hosp', 'labevents.csv.gz'))
    hosp_lab_events['charttime'] = pd.to_datetime(hosp_lab_events['charttime'])
    hosp_lab_events['storetime'] = pd.to_datetime(hosp_lab_events['storetime'])
    hosp_lab_events = hosp_lab_events.dropna(subset=['hadm_id'])

    # load mimic-iv d_labitems table
    print('Loading d_labitems table...')
    d_lab_items_df = pd.read_csv(os.path.join(args.mimic_iv_dir, 'hosp', 'd_labitems.csv.gz'))
    d_lab_items_df = d_lab_items_df.dropna()

    # load mimic-iv d_items table
    print('Loading d_items table...')
    d_items_df = pd.read_csv(os.path.join(args.mimic_iv_dir, 'icu', 'd_items.csv.gz'))

    # get labs of interest
    print('Getting labs of interest...')
    labevents_df = get_labs_of_interest(hosp_lab_events, d_lab_items_df)
    print('labevents_df columns', labevents_df.columns)
    # get vitals of interest
    print('Getting vitals of interest...')
    vitals_df = get_vitals_of_interest(chartevents_df, d_items_df)
    print('vitals_df columns', vitals_df.columns)

    del hosp_lab_events, chartevents_df

    print('labeevents_df')
    print(labevents_df.head())

    print('vitals_df')
    print(vitals_df.head())

    print('Adding time delta vectorized...')
    labevents_df = add_time_delta_vectorized(labevents_df, admissions_df, icustays_df)
    vitals_df = add_time_delta_vectorized(vitals_df, admissions_df, icustays_df)

    print('labeevents_df after adding time delta vectorized')
    print(labevents_df.head())

    print('vitals_df after adding time delta vectorized')
    print(vitals_df.head())

    print('Creating event uom map...')
    labevents_uom_df = create_event_uom_map(labevents_df)
    vitals_uom_df = create_event_uom_map(vitals_df)

    print('labevents_uom_df')
    print(labevents_uom_df)

    print('vitals_uom_df')
    print(vitals_uom_df)

    concat_df = pd.concat([labevents_df, vitals_df], axis=0)
    concat_uom_df = pd.concat([labevents_uom_df, vitals_uom_df], axis=0)

    print('Converting events table to time series array...')
    labevents_ts_df = convert_events_table_to_ts_array(labevents_df)
    vitals_ts_df = convert_events_table_to_ts_array(vitals_df)
    concat_ts_df = convert_events_table_to_ts_array(concat_df)

    print('labevents_ts_df')
    print(labevents_ts_df.head())

    print('vitals_ts_df')
    print(vitals_ts_df.head())

    print('concat_ts_df')
    print(concat_ts_df.head())

    print('Saving time series arrays...')
    labevents_ts_df.to_parquet(os.path.join(args.output_dir, "ts_labs_icu.parquet"), index=False)
    vitals_ts_df.to_parquet(os.path.join(args.output_dir, "ts_vitals_icu.parquet"), index=False)
    concat_ts_df.to_parquet(os.path.join(args.output_dir, "ts_labs_vitals.parquet"), index=False)

    print('Saving event uom map...')
    labevents_uom_df.to_parquet(os.path.join(args.output_dir, "uom_labs_icu.parquet"), index=False)
    vitals_uom_df.to_parquet(os.path.join(args.output_dir, "uom_vitals_icu.parquet"), index=False)
    concat_uom_df.to_parquet(os.path.join(args.output_dir, "uom_labs_vitals.parquet"), index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_iv_dir", type=str, required=True, help='Path to mimic-iv data directory (e.g. mimiciv/3.1/)')
    parser.add_argument("--output_dir", type=str, help='Path to output directory', default='data')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
