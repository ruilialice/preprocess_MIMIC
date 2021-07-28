import sys, copy
import os
import json
import pickle
import pandas as pd
import argparse
from tqdm import tqdm
from utils import convert_to_icd9, convert_to_3digit_icd9, Process_Clinic_Feature
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mimic_original_path', type=str, help='Directory containing original MIMIC-III files.')
    parser.add_argument('patient_path', type=str, help='Directory containing MIMIC-III benchmark root.')
    parser.add_argument('--mimic_resource_path', type=str, help='Directory containing MIMIC-III configure files.',
                        default='mimic_resources/')
    parser.add_argument('--single_visit', type=bool, help='flag whether collect single visit data', default=False)
    parser.add_argument('--even_interval', type=bool, help='flag whether the clinical data is collected in even '
                                                           'interval', default=False)
    parser.add_argument('--save_as_ndarray', type=bool, help='flag whether the clinical data is saved as ndarray',
                        default=True)
    parser.add_argument('--full_icd9', type=bool,
                        help='flag whether collecting full icd9 code', default=True)
    parser.add_argument('--whole_hours_flag', type=bool,
                        help='if true, using the whole clinical data in the visit; '
                             'if false, using selected hours in the visit', default=True)
    parser.add_argument('--selected_hours', type=int,
                        help='using the first 24/48 hours clinical data', default=48)

    args, _ = parser.parse_known_args()

    patientsList = list(filter(str.isdigit, os.listdir(args.patient_path)))
    patientsList = [int(x) for x in patientsList]
    with open(os.path.join(args.mimic_resource_path, 'discretizer_config.json')) as f:
        config = json.load(f)

    # used for collecting clinical features
    if args.save_as_ndarray:
        num_channel = len(config["id_to_channel"])
        begin_idx = [0 for i in range(num_channel)]
        end_idx = [0 for i in range(num_channel)]

        for i in range(num_channel):
            channel_name = config["id_to_channel"][i]
            if config["is_categorical_channel"][channel_name] == False:
                end_idx[i] = begin_idx[i]
            else:
                temp_num_idx = len(config["possible_values"][channel_name])
                end_idx[i] = begin_idx[i] + temp_num_idx - 1

            if i < num_channel - 1:
                begin_idx[i + 1] = end_idx[i] + 1

        # clinical features contain 76 dimensions, header represents the meaning of each dimension,
        # the last 16 dimension is the indicator, if the value is charted in csv, the indicator is 1,
        # if the value is permuted using the previous value, the indicator is 0.
        headers = []
        for i in config["id_to_channel"]:
            if config["is_categorical_channel"][i] == False:
                headers.append(str(i))
            else:
                possible_value = config["possible_values"][i]
                for value in possible_value:
                    headers.append(str(i) + "->" + str(value))
        for i in config["id_to_channel"]:
            headers.append(str(i) + " mask")

    pids = []  # subject_id
    demographic_list = []  # demographic data
    hadm_id_list = []   # visit_id
    mortality_list = []  # mortality in this hospital stay
    visit_interval_list = []  # only for multiple visit
    icd9_list = []  # icd9 code, using index
    clinic_feature_list = []  # clinical features, details in channel_info.json and discretizer_config.json
    chart_time_list = []  # the chart time of clinical features
    icd_dict = {}  # key is icd9, value is idx

    admission_df = pd.read_csv(os.path.join(args.mimic_original_path, 'ADMISSIONS.csv'),
                               usecols=['SUBJECT_ID', 'HADM_ID', 'ADMISSION_LOCATION', 'ADMISSION_TYPE', 'RELIGION', 'MARITAL_STATUS'])

    # collect single visit data
    if args.single_visit:
        for patient in tqdm(patientsList, desc='Iterating over patients'):
            stays_df = pd.read_csv(os.path.join(args.patient_path, str(patient), 'stays.csv'))
            first_intime = pd.to_datetime(stays_df.loc[0, 'INTIME'])
            diagnoses_df = pd.read_csv(os.path.join(args.patient_path, str(patient), 'diagnoses.csv'),
                                       dtype={'ICD9_CODE': str})

            for idx, row in stays_df.iterrows():
                pids.append(int(row['SUBJECT_ID']))  # patient id
                demographic = [row['GENDER'], float(row['AGE']), row['ETHNICITY']]
                temp = admission_df.loc[admission_df['HADM_ID'] == row['HADM_ID']].drop_duplicates().reset_index(drop=True)
                temp_series = temp.loc[0]
                demographic.extend([temp_series['ADMISSION_LOCATION'], temp_series['ADMISSION_TYPE'],
                                    temp_series['RELIGION'], temp_series['MARITAL_STATUS']])
                demographic_list.append(demographic)
                hadm_id_list.append(int(row['HADM_ID']))  # visit id
                mortality_list.append(int(row['MORTALITY']))   # mortality in this hospital stay
                current_intime = pd.to_datetime(row['INTIME'])
                interval = (current_intime - first_intime).days
                visit_interval_list.append(interval)  # visit interval, current day minus first time intime

                cur_icd_list = []
                icd_list = diagnoses_df.loc[diagnoses_df['HADM_ID'] == row['HADM_ID']]['ICD9_CODE'].tolist()
                for icd in icd_list:
                    if args.full_icd9:
                        code = convert_to_icd9(icd)
                    else:
                        code = convert_to_3digit_icd9(icd)
                    if code in icd_dict:
                        cur_icd_list.append(icd_dict[code])
                    else:
                        icd_dict[code] = len(icd_dict)
                        cur_icd_list.append(icd_dict[code])
                icd9_list.append(cur_icd_list)

                clinical_df = pd.read_csv(os.path.join(args.patient_path, str(patient),
                                                       'episode'+str(idx+1)+'_timeseries.csv'))
                if args.save_as_ndarray:
                    clinic_feature, chart_time = Process_Clinic_Feature(cur_clinic_df=clinical_df, begin_idx=begin_idx,
                                       end_idx=end_idx, config=config, args=args)
                    clinic_feature_list.append(clinic_feature)
                    chart_time_list.append(chart_time)

    # collect multiple visit data
    if not args.single_visit:
        for patient in tqdm(patientsList, desc='Iterating over patients'):
            stays_df = pd.read_csv(os.path.join(args.patient_path, str(patient), 'stays.csv'))
            if stays_df.shape[0] < 2:
                continue
            pids.append(patient)
            first_intime = pd.to_datetime(stays_df.loc[0, 'INTIME'])
            diagnoses_df = pd.read_csv(os.path.join(args.patient_path, str(patient), 'diagnoses.csv'),
                                       dtype={'ICD9_CODE': str})

            pid_demographic = []  # demographic data
            pid_hadm_id = []   # visit_id
            pid_mortality = []  # mortality in this hospital stay
            pid_visit_interval = []  # only for multiple visit
            pid_icd9 = []  # icd9 code, using index
            pid_clinic_feature = []  # clinical features, details in channel_info.json and discretizer_config.json
            pid_chart_time = []  # the chart time of clinical features

            for idx, row in stays_df.iterrows():
                demographic = [row['GENDER'], float(row['AGE']), row['ETHNICITY']]
                temp = admission_df.loc[admission_df['HADM_ID'] == row['HADM_ID']].drop_duplicates().reset_index(drop=True)
                temp_series = temp.loc[0]
                demographic.extend([temp_series['ADMISSION_LOCATION'], temp_series['ADMISSION_TYPE'],
                                    temp_series['RELIGION'], temp_series['MARITAL_STATUS']])
                pid_demographic.append(demographic)
                pid_hadm_id.append(int(row['HADM_ID']))  # visit id
                pid_mortality.append(int(row['MORTALITY']))   # mortality in this hospital stay
                current_intime = pd.to_datetime(row['INTIME'])
                interval = (current_intime - first_intime).days
                pid_visit_interval.append(interval)  # visit interval, current day minus first time intime

                cur_icd_list = []
                icd_list = diagnoses_df.loc[diagnoses_df['HADM_ID'] == row['HADM_ID']]['ICD9_CODE'].tolist()
                for icd in icd_list:
                    if args.full_icd9:
                        code = convert_to_icd9(icd)
                    else:
                        code = convert_to_3digit_icd9(icd)
                    if code in icd_dict:
                        cur_icd_list.append(icd_dict[code])
                    else:
                        icd_dict[code] = len(icd_dict)
                        cur_icd_list.append(icd_dict[code])
                pid_icd9.append(cur_icd_list)

                clinical_df = pd.read_csv(os.path.join(args.patient_path, str(patient),
                                                       'episode'+str(idx+1)+'_timeseries.csv'))
                if args.save_as_ndarray:
                    clinic_feature, chart_time = Process_Clinic_Feature(cur_clinic_df=clinical_df, begin_idx=begin_idx,
                                       end_idx=end_idx, config=config, args=args)
                    pid_clinic_feature.append(clinic_feature)
                    pid_chart_time.append(chart_time)

            demographic_list.append(pid_demographic)
            hadm_id_list.append(pid_hadm_id)
            mortality_list.append(pid_mortality)
            visit_interval_list.append(pid_visit_interval)
            icd9_list.append(pid_icd9)
            clinic_feature_list.append(pid_clinic_feature)
            chart_time_list.append(pid_chart_time)

    benchmark_mimic = (pids, demographic_list, hadm_id_list, mortality_list, visit_interval_list,
                      icd9_list, clinic_feature_list, chart_time_list)

    with open('data.pkl', 'wb') as f:
        pickle.dump(benchmark_mimic, f)

    with open('icd_dict.pkl', 'wb') as f:
        pickle.dump(icd_dict, f)


