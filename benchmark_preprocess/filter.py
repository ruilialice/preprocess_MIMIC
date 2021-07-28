import sys, copy
import os
import pickle
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import shutil

# source: https://stackoverflow.com/questions/6996603/how-to-delete-a-file-or-folder
def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter patients whose episode, diagnosis and stay not match '
                                                 'for per-subject data.')
    parser.add_argument('patient_path', type=str, help='Directory containing MIMIC-III benchmark root.')
    args, _ = parser.parse_known_args()
    patient_path = args.patient_path
    patientsList_temp = list(filter(str.isdigit, os.listdir(args.patient_path)))
    patientsList = [int(x) for x in patientsList_temp]

    # record the subject_id that need to be removed
    need_remove = set()

    for pid in tqdm(patientsList, desc='Iterating over patients'):
        stays_path = os.path.join(patient_path, str(pid)+'/stays.csv')
        if not os.path.isfile(stays_path):
            need_remove.add(pid)
            continue
        diagnoses_path = os.path.join(patient_path, str(pid)+'/diagnoses.csv')
        if not os.path.isfile(diagnoses_path):
            need_remove.add(pid)
            continue
        stays_df = pd.read_csv(stays_path)
        diagnoses_df = pd.read_csv(diagnoses_path)
        convert_dict = {'HADM_ID': int,
                        'ICD9_CODE': str
                        }
        diagnoses_df = diagnoses_df.astype(convert_dict)

        for idx, row in stays_df.iterrows():
            stay_hadm = int(row[1])
            stay_icustay = int(row[2])

            # check if the icustay in episode_i equals to stay_icustay
            episode_path = os.path.join(patient_path, str(pid)+'/episode'+str(idx+1)+'.csv')
            if not os.path.isfile(episode_path):
                need_remove.add(pid)
                break
            episode_df = pd.read_csv(episode_path)
            if episode_df.shape[0] < 1:
                need_remove.add(pid)
                break
            cur_icustay = int(episode_df.loc[0, 'Icustay'])
            if stay_icustay != cur_icustay:
                need_remove.add(pid)
                break

            # check if episode_i_timeseries exist
            timeseries_path = os.path.join(patient_path, str(pid)+'/episode'+str(idx+1)+'_timeseries.csv')
            if not os.path.isfile(timeseries_path):
                need_remove.add(pid)
                break
            timeseries_df = pd.read_csv(timeseries_path)
            convert_dict = {'Hours': int}
            timeseries_df = timeseries_df.astype(convert_dict)
            if timeseries_df[timeseries_df.Hours >= 0].shape[0] < 1:
                need_remove.add(pid)
                break

            # check if current hadm has disgnoses code
            temp = diagnoses_df.HADM_ID
            if diagnoses_df[diagnoses_df.HADM_ID == stay_hadm].shape[0] < 1:
                need_remove.add(pid)
                break

    print("{} patients need to be removed".format(str(len(need_remove))))

    print(need_remove)

    need_remove_list = list(need_remove)
    for pid in need_remove_list:
        dir_path = os.path.join(patient_path, str(pid))
        remove(dir_path)





