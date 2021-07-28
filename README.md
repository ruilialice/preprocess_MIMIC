# Preprocess MIMIC benchmark

This project helps to preprocess the MIMIC benchmark dataset, it could collect icd9 codes and the clinical data at the same time. The final data contains the following information:

- pids: the list contains SUBJECT_ID 
- demographic_list: the list contains the demographic data of the patients
- hadm_id_list: the list contains the HADM_ID
- mortality_list: the list contains the mortality status of the patient in the current visit
- visit_interval_list: the list contains the time interval between the current visit and the first visit
- icd9_list: the list contains the diagnosis code for every visit
- clinic_feature_list: the list contains the clinical features
- chart_time_list: the list contains the chart time of the clinical events

## Requirements

Firstly, you  should acquire the MIMIC data from https://mimic.physionet.org/
- numpy 
- pandas 0.23.4

## Preprocess the data

Here are the steps to preprocess the data.

1. Follow the code in https://github.com/YerevaNN/mimic3-benchmarks, in the section "Building a benchmark", finish the first four steps. 
    For step 2,  run the following command. Here, {PATH TO Benchmark root} is the path that you want to save the benchmark root files.
    ```sh
    python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} {PATH TO Benchmark root}
    ```
    For step 3, run the following command. Here, {PATH TO Benchmark root} is the same as step 2.
    ```sh
    python -m mimic3benchmark.scripts.validate_events {PATH TO Benchmark root}
    ```
    For step 4, run the following command. Here, {PATH TO Benchmark root} is the same as step 2.
    ```sh
    python -m mimic3benchmark.scripts.extract_episodes_from_subjects {PATH TO Benchmark root}
    ```

2. After finishing the previous steps, you could clone the repo.
    ```sh
    git clone https://github.com/YerevaNN/mimic3-benchmarks/
    cd preprocess/benchmark_preprocess
    ```

3. The following command helps to filter patients who do not have any diagnosis code or clinical features in a visit.
    ```sh
    python filter.py {PATH TO Benchmark root}
    ```
4. The following command helps to collect the patient information. You could adjust the parameters single_visit, even_interval, full_icd9, whole_hours_flag and selected_hours according to your setting.
    ```sh
    python preprocess_benchmark_mimic.py {PATH TO MIMIC-III CSVs} {PATH TO Benchmark root} --single_visit=False --even_interval=False --save_as_ndarray=True --full_icd9=True --whole_hours_flag=True --selected_hours=48
    ```

## Loading the preprocessed data

You could use the following code to load the data and the dictionary whose key is the icd9 code, and value is the index.
```sh
import pickle
# loading data
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
pids, demographic_list, hadm_id_list, mortality_list, visit_interval_list, icd9_list, clinic_feature_list, chart_time_list = data
# loading the dictionary whose key is the icd9 code, and value is the index
with open('icd_dict.pkl', 'rb') as f:
    icd_dict = pickle.load(f)
```


