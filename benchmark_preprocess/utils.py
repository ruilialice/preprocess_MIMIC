import numpy as np
import pandas as pd
import math

def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4] + '.' + dxStr[4:]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3] + '.' + dxStr[3:]
        else:
            return dxStr

def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr

def Process_Clinic_Feature(cur_clinic_df, begin_idx, end_idx, config, args, timestep=1.0):
    channel_to_id = {}
    tmp = 0
    for channel in config["id_to_channel"]:
        channel_to_id[channel] = tmp
        tmp += 1

    header = list(cur_clinic_df.columns.values)
    assert header[1:] == config["id_to_channel"]
    assert header[0] == "Hours"
    eps = 1e-6
    ts = [float(t) for t in cur_clinic_df.iloc[:, 0]]
    for i in range(len(ts) - 1):
        assert ts[i] < ts[i + 1] + eps

    max_hours = 500 if args.whole_hours_flag else args.selected_hours

    if args.even_interval:
        visit_time = math.ceil(ts[-1])
        if visit_time > max_hours:
            visit_time = int(max_hours)
        N_bins = int(visit_time)
    else:
        start = 0
        temp_idx = 0
        for idx in range(len(ts)):
            if 0 <= ts[idx] <= max_hours:
                temp_idx += 1
            elif ts[idx] > max_hours:
                break
            elif ts[idx] < 0:
                start += 1

        N_bins = temp_idx if temp_idx < 500 else 500

    data = np.zeros(shape=(N_bins, end_idx[-1] + 1), dtype=float)
    mask = np.zeros(shape=(N_bins, len(config["id_to_channel"])), dtype=int)
    original_value = [["" for j in range(len(config["id_to_channel"]))] for i in range(N_bins)]

    def write(data, bin_id, channel, value, begin_pos):
        channel_id = channel_to_id[channel]
        if config["is_categorical_channel"][channel]:
            # print(channel)
            # print(type(value))
            if channel == 'Glascow coma scale total' and type(value) != type("a"):
                temp_val = value
                value = str(int(temp_val))
            elif channel == 'Capillary refill rate' and type(value) != type("a"):
                temp_val = 1.0 * value
                value = str(temp_val)
            category_id = config["possible_values"][channel].index(value)
            N_values = len(config["possible_values"][channel])
            one_hot = np.zeros((N_values,))
            one_hot[category_id] = 1
            for pos in range(N_values):
                data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
        else:
            data[bin_id, begin_pos[channel_id]] = float(value)


    for idx, row in cur_clinic_df.iterrows():
        if args.even_interval:
            t = float(row[header[0]])
            if t < 0:
                continue
            if t > max_hours + eps:
                continue
            bin_id = int(t / timestep - eps)
        else:
            if idx >= start + N_bins:
                continue
            elif idx < start:
                continue
            else:
                bin_id = idx - start
        assert 0 <= bin_id < N_bins

        for j in range(1, len(header)):
            channel = header[j]
            if pd.isnull(row[channel]):
                continue
            channel_id = channel_to_id[channel]
            mask[bin_id][channel_id] = 1
            write(data, bin_id, channel, row[channel], begin_idx)
            original_value[bin_id][channel_id] = row[channel]

    prev_values = [[] for i in range(len(header[1:]))]
    for bin_id in range(N_bins):
        for channel in header[1:]:
            channel_id = channel_to_id[channel]
            if mask[bin_id][channel_id] == 1:
                prev_values[channel_id].append(original_value[bin_id][channel_id])
                continue
            if len(prev_values[channel_id]) == 0:
                imputed_value = config["normal_values"][channel]
            else:
                imputed_value = prev_values[channel_id][-1]
            write(data, bin_id, channel, imputed_value, begin_idx)

    if args.even_interval:
        chart_time = [i for i in range(visit_time)]
    else:
        start = 0
        for idx in range(len(ts)):
            if ts[idx] >= 0:
                start = idx
                break
        chart_time = ts[start:start+temp_idx]

    data = np.hstack([data, mask.astype(np.float32)])

    return data, chart_time
