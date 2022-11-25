from statistics import mean

import numpy as np


def calc_R_period(signal, r_peaks, measurements):
    r_onset = measurements['ECG_R_Onsets']
    r_offset = measurements['ECG_R_Offsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(r_onset[i]) and not np.isnan(r_offset[i]):
            periods.append(r_offset[i] - r_onset[i])
    return mean(periods) if len(periods) else 0.0


def calc_R_amplitude(signal, r_peaks, measurements):
    amplitudes = []
    for peak in r_peaks:
        amplitudes.append(signal[peak])
    return mean(amplitudes) if len(amplitudes) else 0.0


def calc_Q_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_Q_Peaks']
    amplitudes = []
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
    return mean(amplitudes) if len(amplitudes) else 0.0


def calc_S_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_S_Peaks']
    amplitudes = []
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
    return mean(amplitudes)


def calc_T_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_T_Peaks']
    amplitudes = []
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
    return mean(amplitudes) if len(amplitudes) else 0.0


def calc_P_amplitude(signal, r_peaks, measurements):
    x_peaks = measurements['ECG_P_Peaks']
    amplitudes = []
    for peak in x_peaks:
        if not np.isnan(peak):
            amplitudes.append(signal[peak])
    return mean(amplitudes) if len(amplitudes) else 0.0


def calc_T_period(signal, r_peaks, measurements):
    t_onset = measurements['ECG_T_Onsets']
    t_offset = measurements['ECG_T_Offsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(t_onset[i]) and not np.isnan(t_offset[i]):
            periods.append(t_offset[i] - t_onset[i])
    return mean(periods) if len(periods) else 0.0


def calc_P_period(signal, r_peaks, measurements):
    p_onset = measurements['ECG_P_Onsets']
    p_offset = measurements['ECG_P_Offsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(p_onset[i]) and not np.isnan(p_offset[i]):
            periods.append(p_offset[i] - p_onset[i])
    return mean(periods) if len(periods) else 0.0


def calc_Q_period(signal, r_peaks, measurements):
    r_onset = measurements['ECG_R_Onsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(r_onset[i]):
            periods.append(r_peaks[i] - r_onset[i])
    return mean(periods) if len(periods) else 0.0


def calc_S_period(signal, r_peaks, measurements):
    r_offset = measurements['ECG_R_Offsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(r_offset[i]):
            periods.append(r_offset[i] - r_peaks[i])
    return mean(periods) if len(periods) else 0.0

def calc_PR_interval(signal, r_peaks, measurements):
    p_onset = measurements['ECG_P_Onsets']
    r_onset = measurements['ECG_R_Onsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(p_onset[i]) and not np.isnan(r_onset[i]):
            periods.append(r_onset[i] - p_onset[i])
    return mean(periods) if len(periods) else 0.0

def calc_QT_interval(signal, r_peaks, measurements):
    t_offset = measurements['ECG_T_Offsets']
    r_onset = measurements['ECG_R_Onsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(t_offset[i]) and not np.isnan(r_onset[i]):
            periods.append(t_offset[i] - r_onset[i])
    return mean(periods) if len(periods) else 0.0

def calc_PR_segment(signal, r_peaks, measurements):
    p_offset = measurements['ECG_P_Offsets']
    r_onset = measurements['ECG_R_Onsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(p_offset[i]) and not np.isnan(r_onset[i]):
            periods.append(r_onset[i] - p_offset[i])
    return mean(periods) if len(periods) else 0.0

def calc_ST_segmentl(signal, r_peaks, measurements):
    t_onset = measurements['ECG_T_Onsets']
    r_offset = measurements['ECG_R_Offsets']
    periods = []
    for i in range(len(r_peaks)):
        if not np.isnan(r_offset[i]) and not np.isnan(t_onset[i]):
            periods.append(t_onset[i] - r_offset[i])
    return mean(periods) if len(periods) else 0.0


def get_nk_features(signal, r_peaks, measurements):
    features = np.array([
        calc_R_period(signal, r_peaks, measurements),
        calc_R_amplitude(signal, r_peaks, measurements),
        calc_Q_amplitude(signal, r_peaks, measurements),
        calc_S_amplitude(signal, r_peaks, measurements),
        calc_T_amplitude(signal, r_peaks, measurements),
        calc_P_amplitude(signal, r_peaks, measurements),
        calc_T_period(signal, r_peaks, measurements),
        calc_P_period(signal, r_peaks, measurements),
        calc_Q_period(signal, r_peaks, measurements),
        calc_S_period(signal, r_peaks, measurements),
        calc_PR_interval(signal, r_peaks, measurements),
        calc_QT_interval(signal, r_peaks, measurements),
        calc_PR_segment(signal, r_peaks, measurements),
        calc_ST_segmentl(signal, r_peaks, measurements)
    ])
    return features


def add_basic_info(row, arr, name):
    if len(arr) > 0:
        row[f'std_{name}'] = arr.std()
        row[f'mean_{name}'] = arr.mean()
        row[f'median_{name}'] = np.median(arr)
        row[f'max_{name}'] = arr.max()
        row[f'min_{name}'] = arr.min()
        row[f'range_{name}'] = arr.max() - arr.min()
    return row
