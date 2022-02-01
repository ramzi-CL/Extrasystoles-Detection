# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:36:30 2022

@author: RamziAbdelhafidh
"""

import wfdb
import pandas as pd
import os
import glob
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler



def load_arrhythmia_files(mitdb_folder_path):
    files_extension = '/*.dat'
    ecg_files = glob.glob(mitdb_folder_path+files_extension)
    ecg_files = [ecg_file[-7:-4] for ecg_file in ecg_files]
    os.chdir(mitdb_folder_path+"/")
    df = pd.DataFrame()
    for ecg_file in ecg_files:
        record = wfdb.rdrecord(ecg_file, channels=[0])
        annotation = wfdb.rdann(record_name=ecg_file, extension='atr',
                                summarize_labels=True,
                                return_label_elements=['symbol'])
        signal = record.p_signal.flatten()
        record_name = record.record_name
        annotation_symbols = annotation.symbol
        annotation_times = annotation.sample
        df = df.append({'record_name': record_name,
                        'signal': signal,
                        'annotation_symbols': annotation_symbols,
                        'annotation_times': annotation_times},
                       ignore_index=True)
    os.chdir("../notebooks/")
    return df


def train_test_split(df, ds_train_names, ds_test_names):
    df_train = df.loc[df['record_name'].isin(ds_train_names)]
    df_test = df.loc[df['record_name'].isin(ds_test_names)]
    return df_train, df_test


class Butterlife():
    """ Butterworth filter class """

    def __init__(self):
        pass

    def check_params(self, order):
        """ Check parameters """

        assert order < 5, "Filter order is too high (order max = 4, given = "\
            + str(order) + ")"

    def assign_params(self, sig, fs, fc, order, ftype, padding):
        """ Assign parameters """

        self.sig_ = sig
        self.fs_ = fs
        self.fc_ = fc
        self.order_ = order
        self.ftype_ = ftype
        self.padding_ = padding
        self.a_ = np.zeros(order+1)
        self.b_ = np.zeros(order+1)
        self.c_ = 1 / math.tan(fc * math.pi / fs)

        # ---- Get butterworth coefficients
        self.get_coefficients()

    def filt(self, sig, fs, fc, order, ftype, padding=True):
        """ Filter butterworth : low and high pass filter

        Parameters
        ------------------------------------------
        sig: input signal
        fs: Sampling frequency Hz
        fc: cut off frequency
        ftype: Filter type ("low "or "high")
        padding: Flag for keeping original signal size

        Returns
        ------------------------------------------
        filtered signal

        """

        # Chek params
        self.check_params(order)

        # Assign params
        self.assign_params(sig, fs, fc, order, ftype, padding)

        # ---- Get butterworth coefficients
        a = self.a_
        b = self.b_

        # ---- Set params for filtering
        n = len(sig)
        # Extend signal size to init filtering
        sig_padding = np.zeros((n + order + 1, ))
        for i in range(n):
            sig_padding[i + order] = sig[i]
        sig_padding[:order] = sig[0]
        sig_padding[(n-1):] = sig[-1]

        # ---- Y: 1st filter step
        sig_filt1 = np.zeros(n + order + 1)
        sig_filt1[:order] = sig[0]

        for i in range(order, n + 1):
            y = 0
            for k in range(len(b)):
                y += b[k]*sig_padding[i-k] - a[k]*sig_filt1[i-k]
            sig_filt1[i] = y
        sig_filt1[n:] = sig_filt1[n]

        # ---- Z: 2nd filter step
        sig_filt2 = np.zeros(n + 1)
        for i in range(order):
            sig_filt2[n-i] = sig_filt1[n + order - i]

        for i in range(n - order, -1, -1):
            z = 0
            for k in range(len(b)):
                z += b[k]*sig_filt1[i+k+order] - a[k]*sig_filt2[i+k]
            sig_filt2[i] = z

        # ---- Remove dephasing
        sig_filt = np.zeros(n - 1)
        for i in range(len(sig_filt)):
            sig_filt[i] = sig_filt2[i+1]

        # Add sample to keep original signal size
        if padding:
            padding_sample = sig_filt[0]
            sig_filt_final = []
            sig_filt_final.append(padding_sample)
            sig_filt_final.extend(sig_filt)
            sig_filt_final = np.array(sig_filt_final)

        self.sig_filt_ = sig_filt_final

        return sig_filt_final

    def get_coefficients(self):
        """ get butterworth filter coeffcients """

        ftype = self.ftype_

        self.get_lowpass_coefficients()

        if ftype == 'high':
            self.get_highpass_coefficients()

    def get_lowpass_coefficients(self):
        """ get butterworth lowpass filter coeffcients """
        order = self.order_
        a = self.a_
        b = self.b_
        c = self.c_

        if order == 1:
            d = c + 1

            # Coefs a
            a[0] = 1
            a[1] = (1 - c)/d

            # Coefs b
            b[0] = 1/d
            b[1] = b[0]

        elif order == 2:
            q0 = np.sqrt(2)  # resonance term
            d = c**2 + q0*c + 1

            # Coefs a
            a[0] = 1
            a[1] = (- 2*c**2 + 0*c + 2)/d
            a[2] = (1*c**2 - q0*c + 1)/d

            # Coefs b
            b[0] = 1/d
            b[1] = 2*b[0]
            b[2] = 1*b[0]

        elif order == 3:
            d = c**3 + 2*c**2 + 2*c + 1

            # Coefs a
            a[0] = 1
            a[1] = (- 3*c**3 - 2*c**2 + 2*c + 3)/d
            a[2] = (3*c**3 - 2*c**2 - 2*c + 3)/d
            a[3] = (- 1*c**3 + 2*c**2 - 2*c + 1)/d

            # Coefs b
            b[0] = 1/d
            b[1] = 3*b[0]
            b[2] = 3*b[0]
            b[3] = 1*b[0]

        elif order == 4:
            q0 = 0.7654
            q1 = 1.8478
            e = (q0 + q1)*c**3
            f = (2 + q0*q1)*c**2
            g = (q0 + q1)*c

            d = c**4 + e + f + g + 1

            # Coefs a
            a[0] = 1
            a[1] = (- 4*c**4 - 2*e + 0*f + 2*g + 4)/d
            a[2] = (6*c**4 + 0*e - 2*f + 0*g + 6)/d
            a[3] = (- 4*c**4 + 2*e + 0*f - 2*g + 4)/d
            a[4] = (1*c**4 - 1*e + 1*f - 1*g + 1)/d

            # Coefs b
            b[0] = 1/d
            b[1] = 4*b[0]
            b[2] = 6*b[0]
            b[3] = 4*b[0]
            b[4] = 1*b[0]

        self.a_ = a
        self.b_ = b

    def get_highpass_coefficients(self):
        """ get butterworth highpass filter coeffcients """

        order = self.order_
        b = self.b_
        c = self.c_

        if order == 1:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order

        elif order == 2:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order
            b[2] = b[2] * (-c)**order

        elif order == 3:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order
            b[2] = b[2] * (-c)**order
            b[3] = - b[3] * (-c)**order

        elif order == 4:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order
            b[2] = b[2] * (-c)**order
            b[3] = - b[3] * (-c)**order
            b[4] = b[4] * (-c)**order

        self.b_ = b


def filter_median(sig, kernel_size=3):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    sig = np.array(sig)
    assert kernel_size % 2 == 1, "Median filter kernel_size should be odd."
    assert sig.ndim == 1, "Input must be one-dimensional."

    k = (kernel_size - 1) // 2
    y = np.zeros((len(sig), kernel_size), dtype=sig.dtype)
    y[:, k] = sig
    for i in range(k):
        j = k - i
        y[j:, i] = sig[:-j]
        y[:j, i] = sig[0]
        y[:-j, -(i+1)] = sig[j:]
        y[-j:, -(i+1)] = sig[-1]

    sig_filt = np.median(y, axis=1)

    return sig_filt


def filter_ecg(sig, fs):
    # Remove baseline
    inter_1 = 0.465
    inter_2 = 0.945
    sig_med = filter_median(sig, kernel_size=2*int(inter_1*fs/2)+1)
    sig_med = filter_median(sig_med, kernel_size=2*int(inter_2*fs/2)+1)
    sig_med = sig - sig_med

    # Discret Filter
    butter = Butterlife()
    sig_low = butter.filt(sig_med, fs=fs, fc=40, order=4, ftype='low',
                          padding=True)
    sig_filt = butter.filt(sig_low, fs=fs, fc=0.5, order=4, ftype='high',
                           padding=True)

    return sig_filt


def filter_df_ecgs(data, fs, sig_col_name):
    data_ = data.copy()
    data_['filtered_ecg'] = data_.apply(
        lambda x: filter_ecg(x[sig_col_name], fs), axis=1)
    data_.drop(columns=[sig_col_name], axis=1, inplace=True)
    return data_


def extract_template_signal(i, samples, signal, fs):
    previous_sample = samples[i-1]
    next_sample = samples[i+1]
    n_next_sample = samples[i+2]
    born_inf = max(0, previous_sample - int(0.277*fs))
    born_sup = min(len(signal), next_sample + int(0.472*fs),
                   n_next_sample-int(0.208*fs))
    template_signal = signal[born_inf:born_sup]
    return template_signal


def compute_rr(i, samples):
    previous_sample = samples[i-1]
    current_sample = samples[i]
    next_sample = samples[i+1]
    rr_1 = current_sample - previous_sample
    rr_2 = next_sample - current_sample
    rr_1_per = rr_1 / (rr_1 + rr_2)
    rr_2_per = rr_2 / (rr_1 + rr_2)
    return rr_1_per, rr_2_per


def determine_annotation(i, ann, normal_beats, extrasystole_beats):
    annotation = -1
    if (ann[i] in extrasystole_beats) or (ann[i+1] in extrasystole_beats):
        annotation = 1
    elif (ann[i] in normal_beats) and (ann[i+1] in normal_beats) and \
            (ann[i-1] in normal_beats):
        annotation = 0
    return annotation


def split_signal(row, normal_beats, extrasystole_beats, fs):
    dataframe = pd.DataFrame()
    record_name = row['record_name']
    signal = row['filtered_ecg']
    ann = row['annotation_symbols']
    samples = list(row['annotation_times'])
    allowed_beats = normal_beats + extrasystole_beats
    for i in range(1, len(ann)-2):
        anns_seq = [ann[i-1], ann[i], ann[i+1], ann[i+2]]
        if all(e in allowed_beats for e in anns_seq):
            template_signal = extract_template_signal(i, samples, signal, fs)
            rr_1, rr_2 = compute_rr(i, samples)
            annotation = determine_annotation(
                i, ann, normal_beats, extrasystole_beats)
            if annotation in [0, 1]:
                dataframe = dataframe.append({'record_name': record_name,
                                              'template': template_signal,
                                              'rr_1': rr_1,
                                              'rr_2': rr_2,
                                              'label': annotation},
                                             ignore_index=True)
    return dataframe


def generate_templates(data, normal_beats, extrasystole_beats, fs):
    templates_df = pd.DataFrame()
    list_of_df = list(data.apply(
        lambda x: split_signal(x, normal_beats, extrasystole_beats, fs),
        axis=1))
    for df in list_of_df:
        templates_df = pd.concat([templates_df, df], ignore_index=True)
    return templates_df


def scale_template(row):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    signal = row['template']
    scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    return scaled_signal


def minmax_scale(data):
    data_ = data.copy()
    data_['template'] = data_.apply(lambda x: scale_template(x), axis=1)
    return data_
