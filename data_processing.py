import numpy as np
import pandas as pd
import os, time, math, random
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
from collections import Counter
from scipy.signal import butter, lfilter, medfilt
from numpy.fft import rfft, rfftfreq, fft, fftfreq, irfft
from sklearn import preprocessing

''' 
Alternative 1 of generating windows segments:

Uses previous n sequence length acceleration data to predict next time step target (normal, stop)
Try example 3 from following link for current timestamp --> target timestep prediction
https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array

Alternative 2 of generating windows segments, uses tf.datasets:

windows_ds = tf.data.Dataset.from_tensor_slices(windows)
targets_ds = tf.data.Dataset.from_tensor_slices(np.array(targets))
ans = tf.data.Dataset.zip((windows_ds, targets_ds))
ans = ans.shuffle(len(targets))
ans = ans.batch(batch_size)
'''

class DataProcess():
    ''' Uses previous n sequence length acceleration data to predict next time step target (normal, stop)
    Try example 3 from following link for current timestamp --> target timestep prediction
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array
    '''
    def __init__(self, data, patient_id):
        self.timestep = data.iloc[:, 0]
        self.input = self.data_process(np.array(data.iloc[:, 1:10]))
        self.target = np.array(data.iloc[:, 10] - 1)
        self.length = len(self.target)
        self.patient_id = patient_id

    # median and low pass filter, then normalize
    def data_process(self, data):
        med = medfilt(data, kernel_size=(3,1))
        butter = self.butter_lowpass_filter(med, 20, 5)
        preprocessed = None
        min_max_scaler = preprocessing.MinMaxScaler()
        for i in range(data.shape[-1]):
        butter_scaled = min_max_scaler.fit_transform(butter[:,i].reshape(-1, 1))
        if preprocessed is None:
            preprocessed = butter_scaled
        else:
            preprocessed = np.column_stack((preprocessed,butter_scaled))
        return preprocessed

    def butter_lowpass_filter(self, data, low_cut, order=5):
        b, a = butter(order, low_cut, btype='low', fs=64)
        y = lfilter(b, a, data)
        return y

    # 
    def get_segments(self, sequence_length=128, stride=64, sampling_rate=1, batch_size=256):
        windows = None
        targets = []
        for i in range(0, self.length - sequence_length, 1):
        if i % stride == 0:
            if windows is None:
            windows = np.array([self.input[i:i+sequence_length, ...]])
            else:
            windows = np.concatenate((windows,[self.input[i:i+sequence_length, ...]]), axis=0)
            targets.append(max(self.target[i:i+sequence_length]))
        else:
            # Sample more negative results to balance out target class distribution
            if i % 32 == 0:
            if max(self.target[i:i+sequence_length]) == 1:
                if windows is None:
                windows = np.array([self.input[i:i+sequence_length, ...]])
                else:
                windows = np.concatenate((windows,[self.input[i:i+sequence_length, ...]]), axis=0)
                targets.append(1)
        print("patient ID: ", self.patient_id + 1)  
        print("Training Target Distribution: ", sorted(Counter(targets).items(), key=lambda x:(x[0])))
        return windows, np.array(targets)

    def get_testing_data(self, sequence_length=128, stride=64, sampling_rate=1, batch_size=32):
        '''
        Returns window segments, target labels, and class distribution with even sampling
        '''
        windows = None
        targets = []
        for i in range(0, self.length - sequence_length, 1):
        if i % stride == 0:
            if windows is None:
            windows = np.array([self.input[i:i+sequence_length, ...]])
            else:
            windows = np.concatenate((windows,[self.input[i:i+sequence_length, ...]]), axis=0)
            targets.append(max(self.target[i:i+sequence_length]))

        freqs = Counter(targets)    
        targets = np.array(targets)
        print("Testing Target Distribution: ", sorted(freqs.items(), key=lambda x:(x[0])))
        return windows, targets, freqs

    # Look at visual plot, there are breaks in the experiment, currently considering as one segment
    def deal_with_discontinuities(self):
        pass