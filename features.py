#This module describes the features that can be extracted from the raw data
import numpy as np
from math import sqrt
from scipy import stats
from numpy.fft import fftfreq

#This section describes some of the basic statistics
#that is going to be extracted from the raw data
#For both time and frequency domain
def get_min(data):
    return np.min(data)

def get_max(data):
    return np.max(data)

def get_mean(data):
    return np.mean(data)

def get_std(data):
    return np.std(data)

#For frequency domain only
def get_peak_frequency(data):
    return np.argmax(data)

def get_skewness(data):
    return stats.skew(data)

#This section defines some utility function used for feature extraction
def fast_fourier(data):
    return np.fft.fft(data)

def get_magnitude(acc_x, acc_y, acc_z):
    magnitude = []
    for i in range(len(acc_x)):
        magnitude.append(sqrt(pow(acc_x[i], 2) + pow(acc_y[i], 2) + pow(acc_z[i], 2)))
    return np.asarray(magnitude)

#This section describes the high level features that is going to be used for machine learning
def get_body_acc_t(acc):
    f_temp = []
    f_temp.append(get_min(acc))
    f_temp.append(get_max(acc))
    f_temp.append(get_mean(acc))
    f_temp.append(get_std(acc))
    return f_temp

def get_body_acc_f(acc):
    f = fftfreq(len(acc))
    mask = f >= 0
    acc = 2 * acc[mask]
    f_temp = []
    f_temp.append(get_min(acc))
    f_temp.append(get_max(acc))
    f_temp.append(get_mean(acc))
    f_temp.append(get_std(acc))
    f_temp.append(get_peak_frequency(acc))
    f_temp.append(get_skewness(acc))
    return f_temp

#This is the main feature extraction algorithm
def extract(window_slice):
    features = []
    window_slice = np.asarray(window_slice)
    window_slice_transpose = window_slice.transpose()
    window_slice_transpose_f = np.abs(fast_fourier(window_slice_transpose)/len(window_slice))  

    t_acc_x = window_slice_transpose[0]
    t_acc_y = window_slice_transpose[1]
    t_acc_z = window_slice_transpose[2]
    t_acc_mag = get_magnitude(t_acc_x, t_acc_y, t_acc_z) 

    f_acc_x = window_slice_transpose_f[0]
    f_acc_y = window_slice_transpose_f[1]
    f_acc_z = window_slice_transpose_f[2]
    f_acc_mag = get_magnitude(f_acc_x, f_acc_y, f_acc_z)

    #extract features
    features.extend(get_body_acc_t(t_acc_x))
    features.extend(get_body_acc_t(t_acc_y))
    features.extend(get_body_acc_t(t_acc_z))
    features.extend(get_body_acc_t(t_acc_mag))

    features.extend(get_body_acc_f(f_acc_x))
    features.extend(get_body_acc_f(f_acc_y))
    features.extend(get_body_acc_f(f_acc_z))
    features.extend(get_body_acc_f(f_acc_mag))

    return features