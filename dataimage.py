import numpy as np
import matplotlib.pyplot as plt
import math
import os
import re
from scipy.signal import butter, lfilter, find_peaks_cwt, spectrogram, convolve2d
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import stft
fs = 48000

def datapro():
    filepath = 'D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/data1/'
    files = os.listdir(filepath)
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, file)
        if len(res) == 3 and int(res[1]) >= 720:
            filename = filepath+file
            rawdata = np.memmap(filename, dtype=np.float32, mode='r')
            datatf = []
            for channelID in range(0, 8):
                fc = 17350 + 700 * channelID
                data = butter_bandpass_filter(rawdata, fc - 100, fc + 100, 48000)
                data1 = data[48000:]
                freq, t, zxx = spectrogram(data1, fs=fs, nperseg=2048, noverlap=1024, nfft=48000)
                zxx = np.abs(np.diff(zxx)) * 10000000
                conv_factor = np.ones((3, 3))
                zxx2 = convolve2d(zxx, conv_factor, mode="same")
                zxx2 = zxx2[fc-40:fc+41, :]
                if len(datatf):
                    datatf = np.vstack((datatf, zxx2))
                else:
                    datatf = zxx2
            datatf = np.transpose(datatf)
            np.savez_compressed('./data/lipcontrol/cutdata4/datapre%d-%d-%d' % (int(res[0]), int(res[1]), int(res[2])),
                            datapre=datatf)


def chord_extract(trendI, trendQ):
    datachord1 = []
    for i in range(10, trendI.shape[0]):
        sample = ((trendI[i]-trendI[i-10])**2+(trendQ[i]-trendQ[i-10])**2)**0.5
        datachord1.append(sample)
    datachord2 = []
    for i in range(0, 10):
        datachord2.append(0)
    for i in range(10, len(datachord1)):
        sample = datachord1[i] - datachord1[i-10]
        datachord2.append(sample)
    return datachord1, datachord2

def move_average(data):
    win_size = 300
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    data = data.reshape((new_len, win_size))
    result = np.zeros(new_len)
    for i in range(new_len):
        result[i] = np.mean(data[i, :])
    return result


def getI(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulCos = np.cos(2 * np.pi * f * times) * data
    return mulCos


def getQ(data, f):
    times = np.arange(0, len(data)) * 1 / fs
    mulSin = -np.sin(2 * np.pi * f * times) * data
    return mulSin


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    datapro()