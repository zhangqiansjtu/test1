import numpy as np
import matplotlib.pyplot as plt
import math
import os
import re
from scipy.signal import butter, lfilter, find_peaks_cwt
from statsmodels.tsa.seasonal import seasonal_decompose

fs = 48000


def datapro():
    filepath = './data/lipcontrol/cutdata/'
    files = os.listdir(filepath)
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, file)
        if len(res)==3 :
            filename = filepath+file
            data = np.load(filename)
            rawdata = data['watchdata']
            dataI=[]
            dataQ=[]
            for channelID in range (0,8):
                fc = 17350 + 700 * channelID
                data = butter_bandpass_filter(rawdata, fc - 300, fc + 300, 48000)
                f = fc
                I = getI(data, f)
                I = move_average(I)
                Q = getQ(data, f)
                Q = move_average(Q)
                decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
                trendQ = decompositionQ.trend
                decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
                trendI = decompositionI.trend
                trendI = trendI[10:]
                trendQ = trendQ[10:]
                if len(dataI):
                    dataI = np.vstack((dataI, trendI))
                    dataQ = np.vstack((dataQ, trendQ))
                else:
                    dataI = trendI
                    dataQ = trendQ
            dataIQ=np.vstack((dataI,dataQ))
            dataIQ=np.transpose(dataIQ)
            np.savez_compressed('./data/lipcontrol/cutdata/datapre/datapre%d-%d-%d' % (int(res[0]), int(res[1]),int(res[2])),
                            datapre=dataIQ)


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