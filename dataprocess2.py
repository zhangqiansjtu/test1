import numpy as np
import matplotlib.pyplot as plt
import math
import os
import re
from scipy.signal import butter, lfilter, find_peaks_cwt
from statsmodels.tsa.seasonal import seasonal_decompose

fs = 48000
def medfilter (data):
    pdata=data.copy()
    for j in range(3,len(data)-2):
        pdata[j] = 0.15 * data[j - 2] + 0.2 * data[j - 1] + 0.3 * data[j] + 0.2 * data[j + 1] + 0.15 * data[j + 2]
    return pdata
def datapro():
    filepath = 'D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/data3/'
    files = os.listdir(filepath)
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, file)
        if len(res) == 3 and int(res[1]) >= 0 and int(res[1]) < 960:
            filename = filepath+file
            rawdata = np.memmap(filename, dtype=np.float32, mode='r')
            dataI = []
            dataQ = []
            dataC1 = []
            dataC2 = []
            for channelID in range(0, 8):
                fc = 17350 + 700 * channelID
                data = butter_bandpass_filter(rawdata, fc - 100, fc + 100, 48000)
                f = fc
                I = getI(data, f)
                I = move_average_overlap(I)
                Q = getQ(data, f)
                Q = move_average_overlap(Q)
                decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
                trendQ = decompositionQ.trend
                decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
                trendI = decompositionI.trend
                trendI = trendI[480:]
                trendQ = trendQ[480:]
                difftrendI = []
                for i in range(5, len(trendI)):
                    difftrendI.append((trendI[i] - trendI[i - 5]) * 1000)
                difftrendQ = []
                for i in range(5, len(trendQ)):
                    difftrendQ.append((trendQ[i] - trendQ[i - 5]) * 1000)
                difftrendI = medfilter(difftrendI)
                difftrendI = medfilter(difftrendI)
                difftrendQ = medfilter(difftrendQ)
                difftrendQ = medfilter(difftrendQ)
                difftrendI = np.around(difftrendI, decimals=6)
                difftrendQ = np.around(difftrendQ, decimals=6)
                # datachord1, datachord2 = chord_extract(trendI, trendQ)
                # plt.figure()
                # plt.plot(datachord2)
                # plt.show()
                if len(dataI):
                    dataI = np.vstack((dataI, difftrendI))
                    dataQ = np.vstack((dataQ, difftrendQ))
                    # dataC1 = np.vstack((dataC1, datachord1))
                    # dataC2 = np.vstack((dataC2, datachord2))
                else:
                    dataI = difftrendI
                    dataQ = difftrendQ
                    # dataC1 = datachord1
                    # dataC2 = datachord2
            dataIQ = np.vstack((dataI, dataQ))
            dataIQ = np.transpose(dataIQ)
            # dataC = np.vstack((dataC1, dataC2))
            # dataC = np.transpose(dataC)
            np.savez_compressed('./data/lipcontrol/cutdata12/datapre%d-%d-%d' % (int(res[0]), int(res[1]),int(res[2])),
                            datapre=dataIQ)


def chord_extract(trendI, trendQ):
    datachord1 = []
    for i in range(0, 10):
        datachord1.append(0)
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
    win_size = 200
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    data = data.reshape((new_len, win_size))
    result = np.zeros(new_len)
    for i in range(new_len):
        result[i] = np.mean(data[i, :])
    return result

def move_average_overlap(data):
    win_size = 200
    new_len = len(data) // win_size
    data = data[0:new_len * win_size]
    new_len = new_len*2
    result = np.zeros(new_len)
    for index in range(0, new_len):
        start =  (index/2)*win_size
        end = (index/2+1)*win_size
        result[index] = np.mean(data[int(start):int(end)])
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