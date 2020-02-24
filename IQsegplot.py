import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import butter, lfilter, find_peaks_cwt
from statsmodels.tsa.seasonal import seasonal_decompose

fs = 48000
cutID = 2
userID = 0
def main():
    getData()
    return

def getData():

    filepath = 'D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/'
    files = os.listdir(filepath)
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, file)
        if len(res) == 2 and int(res[0]) == cutID and int(res[1]) == 2:
            file=filepath+file
            rawdata = np.memmap(file, dtype=np.float32, mode='r')
            cutpath='./data/lipcontrol/cutdata2/'
            cutfiles = os.listdir(cutpath)
            cutindex = -1
            for cutfile in cutfiles:
                cutpattern = re.compile(r'\d+')
                cutres = re.findall(cutpattern, cutfile)
                if len(cutres) == 3 and int(cutres[0]) == cutID:
                    if int(cutres[1]) > cutindex:
                        cutindex = int(cutres[1])
            cutindex=cutindex+1
            dataI = []
            dataQ = []
            for channelID in range(0, 8):
                fc = 17350 + 700 * 0
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
            dataIQ = np.vstack((dataI, dataQ))
            dataIQ = np.transpose(dataIQ)
            cut_signal(dataIQ, cutID, cutindex, userID)
    return rawdata

def cut_signal(rawdata, cutID, cutindex, userID):
    I = rawdata[:,0]
    plt.figure()
    plt.plot(I, color="red")
    pos = plt.ginput(50, timeout=100000)
    print(pos)
    for i in range(len(pos) // 2):
        j = i * 2
        time = [k for k in range(len(I))]
        time_start=pos[j][0]
        time_end = pos[j+1][0]
        sub_index = np.where((time_start < time) & (time < time_end))[0]
        subdata=rawdata[sub_index]
        subdata=np.array(subdata)
        plt.figure()
        subI = subdata[:,0]
        plt.plot(subI,color='red')
        # np.savez_compressed('./data/lipcontrol/cutdata2/data%d-%d-%d'%(cutID,cutindex,userID),watchdata=subdata)
        cutindex=cutindex+1
    plt.show()



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
    main()
