import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.signal import butter, lfilter, find_peaks_cwt
from statsmodels.tsa.seasonal import seasonal_decompose

fs = 48000
cutID = 7
userID=0
def main():
    getData()
    return

def getData():

    filepath = 'D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/'
    files = os.listdir(filepath)
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern, file)
        if len(res) == 2 and int(res[0]) == cutID and int(res[1]) == 0:
            file=filepath+file
            rawdata = np.memmap(file, dtype=np.float32, mode='r')
            cutpath='./data/lipcontrol/cutdata/'
            cutfiles = os.listdir(cutpath)
            cutindex = -1
            for cutfile in cutfiles:
                cutpattern = re.compile(r'\d+')
                cutres = re.findall(cutpattern, cutfile)
                if len(cutres) == 3 and int(cutres[0]) == cutID:
                    if int(cutres[1]) > cutindex:
                        cutindex = int(cutres[1])
            cutindex=cutindex+1
            cut_signal(rawdata, cutID, cutindex, userID)
    return rawdata

def cut_signal(rawdata, cutID, cutindex, userID):
    fc = 17350 + 700 * 0
    # rawdata = rawdata[600000:800000]
    data = butter_bandpass_filter(rawdata, fc - 300, fc + 300, 48000)
    f = fc
    I = getI(data, f)
    I = move_average(I)
    Q = getQ(data, f)
    Q = move_average(Q)
    # plt.figure()
    # plt.plot(I, color="red")
    # rawdata = rawdata[1250000:1275000]
    #
    # data1 = butter_bandpass_filter(rawdata, fc - 300, fc + 300, 48000)
    #
    # I1 = getI(data1, f)
    # plt.figure()
    # plt.plot(rawdata, color="red")
    # plt.xlim((0, 1000))
    # plt.figure()
    # plt.plot(I1, color="red")
    # I1 = butter_lowpass_filter(I1, 100, fs)
    # I1 = move_average(I1)
    # Q1 = getQ(rawdata, f)
    # Q1 = butter_lowpass_filter(Q1, 100, fs)
    # plt.figure()
    # plt.plot(I1,Q1, color="red")
    datachord = chord_extract(I, Q)
    plt.figure()
    plt.plot(datachord, color="red")
    # signaldata = []
    # for i in range(0, I1.shape[0]):
    #     signalsample = complex(Q1[i], I1[i])
    #     signaldata.append(signalsample)
    # signal_ph = np.angle(signaldata)
    # signal_ph = np.unwrap(signal_ph)
    # signal_am = np.abs(signaldata)
    # signal_phmean = np.mean(signal_ph[:20])
    # signal_ammean = np.mean(signal_am[:20])
    # signal_ph = signal_ph - signal_phmean
    # plt.show()
    # plt.figure()
    # plt.plot(signal_ph, color="red")
    # pos = plt.ginput(50, timeout=100000)
    # print(pos)
    # for i in range(len(pos) // 2):
    #     j = i * 2
    #     time = [k for k in range(len(data))]
    #     time_start=pos[j][0]
    #     time_end = pos[j+1][0]
    #     sub_index = np.where((time_start < time) & (time < time_end))[0]
    #     sub_indexstart=sub_index[0] * 300
    #     sub_indexend = sub_index[-1] * 300
    #     subdata=rawdata[sub_indexstart:sub_indexend]
    #     subdata = rawdata[590000:1200000]
    #     plt.figure()
    #     subdatadata = butter_bandpass_filter(subdata, fc - 300, fc + 300, 48000)
    #     subI = getI(subdatadata, f)
    #     subI = move_average(subI)
    #     plt.plot(subI,color='red')
    #     # np.savez_compressed('./data/lipcontrol/cutdata/data%d-%d-%d'%(cutID,cutindex,userID),watchdata=subdata)
    #     cutindex=cutindex+1
    plt.show()


def chord_extract(trendI, trendQ):
    datachord = []
    for i in range(10, trendI.shape[0]):
        sample = ((trendI[i]-trendI[i-10])**2+(trendQ[i]-trendQ[i-10])**2)**0.5
        datachord.append(sample)
    return datachord


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
