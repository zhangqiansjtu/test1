import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks_cwt, spectrogram, convolve2d
from statsmodels.tsa.seasonal import seasonal_decompose
# from scipy.signal import find_peaks
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy.signal import stft
import os
import re


fs = 48000


def main():
    getData()
    return

def medfilter (data):
    pdata=data.copy()
    for j in range(3, len(data)-2):
        pdata[j] = 0.15 * data[j - 2] + 0.2 * data[j - 1] + 0.3 * data[j] + 0.2 * data[j + 1] + 0.15 * data[j + 2]
    return pdata

def getData():
    filepath = 'D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/data3/'
    files=os.listdir(filepath)
    for file in files:
        pattern = re.compile(r'\d+')
        res = re.findall(pattern,file)
        if len(res) == 3 and int(res[0]) == 11 and int(res[1]) >= 30 and int(res[1]) < 40:
            filename = filepath+file
            rawdata = np.memmap(filename, dtype=np.float32, mode='r')
            # freq, t, zxx = stft(rawdata, fs=fs, nperseg=48000, noverlap=47000)
            # plt.figure()
            # plt.plot(rawdata)
            # plt.pcolormesh(t, freq, np.abs(zxx))
            # rawdata=rawdata[70001:]
            # plt.figure()
            dataphase = []
            plt.figure()
            for channelID in range(0, 8):
                fc = 17350+700*channelID
                data = butter_bandpass_filter(rawdata, fc-150, fc+150, 48000)
                data1 = data[72000:]
                freq, t, zxx = spectrogram(data1, fs=fs, nperseg=2048, noverlap=1024, nfft=48000)
                # t=t[30:-10]
                freq = freq[fc-50:fc+51]
                # zxx[fc-1:fc+2, :] = 0
                zxx = zxx[fc-50:fc+51, :]
                # zxx = np.abs(zxx)
                zxx = np.abs(np.diff(zxx))*100000000
                conv_factor = np.ones((3, 3))
                zxx2 = convolve2d(zxx, conv_factor, mode="same")
                # zxx = 10*np.log10(zxx)

                # plt.figure()
                # plt.pcolormesh(t, freq, zxx)
                # # plt.ylim((fc-50, fc+50))
                # plt.colorbar()
                # plt.figure()
                # plt.pcolormesh(t, freq, zxx2)
                # plt.ylim((fc-50, fc+50))
                # plt.colorbar()
                # aa = zxx[:, 100]
                # plt.figure()
                # plt.plot(aa)
                # datafft = abs(fft(data))
                # xf = np.arange(len(data))  # 频率
                # xf = xf / (data.shape[0]/fs)
                # plt.figure()
                # plt.plot(xf, datafft)
                f = fc
                I1 = getI(data, f)
                I = move_average_overlap(I1)
                Q1 = getQ(data, f)
                Q = move_average_overlap(Q1)

                decompositionQ = seasonal_decompose(Q, freq=10, two_sided=False)
                trendQ = decompositionQ.trend
                trendQ = trendQ[480:]
                decompositionI = seasonal_decompose(I, freq=10, two_sided=False)
                trendI = decompositionI.trend
                trendI = trendI[480:]
                difftrendI = []
                for i in range(5, len(trendI)):
                    difftrendI.append((trendI[i] - trendI[i-5])*1000)
                difftrendQ = []
                for i in range(5, len(trendQ)):
                    difftrendQ.append((trendQ[i] - trendQ[i-5])*1000)
                plt.subplot(211)
                plt.plot(difftrendI, label=channelID)
                plt.ylabel('Phase')
                plt.legend()
                plt.subplot(212)
                plt.plot(difftrendQ, label=channelID)
                plt.ylabel('Amplitude')
                plt.legend()
                signaldata=[]
                for i in range (0, trendI.shape[0]):
                    signalsample=complex(trendQ[i],trendI[i])
                    signaldata.append(signalsample)
                signal_ph=np.angle(signaldata)
                signal_ph = np.unwrap(signal_ph)
                signal_am = np.abs(signaldata)
                signal_ph = medfilter(signal_ph)
                signal_am = medfilter(signal_am)
                signal_ph = medfilter(signal_ph)
                signal_am = medfilter(signal_am)
                signal_phmean=np.mean(signal_ph[:20])
                signal_ammean = np.mean(signal_am[:20])
                signal_ph = signal_ph - signal_phmean
                signal_am = signal_am - signal_ammean
                signal_am = signal_am*1000
                signal_ph = np.around(signal_ph, decimals=5)
                signal_am = np.around(signal_am, decimals=5)
                signal_diffph = []
                for i in range(5, len(signal_ph)):
                    signal_diffph.append((signal_ph[i] - signal_ph[i-5])*2)
                signal_diffam = []
                for i in range(5, len(signal_am)):
                    signal_diffam.append((signal_am[i] - signal_am[i-5])*2)
                # signal_diffph = np.diff(signal_ph)
                # plt.subplot(211)
                # plt.plot(signal_diffph, label=channelID)
                # plt.ylabel('Phase')
                # plt.legend()
                # plt.subplot(212)
                # plt.plot(signal_diffam, label=channelID)
                # plt.ylabel('Amplitude')
                # plt.legend()
                # plt.plot(signal_ph)
                # plt.axis('off')
                # plt.figure()
                # plt.subplot(211)
                # plt.plot(trendI)
                # plt.subplot(212)
                # plt.plot(trendQ)
                # datafft=abs(fft(data))
                # xf = np.arange(len(data))  # 频率
                # xf = xf / (data.shape[0]/fs)
                # plt.figure()
                # plt.plot(trendI, trendQ)
                # plt.figure()
                #
                datachord=chord_extract(trendI, trendQ)
                freq, t, zxx = stft(datachord, fs=160, nperseg=160, noverlap=155)
                chordtf = np.abs(zxx[:40, :])
                chordtf[0, :] = 0
                # plt.figure()
                # plt.subplot(313)
                # plt.plot(datachord)
                # plt.subplot(211)
                # plt.pcolormesh(t, freq[:20], chordtf)
                # plt.ylim((1, 19))
                # freq, t, zxx = stft(signal_am, fs=160, nperseg = 160, noverlap = 155)
                # chordtf = np.abs(zxx[:20, :])
                # chordtf[0,:] = 0
                # plt.subplot(212)
                # plt.pcolormesh(t, freq[:40], chordtf)
                # plt.ylim((1, 39))
                # plt.plot(datachord)
                if len(dataphase):
                    dataphase = np.vstack((dataphase, signal_ph))
                    dataam = np.vstack((dataam, signal_am))
                else:
                    dataphase = signal_ph
                    dataam = signal_am
            # dataft = []
            # for i in range(dataam.shape[1]):
            #
            #     f_sample = dataam[:, i]
            #     ft_sample = abs(ifft(f_sample))
            #     dataft.append(ft_sample)
            # dataft = np.transpose(np.array(dataft))
            # dataft = dataft[1:, :]
            # plt.figure()
            # plt.pcolormesh(dataft)
    plt.show()

def chord_extract(trendI, trendQ):
    datachord = []
    for i in range(10, trendI.shape[0]):
        sample = ((trendI[i]-trendI[i-10])**2+(trendQ[i]-trendQ[i-10])**2)**0.5
        datachord.append(sample)
    return datachord




def getPhase1(I, Q):
    derivativeQ = getDerivative(Q)
    derivativeI = getDerivative(I)
    # phase=np.unwrap(2*())+np.pi/2))/2
    # distance=distanceLine(phase,20000)
    # plt.plot(distance)
    # plt.show()
    derivativeQ = np.asarray(derivativeQ)
    derivativeQ[np.where(derivativeQ==0)]=0.000001
    arcValue = np.arctan(-np.asarray(derivativeI) / (derivativeQ))
    newData = unwrap(arcValue)
    plt.plot(newData)
    plt.show()


def unwrap(data):
    resultData = []
    diffs = np.roll(data, -1) - data
    diffs = diffs[:len(data) - 1]
    first_value = data[0]
    resultData.append(first_value)
    previous_value = first_value
    current_value=None
    for diff in diffs:
        if diff > np.pi / 2:
            current_value = previous_value + diff - np.pi
            resultData.append(current_value)
        elif diff < -np.pi / 2:
            current_value = previous_value + diff + np.pi
            resultData.append(current_value)
        else:
            current_value=previous_value+diff
            resultData.append(current_value)
        previous_value = current_value
    return np.asarray(resultData)

def getDerivative(data):
    derivativeQ = []
    for i in range(len(data) - 1):
        derivativeQ.append((data[i + 1] - data[i]))
    return derivativeQ


def removeDC(data):
    return data - np.mean(data)


def distanceLine(phase, freq):
    distances = np.zeros(len(phase) - 1)
    for i in np.arange(1, len(phase)):
        phaseDiff = phase[0] - phase[i]
        distanceDiff = 343 / (2 * np.pi * freq) * phaseDiff
        distances[i - 1] = distanceDiff
    distances = distances / 2
    return distances


def getPhase(Q, I):
    if I == 0 and Q > 0:
        return np.pi / 2
    elif I == 0 and Q < 0:
        return 3 / 2 * np.pi
    elif Q == 0 and I > 0:
        return 0
    elif Q == 0 and I < 0:
        return np.pi
    tanValue = Q / I
    tanPhase = np.arctan(tanValue)
    resultPhase = 0
    if I > 0 and Q > 0:
        resultPhase = tanPhase
    elif I < 0 and Q > 0:
        resultPhase = np.pi + tanPhase
    elif I < 0 and Q < 0:
        resultPhase = np.pi + tanPhase
    elif I > 0 and Q < 0:
        resultPhase = 2 * np.pi + tanPhase
    return resultPhase


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
    main()
