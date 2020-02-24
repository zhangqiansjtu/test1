import numpy as np
import csv
import tensorflow as tf
import platform
import os
from scipy import interpolate
import random
import re

class Data_Control:
    def __init__(self, filepath, n_classes):
        self.n_classes = n_classes
        self.files = os.listdir(filepath)
        self.X, self.allindex, self.Ys, self.Ylen, self.userID, self.len_rate = self.loadfile(filepath)
        self.alldata = np.array(self.X)
        self.alluser = np.array(self.userID)
        self.allindex = np.array(self.allindex)
        self.imagedata,self.seq_len,self.rowimg,self.img = self.dataimage(self.alldata)
        self.imagepadding, self.imagelabel, self.s_len, self.label_len = self.padding_ctc(self.imagedata,self.seq_len,self.rowimg,self.img,self.Ys,self.Ylen)
        self.alldata = np.array(self.imagepadding)
        self.alllabel = np.array(self.imagelabel)
        trainindex,testindex=self.indexsplit(len(self.X), False)
        print(len(testindex))
        print(testindex)
        self.npfiles = np.array(self.files)
        self.traindata=self.alldata[trainindex]
        self.trainlabel= self.alllabel[trainindex]
        self.trainlabel_len=self.label_len[trainindex]
        self.trainlen = self.s_len[trainindex]
        self.testdata=self.alldata[testindex]
        self.testlabel= self.alllabel[testindex]
        self.testlabel_len = self.label_len[testindex]
        self.testlen = self.s_len[testindex]
        self.batch_id = 0


    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.8)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex = []
            testindex =[]
            for i in range(indexlength):
                if self.allindex[i] < 2000 and self.len_rate[i] >= 0:

                    if self.alluser[i] == 1 and self.len_rate[i] == 0:
                        testindex.append(i)
                    elif self.alluser[i] != 1:
                            trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
        return trainindex, testindex

    def loadfile(self,filepath):
        raw_data = []
        rawlabel = []
        rawlabel_len = []
        rawindex = []
        len_rate = []
        raw_user = []
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if (len(res) == 3 and int(res[1]) >= 0) or (len(res) == 4 and int(res[1]) >= 0 and int(res[3]) == 0):
                filename = filepath + file
                data = np.load(filename)
                sample = data['datapre']
                raw_data.append(sample[:, :])
                samplela = [int(res[0])]
                samplela.extend([self.n_classes+2])
                rawindex.append(int(res[1]))
                rawlabel.append(samplela)
                rawlabel_len.append(len(samplela))
                raw_user.append(int(res[2]))
                if len(res) == 3:
                    len_rate.append(0)
                else:
                    len_rate.append(int(res[3]))
        return raw_data, rawindex, rawlabel, rawlabel_len, raw_user, len_rate

    def dataimage(self,xdata):
        imagedata = []
        len1 = xdata.shape[0]
        seq_len = []
        for i in range(0, len1):
            tmp = xdata[i]
            tmp = np.array(tmp)
            len2 = len(tmp)
            j = 50
            img = []
            while j < len2-51:
                img.append(tmp[j-50:j+51, :])
                j = j+50
            imgnp = np.array(img)
            rowimg = imgnp.shape[1]
            colimg = imgnp.shape[2]
            imagedata.append(img)
            seq_len.append(len(img))

        return imagedata, seq_len, rowimg, colimg

    def padding_ctc(self,data,slen,rowimg,colimg,ys,ylen):
        """construct input for lstm-ctc, seq2seq"""
        raw_data = data
        lengths = slen
        max_length = max(lengths)
        num_samples = len(lengths)
        #防止出现某个偏大值
        median_length = np.median(lengths)
        print("max_length= %d, median_length= %d" % (max_length, median_length))
        num_samples = len(lengths)
        set_length = 60
        padding_data = np.zeros([num_samples, set_length, rowimg,colimg])
        padding_data[:, :, :, :] = padding_data[:, :, :, :] - 1
        for idx, seq in enumerate(raw_data):
            if len(seq) > set_length:
                seq = seq[:set_length]
            padding_data[idx, :len(seq), :] = seq
        # label全部补全
        padding_label = np.zeros([num_samples, max(ylen)]) + self.n_classes
        for idx, seq in enumerate(ys):
            padding_label[idx, :len(seq)] = seq
        return padding_data, padding_label, np.array(slen), np.array(ylen)
