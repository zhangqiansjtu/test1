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
        self.X, self.sentindex, self.index, self.Ys, self.Ylen = self.loadfile(filepath)
        self.alldata = np.array(self.X)
        self.imagedata,self.seq_len,self.rowimg,self.img = self.dataimage(self.alldata)
        self.imagepadding, self.imagelabel, self.s_len, self.label_len = self.padding_ctc(self.imagedata,self.seq_len,self.rowimg,self.img,self.Ys,self.Ylen)
        self.alldata=np.array(self.imagepadding)
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
        self.testfile=self.npfiles[testindex]
        print(self.testfile)
        self.batch_id = 0


    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.8)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            # trainindex=[]
            # testindex =[]
            # randomsentind = list(range(20))
            # np.random.shuffle(randomsentind)
            # # testsentindex = randomsentind[:int(len(randomsentind) * 0.1)]
            # testsentindex = [1]
            # trainsentindex = list(filter(lambda j: j not in testsentindex, list(range(20))))
            # for i in range(indexlength):
            #     if self.sentindex[i] in testsentindex:
            #         testindex.append(i)
            #     else:
            #         trainindex.append(i)
            # np.random.shuffle(trainindex)
            trainindex = []
            testindex =[]
            for i in range(indexlength):
                if self.index[i] >= 0:

                    if self.index[i] >= 20 and self.index[i] < 30:
                        testindex.append(i)
                    else:
                        trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
        return trainindex, testindex

    def loadfile(self,filepath):
        raw_data = []
        rawlabel = []
        rawlabel_len = []
        rawsenin = []
        rawindex = []
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if len(res) > 1:
                filename = filepath + file
                data = np.load(filename)
                sample = data['datapre']
                raw_data.append(sample[:, :-1])
                rawsenin.append(int(res[0]))
                samplela = sample[:, -1]-1
                rm_rep_y = [v for v in samplela if v != -1]
                rm_rep_y.extend([self.n_classes+2])
                rawindex.append(int(res[1]))
                rawlabel.append(rm_rep_y)
                rawlabel_len.append(len(rm_rep_y))
        return raw_data, rawsenin, rawindex, rawlabel, rawlabel_len

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
        padding_data = np.zeros([num_samples, max_length, rowimg,colimg])
        padding_data[:, :, :, :] = padding_data[:, :, :, :] - 1
        for idx, seq in enumerate(raw_data):
            padding_data[idx, :len(seq), :, :] = seq
        # label全部补全
        padding_label = np.zeros([num_samples, max(ylen)]) + self.n_classes
        for idx, seq in enumerate(ys):
            padding_label[idx, :len(seq)] = seq
        return padding_data, padding_label, np.array(slen), np.array(ylen)
