import numpy as np
import csv
import Utils
import os
import re


class Data_Control:
    def __init__(self, filepath):
        self.files = os.listdir(filepath)
        self.X, self.Xlen, self.Y,self.flen,self.dataindex,self.userID, self.len_rate = self.loadfile(filepath)
        self.alldata = np.array(self.X)
        self.alllabel = np.array(self.Y)
        self.allindex = np.array(self.dataindex)
        self.alluser = np.array(self.userID)
        self.imagedata, self.seq_len, self.rowimg, self.colimg = self.dataimage(self.alldata)
        self.imagepadding, self.s_len = self.padding(self.imagedata, self.seq_len, self.rowimg, self.colimg)
        self.alldata = np.array(self.imagepadding)
        trainindex, testindex = self.indexsplit(self.alldata.shape[0], True)
        print(len(testindex))
        self.npfiles = np.array(self.files)
        self.traindata = self.alldata[trainindex]
        self.trainlabel = self.alllabel[trainindex]
        self.trainlen = self.s_len[trainindex]
        self.testdata = self.alldata[testindex]
        self.testlabel = self.alllabel[testindex]
        self.testlen = self.s_len[testindex]
        self.testfile = self.npfiles[testindex]
        self.batch_id = 0


    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.7)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex=[]
            testindex =[]
            for i in range(indexlength):
                if self.allindex[i] >= 0:

                    if self.alluser[i] == 1 and self.len_rate[i] == 0:
                        testindex.append(i)
                    elif self.alluser[i] != 1:
                            trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
        return trainindex, testindex

    def loadfile(self, filepath):
        raw_data = []
        raw_data_len = []
        raw_label = []
        raw_index=[]
        raw_user=[]
        len_rate = []
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if (len(res) == 3 and int(res[1]) >= 900) or (len(res) == 4 and int(res[1]) >= 900 and int(res[3]) == 0):
                filename = filepath+file
                data = np.load(filename )
                sample = data['datapre']
                sample=sample[:,:]
                raw_data.append(sample)
                featurelen = sample.shape[1]
                raw_data_len.append(sample.shape[0])
                raw_label.append(int(res[0]))
                raw_index.append(int(res[1]))
                raw_user.append(int(res[2]))
                if len(res) == 3:
                    len_rate.append(0)
                else:
                    len_rate.append(int(res[3]))
        return raw_data, raw_data_len, raw_label, featurelen, raw_index, raw_user, len_rate

    def dataimage(self, xdata):
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
                img.append(tmp[j-50:j+50, :])
                j = j+50
            imgnp = np.array(img)
            rowimg = imgnp.shape[1]
            colimg = imgnp.shape[2]
            imagedata.append(img)
            seq_len.append(len(img))
        return imagedata, seq_len, rowimg, colimg

    def padding(self, data, slen, rowimg, colimg):
        """construct input for lstm-ctc, seq2seq"""
        raw_data = data
        lengths = slen
        max_length = max(lengths)
        #防止出现某个偏大值
        median_length = np.median(lengths)
        print("max_length= %d, median_length= %d" % (max_length, median_length))
        num_samples = len(lengths)
        set_length = 60
        padding_data = np.zeros([num_samples, 60, rowimg, colimg])
        padding_data[:, :, :,:] = padding_data[:, :, :] - 1
        for idx, seq in enumerate(raw_data):
            if len(seq) > set_length:
                seq = seq[:set_length]
            padding_data[idx, :len(seq), :] = seq
        return padding_data, np.array(slen)