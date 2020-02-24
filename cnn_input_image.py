import numpy as np
import csv
import Utils
import os
import re

class Data_Control:
    def __init__(self, filepath):
        self.files = os.listdir(filepath)
        self.X, self.Xlen, self.Y,self.flen,self.dataindex,self.userID = self.loadfile(filepath)
        self.alldata = np.array(self.X)
        self.alllabel = np.array(self.Y)
        self.allindex = np.array(self.dataindex)
        self.alluser = np.array(self.userID)
        self.padding_data, _, self.length = self.cnn_padding(self.alldata, self.Xlen,self.flen )
        self.alldata = np.array(self.padding_data)
        trainindex, testindex = self.indexsplit(self.alldata.shape[0], False)
        print(len(testindex))
        self.npfiles = np.array(self.files)
        self.traindata = self.alldata[trainindex]
        self.trainlabel = self.alllabel[trainindex]
        self.trainuser = self.alluser[trainindex]
        self.testdata = self.alldata[testindex]
        self.testlabel = self.alllabel[testindex]
        self.testuser = self.alluser[testindex]
        self.testfile = self.npfiles[testindex]
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
                if self.allindex[i] >= 0 :

                    if self.allindex[i] >= 740 and self.allindex[i] < 750:
                        testindex.append(i)
                    else:
                        trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
        return trainindex, testindex

    def loadfile(self,filepath):
        raw_data = []
        raw_data_len = []
        raw_label = []
        raw_index = []
        raw_user = []
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if len(res) > 1 and int(res[1]) >= 600:
                filename = filepath+file
                data = np.load(filename)
                sample = data['datapre']
                sample = sample[:,:]
                featurelen = sample.shape[1]
                raw_data.append(sample)
                raw_data_len.append(sample.shape[0])
                raw_label.append(int(res[0]))
                raw_index.append(int(res[1]))
                raw_user.append(int(res[2]))
        return raw_data, raw_data_len, raw_label,featurelen,raw_index,raw_user

    def cnn_padding(self, data, slen,flen):
        raw_data = data
        lengths = slen
        median_length = int(np.median(lengths))
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, sample in enumerate(raw_data):
            temp = [Utils.resampling(np.arange(0, len(x), 1), x, [0, len(x)], median_length)[1] for x in np.array(sample).transpose()]
            padding_data[idx, :, :] = np.array(temp).transpose()
        return padding_data, np.array(slen), median_length
