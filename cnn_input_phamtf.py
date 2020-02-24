import numpy as np
import Utils
import os
import re

class Data_Control:
    def __init__(self, filepath1, filepath2):
        files1 = os.listdir(filepath1)
        files2 = os.listdir(filepath2)
        self.X1, self.Xlen1, self.alllabel, self.flen1, self.allindex, self.alluser = self.loadfile(filepath1, files1)
        self.padding_data1, _, self.length1 = self.cnn_padding(self.X1, self.Xlen1, self.flen1)
        self.alldata1 = np.array(self.padding_data1)
        self.X2, self.Xlen2, _, self.flen2, _, _ = self.loadfile(filepath2, files2)
        self.padding_data2, _, self.length2 = self.cnn_padding(self.X2, self.Xlen2, self.flen2)
        self.alldata2 = np.array(self.padding_data2)
        trainindex, testindex = self.indexsplit(self.alldata1.shape[0], False)
        print(len(testindex))
        self.traindata_pham = self.alldata1[trainindex]
        self.testdata_pham = self.alldata1[testindex]
        self.traindata_tf = self.alldata2[trainindex]
        self.testdata_tf = self.alldata2[testindex]
        self.trainlabel = self.alllabel[trainindex]
        self.trainuser = self.alluser[trainindex]
        self.testlabel = self.alllabel[testindex]
        self.testuser = self.alluser[testindex]
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
                if self.allindex[i] < 140:

                    if self.allindex[i] >= 130  and self.allindex[i] < 140:
                        testindex.append(i)
                    else:
                        trainindex.append(i)
            print(trainindex)
            print(len(trainindex))
            np.random.shuffle(trainindex)
        return trainindex, testindex

    def loadfile(self, filepath, files):
        raw_data = []
        raw_data_len = []
        raw_label = []
        raw_index = []
        raw_user = []
        for file in files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if len(res) > 1 and int(res[1]) < 140:
                filename = filepath+file
                data = np.load(filename)
                sample = data['datapre']
                sample = sample[:, :]
                featurelen = sample.shape[1]
                raw_data.append(sample)
                raw_data_len.append(sample.shape[0])
                raw_label.append(int(res[0]))
                raw_index.append(int(res[1]))
                raw_user.append(int(res[2]))
        return np.array(raw_data), raw_data_len, np.array(raw_label), featurelen, np.array(raw_index), np.array(raw_user)

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

    def next_train(self, batch_size):
        if self.batch_id >= len(self.traindata):
            self.batch_id = 0
        batch_X = (self.traindata[self.batch_id:min(self.batch_id + batch_size, len(self.traindata))])
        batch_Y = (self.trainlabel[self.batch_id:min(self.batch_id +
                                                            batch_size, len(self.traindata))])
        self.batch_id = min(self.batch_id + batch_size, len(self.traindata))
        return batch_X, batch_Y
