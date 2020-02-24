import numpy as np
import Utils
import os
import re

class Data_Control:
    def __init__(self, filepath):
        self.files = os.listdir(filepath)
        self.X, self.Xlen, self.Y, self.flen, self.dataindex, self.userID, self.len_rate = self.loadfile(filepath)
        self.alldata = np.array(self.X)
        self.alllabel = np.array(self.Y)
        self.allindex = np.array(self.dataindex)
        self.alluser = np.array(self.userID)
        self.padding_data, _, self.length = self.cnn_padding1(self.alldata, self.Xlen,self.flen )
        self.alldata = np.array(self.padding_data)
        trainindex, testindex = self.indexsplit(self.alldata.shape[0], False)
        print(len(testindex))
        # self.npfiles = np.array(self.files)
        self.traindata = self.alldata[trainindex]
        self.trainlabel = self.alllabel[trainindex]
        self.trainuser = self.alluser[trainindex]
        self.testdata = self.alldata[testindex]
        self.testlabel = self.alllabel[testindex]
        self.testuser = self.alluser[testindex]
        # self.testfile = self.npfiles[testindex]
        self.batch_id = 0


    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.8)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex = []
            testindex = []
            for i in range(indexlength):
                if self.allindex[i] < 2000 and self.len_rate[i] >= 0:

                    if self.alluser[i] == 7 and self.len_rate[i] == 0:
                        testindex.append(i)
                    elif self.alluser[i] != 7:
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
        len_rate = []
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if (len(res) == 3 and int(res[1]) >= 0) or (len(res) == 4 and int(res[1]) >= 0 and int(res[3]) == 0):
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
                if len(res) == 3:
                    len_rate.append(0)
                else:
                    len_rate.append(int(res[3]))
        return raw_data, raw_data_len, raw_label,featurelen, raw_index, raw_user, len_rate

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

    def cnn_padding1(self, data, slen,flen):
        raw_data = data
        lengths = slen
        median_length = int(np.median(lengths))
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, sample in enumerate(raw_data):
            temp = np.zeros([flen, median_length])
            sample = np.transpose(sample)
            if slen[idx] < median_length:
                len_diff = median_length - slen[idx]
                len_diff1 = len_diff//2
                len_diff2 = len_diff-len_diff1
                for xidx, x in enumerate(sample):
                    aa = [x[0]]*len_diff1
                    bb = [x[-1]]*len_diff2
                    cc = x.tolist()
                    temp[xidx,:] = [x[0]]*len_diff1+x.tolist()+[x[-1]]*len_diff2
            if slen[idx] > median_length:
                len_diff = slen[idx] - median_length
                # len_diff1 = len_diff//2
                # len_diff2 = len_diff-len_diff1
                temp = sample[:, len_diff:slen[idx]]
            if slen[idx] == median_length:
                temp = sample
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
