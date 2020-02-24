import re
import os
import numpy as np
import matplotlib.pyplot as plt
import Utils
import random

filepath = './data/lipcontrol/cutdata12/'
files = os.listdir(filepath)
rate_ind = 4
for file in files:
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, file)
    v_rand = random.randint(0,100)/1000-0.05
    length_rate =1.4 + v_rand
    flen = 16
    if len(res) == 3  and int(res[1]) >= 0 and int(res[1]) < 1000:
        filename = filepath + file
        data = np.load(filename)
        sample = data['datapre']
        new_len = int(len(sample)*length_rate)
        new_data = np.zeros([new_len, flen])
        temp = [Utils.resampling(np.arange(0, len(x), 1), x, [0, len(x)], new_len)[1] for x in
                np.array(sample).transpose()]
        new_data[:, :] = np.array(temp).transpose()
        # plt.figure()
        # for i in range(0, 8):
        #     sample_channel = sample[:, i]
        #     plt.plot(sample_channel)
        # print()
        #
        # plt.show()
        np.savez_compressed('./data/lipcontrol/cutdata12/datapre%d-%d-%d-%d' % (int(res[0]), int(res[1]), int(res[2]), rate_ind),
                            datapre=new_data)