import re
import os
import numpy as np
import matplotlib.pyplot as plt

filepath = './data/lipcontrol/cutdata6/'
files = os.listdir(filepath)
for file in files:
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, file)
    if len(res) == 3 and int(res[0]) == 2 and int(res[1]) >= 40 and int(res[1]) < 50:
        filename = filepath + file
        data = np.load(filename)
        sample = data['datapre']
        plt.figure()
        for i in range(0, 8):
            sample_channel = sample[:,i]
            plt.plot(sample_channel)
        print()

plt.show()