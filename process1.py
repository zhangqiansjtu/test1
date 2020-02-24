import os
import re
import sys


userID = 6
cutindex = 100
filepath = 'D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/data1/'
files = os.listdir(filepath)
for file in files:
    pattern = re.compile(r'\d+')
    res = re.findall(pattern, file)
    if len(res) == 1:
        filename = filepath + file
        newname = str((int(res[0]))//10) + "-" + str(int(res[0])%10+950)+"-"+str(userID)
        os.rename(filename , "D:/zq/OneDrive/experiments/2019/20191013/lipcontrol/data1/"+newname+".pcm")