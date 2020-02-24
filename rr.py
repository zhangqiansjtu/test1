import os
import re

filepath = './data/1/'
files = os.listdir(filepath)
for file in files:
    if re.match('.*0[7]$', file):
        print()