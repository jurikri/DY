# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:46:38 2022

@author: MSBak
"""

mainpath = 'C:\\Users\\MSBak\\Desktop\\x\\'
filename = 'CellCounter_s210226_1L.xml'

file = open(mainpath + filename, 'r')
tsave = []
while True:
    line = file.readline()
    if not line:
        break
    # print(line)
    tsave.append(line)
file.close()

mssave = []
for row in range(len(tsave)-2):
    if tsave[row] == '         <Marker>\n':
        
        s = tsave[row+1].find('<MarkerX>')
        e = tsave[row+1].find('</MarkerX>')
        X = tsave[row+1][s+9:e]
        
        s = tsave[row+2].find('<MarkerY>')
        e = tsave[row+2].find('</MarkerY>')
        Y = tsave[row+2][s+9:e]
        
        mssave.append([X, Y])
        
import pandas as pd
import numpy as np

mssave = np.array(mssave)

mssave = np.concatenate((np.zeros(mssave.shape), mssave), axis=1)
mssave.shape
msout = pd.DataFrame(mssave)

msout.to_csv(mainpath + filename[:-3] + 'csv', index=False)









        
        
        
        









