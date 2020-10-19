# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:05:57 2020

@author: claud
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing


filename='alz.csv'
original=pd.read_csv(filename,header=None,skiprows=1) #no header


original =original.T
original.to_csv('Transposedalz.csv',header=None)
df1 = pd.read_csv('Transposedalz.csv', header=None, nrows=1)
df1 = df1.drop([0],axis=1)
df1 = df1.T.reset_index(drop=True).T
#Drop headings and labels
original = original.drop([0])
original = original.replace(to_replace = np.nan, value = -99) 


x = original


scaled = preprocessing.scale(x,axis=0,with_mean =True, with_std = True)
scaled = abs(scaled)
scaled = np.log2(scaled, where=scaled>0)
scaled = abs(scaled)
scaled = preprocessing.scale(scaled,axis=0,with_mean =True, with_std = True)

#Create dataframe from scaled data
final = pd.DataFrame(scaled)
#Append the labels dataframe
alz = df1.append(final, ignore_index=True)
#Save as csv to be used in microarray_feature_selection.py
#alz = alz.T
#Save as csv to be used in microarray_feature_selection.py
alz.to_csv('processed_alz.csv',header=None,index = False)
