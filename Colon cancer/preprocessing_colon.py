# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:40:23 2020

@author: claud
"""

#Preprocessing expression data for DNA array analysis for cancer classification
#Preprocessing
#Author: Claudio Jardim

import pandas as pd
import numpy as np
from sklearn import preprocessing


filename='Colon.csv'
original=pd.read_csv(filename,header=None) #no header
original.to_csv('Transposedcolon.csv',header=None)
original = original.drop([0],axis=0)
original = original.T.reset_index(drop=True).T
original = original.drop([0],axis=1)
original = original.replace(to_replace = np.nan, value = -99) 
df1 = pd.read_csv('Transposedcolon.csv', header=None, nrows=1)
df1 = df1.drop([0,1],axis=1)
df1 = df1.T.reset_index(drop=True).T
x = original.to_numpy()



scaled = preprocessing.scale(x,axis=0,with_mean =True, with_std = True)
#scaled = abs(scaled)
scaled = np.log2(scaled, where=scaled>0)
scaled = abs(scaled)
scaled = preprocessing.scale(x,axis=0,with_mean =True, with_std = True)
#Create dataframe from scaled data
final = pd.DataFrame(scaled)
#Append the labels dataframe
alz = df1.append(final, ignore_index=True)
#Save as csv to be used in microarray_feature_selection.py

#Save as csv to be used in microarray_feature_selection.py
alz.to_csv('processed_colon.csv',header=None,index = False)