# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:00:23 2020

@author: claud
"""

import pandas as pd
import numpy as np

filename = 'leukemia_big.csv'
original = pd.read_csv(filename, header=None)
original = original.T

df1 = original.pop(0) # remove column Class_Type and store it in df1
original['Class_Type']=df1 # add Class_Type series as a 'new' column.



original.to_csv('leukemia_original.csv',header=None)