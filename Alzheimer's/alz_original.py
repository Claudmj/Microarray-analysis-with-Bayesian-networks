# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:20:06 2020

@author: claud
"""

import pandas as pd
import numpy as np

filename = 'alz.csv'
original = pd.read_csv(filename)


df1 = original.pop('Class_Type') # remove column Class_Type and store it in df1
original['Class_Type']=df1 # add Class_Type series as a 'new' column.



original.to_csv('alz_original.csv',header=None)