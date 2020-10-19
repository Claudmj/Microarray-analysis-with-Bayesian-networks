# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 12:42:07 2020

@author: claud
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score, recall_score, f1_score


filename='alz_AMB_feature_selection_BN.csv'
df=pd.read_csv(filename) #no header


#Find common genes selected by both feature selections
filename_amb = 'alz_AMB_feature_selection_BN.csv'
filename_pcc = 'alz_feature_selection.csv'
amb_df = pd.read_csv(filename_amb,nrows=0) 
pcc_df = pd.read_csv(filename_pcc,nrows=0)
genes = pcc_df.append(amb_df,ignore_index=True)

#Get the clustering results of the BN clustering in cluster
cluster = []
data = []
for i in np.arange(0,len(df)):
    data.append(df.iloc[i,1:-1].tolist()) #literal_eval prevents a list as been written as a string
    if df.iloc[i,-1] == "Cluster 2":
            cluster.append(2)
    else:
            cluster.append(1)
 
            
#Make labels 1 and 0 
labels = []
for i in np.arange(0,len(df)):
    data.append(df.iloc[i,1:-1].tolist()) #literal_eval prevents a list as been written as a string
    if df.loc[i,'N0'] == "Healthy Control":
            labels.append(2)
    else:
            labels.append(1)
      
            
# Performance measures       
c = confusion_matrix(labels, cluster)
print(c)
accuracy = accuracy_score(labels, cluster)
accuracy2 = (c[0,0]+c[1,1])/71
print('The accuracy is:' ,accuracy) 
precision = c[1,1]/(c[1,1]+c[0,1])
precision2 = precision_score(labels,cluster)
recall = c[1,1]/(c[1,1]+c[1,0])
recall2 = recall_score(labels,cluster)
print('The precision is: ',precision2)
print('The recall is: ',recall)
f1 = f1_score(labels,cluster)
f12 = 2 * (precision * recall) / (precision + recall)
print('The F1 score is:' ,f1)
