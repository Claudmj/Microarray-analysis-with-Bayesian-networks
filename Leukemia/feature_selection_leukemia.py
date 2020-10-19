# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:41:14 2020

@author: u20285826
"""

#DNA array analysis for cancer classification
#Feature selection
#Author: Alta de Waal

#Create an 'ideal' gene expression vector: For each of the 7000+ gene vectors, calculate the Pearson
#correlation with the ideal gene.

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing

#import data into dataframe
#golub = pd.read_csv("Golub_Dataset.txt", delimiter='\t')
filename='leukemia_big.csv'
golub=pd.read_csv(filename, header=None) #no header
n = 50
#transpose dataframe
golub = golub.T

#sort dataframe according to All/AML
golub = golub.sort_values(by = 0) #column 0 contains the labels

#add ideal gene vector
golub['ideal_gene'] = golub[0].apply(lambda x: 0 if x == 'ALL' else 1)


#calculate pearson correlation between all the genes (1:7129) and the ideal gene.
pearson_correlation = []
for i in np.arange(1,golub.shape[1] - 1):
    gene_corr = golub['ideal_gene'].corr(pd.to_numeric(golub.iloc[:,i]))
    pearson_correlation.append((i,gene_corr))
    
#sort the list of tuples (pearson_correlation)
sorted_pearson_correlation = sorted(pearson_correlation, key=lambda x: x[1], reverse=True)

#get the top correlated genes from the dataframe
top_features_index = [i[0] for i in sorted_pearson_correlation[0:n]]

top_features_genes = golub.iloc[:,top_features_index]
#visualise
sns.heatmap(top_features_genes.astype('float'))
#add labels
top_features_genes['Class_Type'] = golub[0]
#write to csv
top_features_genes.to_csv('PCC_Leukemia_feature_selection.csv', header = True)

#Select random features for exp2
random = golub.sample(n= 10, axis=1, replace=False)
#heatmap for random genes
#randmap = sns.heatmap(random.astype('float'),cmap="YlGnBu")
#randmap
random['ALL/AML'] = golub[0]
random.to_csv('random_leukemia.csv', header = True)