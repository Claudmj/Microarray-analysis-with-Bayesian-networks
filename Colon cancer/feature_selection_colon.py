# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:24:46 2020

@author: claud
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing

#import data into dataframe
#golub = pd.read_csv("Golub_Dataset.txt", delimiter='\t')
filename='processed_colon.csv'
golu=pd.read_csv(filename,header=None,skiprows=0) #no header
n = 50
#transpose dataframe
golub = golu.T

#sort dataframe according to All/AML
#golub = golu.sort_values(by = 0) #column 0 contains the labels

#add ideal gene vector
golub['ideal_gene'] = golub[0].apply(lambda x: 0 if x == 'Serrated CRC' else 1)


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
top_features_genes.to_csv('PCC_colon_feature_selection.csv', header = False)

#Select random features for exp2
random = golub.sample(n= n, axis=1, replace=False)
#heatmap for random genes
#sns.heatmap(random.astype('float'),cmap="YlGnBu")
random['0'] = golub[0]
random.to_csv('random_colon.csv', header = True)