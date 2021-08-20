# Microarray-analysis-with-Bayesian-networks
Microarray analysis with Bayesian hierarchical clustering (BHC) and Bayesian network (BN) clustering on three microarray datasets. Pearson's correlation coefficient (PCC) and an augmented Markov blanket (AMB) are used for feature selection.
The 3 datasets are: a leukemia dataset, a colon cancer dataset and an Alzheimer's disease dataset. For each dataset the data is first processed, feature selection is performed (if needed) and then clustering is performed. There are separate scripts to performe the preprocessing, feature selection and clustering for each dataset.


## Preprocessing
The preprocessing scripts are named 'preprocessing_datasetname'. The correct dataset should be used with the matching processing script. Once the data has been preprocessed we can perform feature selection. (The leukemia dataset 'leukemia_big.csv' has already been preprocessed).

## Feature selection
PCC feature selection can be performed using the scripts labelled 'PCC_feature_selection_datasetname'. The dataset created by the preprocessing script should be used for the feature selection. The correct dataset ('processed_datasetname') should be used with the matching feature selection script. AMB feature selection was done in Bayesialab software.

## Clustering
The BHC scripts are named 'BHC_datasetname'. The dataset created by the feature selection scipt should be used here. The correct dataset ('PCC_datasetname_feature_selection') should be used with the matching BHC script. BN clustering was performed in Bayesialab software. The BHC scripts named 'BHC_datasetname_mb'should be used for the AMB feature selection datasets made by Bayesialab. The correct dataset ('datasetname_AMB_feature_selection_BN') should be used with the matching BHC script.

## BN checks
The script named 'BN_checks'can be used to calculate the performance metrics of the BN clustering. The 'datasetname_AMB_feature_selection_BN' and 'datasetname_PCC_feature_selection_BN' files from Bayesialab can be checked using this script.

## Bayesialab
The BN clustering results done in Bayesialab.
