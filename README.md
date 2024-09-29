# gPRINT

Python 3.6.15; R 4.2.2 

Tools for gene-printing-based single-cell identification of human disease-specific cell subtypes across heterogeneous datasets


**#Install**

R packages :
Seurat 5.1.0 ; tidyverse 2.0.0 ;

Python packages : 
numpy 1.19.5 ; pandas 1.1.5 ; Keras 2.2.4 ; matplotlib 3.3.1 ; sklearn 0.0.post5 ; 

**#Data Preprocessing**

All scRNA-seq data were preprocessed using R (version 3.6.1). For single-cell data, genes with a total expression less than 2 were removed. When the dataset served as a reference set, cell types with fewer than ten cells were excluded, as models trained on very few cells can lead to overfitting. No specific requirements were imposed when the dataset served as a test set. Genes in both reference and test sets were arranged in the order of genes in the HG38 reference genome (by genomic termination point). The test set needed to have the same genes as the reference set; if some genes were missing in the test set, they were filled with zeros. The Seurat package's NormalizeData() and ScaleData() functions were used for data standardization and normalization in preparation for subsequent gPRINT method execution.

If the gPRINT_group algorithm is used, a library of marker genes (e.g., PanglaoDB database) corresponding to various cell types is utilized, or the Seurat package's FindAllMarkers() function is employed to calculate characteristic genes for each cell cluster. These genes are then transformed into one-dimensional "feature prints" according to the order of genes in the HG38 reference genome and input into the gPRINT_group model.

Detailed in /data/code/gPRINT/Data_preprocess.R or /data/code/gPRINT_group/External_test_group_preprocess.R


**#Building gPRINT Method for Single-Cell Annotation**

Based on the "gene prints," we employ a one-dimensional convolutional neural network (1D CNN) consisting of an input layer, five hidden layers, and an output layer. The input layer has the same number of nodes as the reference genes. The hidden layers include 2 convolutional layers, 1 pooling layer, 1 Flatten layer, and 1 fully connected layer. The pooling layer aims to retain main features, reduce computational load, and the Flatten layer connects the convolutional layers and Dense fully connected layer. The main purpose of the fully connected layer is to perform non-linear transformation on features extracted by the preceding convolutional layers, capturing correlations among these features and mapping them to the output space. The model's input dimension is the number of genes in the reference dataset, the convolutional layers have 16 convolutional kernels with a size of 3, and the Dense layer uses the softmax function.

Based on the "feature prints," we utilize backpropagation (BP) neural network. The BP neural network is a multi-layer feedforward neural network, and its training process mainly involves two stages: the forward propagation of signals and the backward propagation of errors. It uses the steepest descent method as the learning rule, continually adjusting the network's weights and thresholds through backpropagation to minimize the sum of squared errors.


**#Detailed in**

/code/


**#Accuracy and F1 Score Evaluation**

When machine learning is applied to classification tasks, metrics such as precision, recall, accuracy, F1-score, and receiver operating characteristic (ROC) curve are commonly used for model evaluation.


**#Demonstration**

If you want to select a model with an established database for cell type annotation, 


you can run the cell to cell level model: **gPRINT_reference.sh**


For example: bash gPRINT_reference.sh human_Pancreas349 Pancreas


Or run the group to group level model: **gPRINT_group_reference.sh**



For example: bash gPRINT_group_reference.sh human_Pancreas349 Pancreas



If you want to select a cell type annotation mapping for a reference dataset you provide yourself, 


you can run the cell to cell level model: **gPRINT.sh**


For example: bash gPRINT.sh Wk12_14 Wk17_18


Or run the group to group level model: **gPRINT_group.sh**


For example: bash gPRINT_group.sh Wk12_14 Wk17_18


