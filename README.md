# mpmr-biomarker
Supervised learning on multiparametric MRI for predicting FUS treatment outcomes 

# **Project Summary**
The dataset for this project consists of several co-registered multi-parametric MRI images in rabbit quadriceps muscles, such as 4D temperature rise maps, 3D ADC maps, and 3D T2w anatomical images. MR Images were acquired 10 minutes prior-to and 10 minutes after a non-invasive thermal ablation procedure called focused ultrasound (FUS), which aims to kill tumors in the quadriceps. MR images were registered to 3D-rendered H&E sections acquired 3-5 days after treatment, which were segmented into a binary mask of non-viable (positive class) and viable (negative class) voxels as determined by pathology. 

This project implements supervised machine learning in Python to learn an acute, multi-parametric MRI biomarker of thermal cell death (non-viable), using the pathology ground-truth as the classifier label. Specifically, a logistic regression classifier and a random forest classifier are explored. These classifiers are compared to the gold-standard MRI biomarker of thermal ablation non-viability, which is Contrast-Enhanced T1w imaging. 

**General Methods:** 
* Merge and Reshape MRI data into N x M feature space, where N=number voxels and M=number of features
* 5-fold Gridsearch hyper-parameter tuning on training dataset
* Train/test best estimators
* ROC Analysis of best estimators
* Cross-validition for inter-subject variability 
* Performance statistics (f1-score, Presicion, Recall)
  
# **Order of data processing**
1. [compile_data](/compile_data.py)  -  converts .mat MRTI data to NIFTI format
2. [preprocess_data](/preprocess_data.py)  -   saves MPMR features and clinical data as linear format
3. [get_combined_dataset](/get_combined_dataset.py)  -  creates rabbit_dictionary.pkl and data_splits_dictionary.pkl, which are the train/validation data splits used for grid_search and training classifiers 
4. [classifier_grid_search](/classifier_grid_search.py)  -   runs hyper-parameter grid_search classifiers, saves best estimators in grid search_dictionary.pkl
5. [train_classifers](/train_classifers.py)   -   defines classifiers with optimal hyper-parameters as “Predictor” objects, trains all Predictor objects on combined data splits and CV data splits
6. [evaluate_biomarkers](/evaluate_biomarkers.py)  -   computes optimal thresholds, ROCs, and performance stats for combined data set and CV folds. Creates several charts for visualizing classifier performance. Creates figures for publication. 
7. [create_tables](/create_tables.py)  -   creates tables for publication 

# **Other scripts**
1. [biomarker_class](/biomarker_class.py)  -   defines the Predictor class, which stores information about trained estimators and manually-defined binary segmentations. Includes methods for retrieving subject-specific data, data splits, and wrappers for train/fit. 
2. [tools](/tools.py)  -   other functions needed for compiling data and calculating classifier performance stats

# **Credits & Dependencies**
* Authored by Sara L. Johnson

* This project expanded upon a forked repository created by Blake Zimmerman, PhD which can be found at: https://github.com/blakezim/noncontrastbiomarkers. 

* In addition to standard libraries, project dependencies include: https://github.com/blakezim/CAMP
  
