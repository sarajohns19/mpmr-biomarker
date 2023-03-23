#%%
import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import os
import glob
import torch
import pickle
#from pickle import dump, load
import numpy as np
import pandas as pd
from skimage import measure
#import CAMP.camp.FileIO as io
import camp.FileIO as io
import sklearn.metrics as metrics
#import CAMP.camp.StructuredGridOperators as so
import camp.StructuredGridOperators as so
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline, Pipeline
#from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_validate
from imblearn.under_sampling import RandomUnderSampler

import biomarker_class
from biomarker_class import Predictor

import tools as tls
import matplotlib
matplotlib.use('module://backend_interagg')
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

# full_feat_list = ['ctd', 'max', 't2', 'adc']
# if feats:
#     for j, r in enumerate(rabbit_list):
#         quad_feats = quadROI_feats[j].copy()
#         quad_feats_list = []
#         for i, f in enumerate(full_feat_list):
#             if f in feats:
#                 quad_feats_list.append(quad_feats[:, i, None, :, :])
#
#         quadROI_feats[j] = (torch.cat(quad_feats_list, 1))
#

def compile_data(data_mask='quad', feats=None):

    # Create empty data_dict
    data_dict = {}
    # Create empty rabbit_dict
    data_dir = dataroot + 'Data/'
    rabbit_list = [x.split('/')[-1] for x in sorted(glob.glob(f'{data_dir}/18_*'))]
    rabbit_dict = {f'{x}': {} for x in rabbit_list}

    # Get linearized data for training the classifiers from all subjects
    # trainROI = small ROI surrounding histology (not using)
    # quadROI = entire quadricep
    trainROI_feats, trainROI_labels, quadROI_feats, quadROI_labels = tls.get_data(dataroot, feats=feats)  # feats=['ctd', 'max', 't2', 'adc'])
    trainROI_clinical, quadROI_clinical = tls.get_data_clinical(dataroot)

    if data_mask == 'quad':

        return quadROI_feats, quadROI_labels, quadROI_clinical

    if data_mask == 'train':
        return trainROI_feats, trainROI_labels, trainROI_clinical


def define_classifiers():
    """Create classifiers with best hyper-parameters from gridsearch results"""

    # ---DEFINE CLASSIFIERS---
    # since the GridSearchCV was applied to different training data set than the one used here,
    # I need to completely re-train the classifiers using the appropriate data set:
    #   - for combined data results, use the data_dict train/test set
    #   - for subject-specific results, use 10-fold CV on full data set (no validation set)

    # create dictionary of the best classifiers to train from GridSearch output
    # (load output of 'classifier_grid_search.py')

    lrc_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('sample', RandomUnderSampler(
                                       sampling_strategy='majority',
                                       random_state=0)),
                                   ('model', LogisticRegression(
                                       class_weight='balanced',
                                       max_iter=1000,
                                       penalty='l1',
                                       solver='liblinear',
                                       C=.001
                                   ))])
    rfc_pipeline = Pipeline(steps=[('scaler', StandardScaler()),
                                   ('sample', RandomUnderSampler(
                                       sampling_strategy='majority',
                                       random_state=0)),
                                   ('model', RandomForestClassifier(
                                       n_jobs=16,
                                       random_state=0,
                                       class_weight='balanced',
                                       min_samples_split=2,
                                       n_estimators=400,
                                       max_depth=40
                                   ))])

    lrc = Predictor(model=lrc_pipeline, name='LRC-MP', plt_color=matplotlib._cm._tab10_data[9], feats=['ctd', 'max', 't2', 'adc'])
    rfc = Predictor(model=rfc_pipeline, name='RFC-MP', plt_color=matplotlib._cm._tab10_data[1], feats=['ctd', 'max', 't2', 'adc'])

    lrc_th = Predictor(model=clone(lrc_pipeline), name='LRC', plt_color=matplotlib._cm._tab10_data[9], feats=['ctd', 'max'])
    rfc_th = Predictor(model=clone(rfc_pipeline), name='RFC', plt_color=matplotlib._cm._tab10_data[1], feats=['ctd', 'max'])

    #clfs = [lrc, rfc]
    return lrc, rfc, lrc_th, rfc_th

def train_classifiers(data_dict_folder, instance_folder, test_per, k):


    # ---CREATE INSTANCE FOLDER----

    instance_parameters = {}
    instance_parameters['rabbit_dict_file'] = f'{dataroot}Dictionaries/Combined Datasets/{data_dict_folder}/rabbit_dictionary.pkl'
    instance_parameters['data_dict_file'] = f'{dataroot}Dictionaries/Combined Datasets/{data_dict_folder}/data_splits_dictionary.pkl'
    instance_parameters['cv_kfolds'] = k
    instance_parameters['cv_testpercent'] = test_per
    if not os.path.exists(f'{dataroot}Instances/{instance_folder}'):
        os.mkdir(f'{dataroot}Instances/{instance_folder}')

    # ---LOAD data_dict and rabbit_dict ----------

    with open(instance_parameters['data_dict_file'], 'rb') as f:
        data_dict = pickle.load(f)

    with open(instance_parameters['rabbit_dict_file'], 'rb') as f:
        rabbit_dict = pickle.load(f)

    feats = instance_parameters['features'] = data_dict['features']


    # ---SET-UP: CV TRAINING------

    K = k
    test_percent = test_per

    # Get linearized data for training the classifiers from all subjects
    #quadROI_feats, quadROI_labels, quadROI_clinical = compile_data(data_mask='quad', feats=feats)
    quadROI_feats, quadROI_labels, quadROI_clinical = tls.get_data_roi(dataroot, roi='quad', feats=feats)

    # create linearlized vector of the subject numbers
    quadROI_subject = []
    for i, r in enumerate(rabbit_dict):
        quad_labels = quadROI_labels[i].clone()
        quadROI_subject.append(torch.ones(quad_labels.shape) * i + 1)
        del quad_labels

    # concatenate data from all subjects
    quad_feats = torch.cat(quadROI_feats, 0)
    #quad_feats = quad_feats.reshape(quad_feats.shape[0], -1).squeeze()
    quad_labels = torch.cat(quadROI_labels, 0)
    quad_subjects = torch.cat(quadROI_subject, 0)
    quad_clinical = torch.cat(quadROI_clinical, 0)

    # create the CV data split object (StratifiedShuffleSplit
    X = quad_feats
    y = quad_labels
    group = quad_subjects
    sss = StratifiedShuffleSplit(n_splits=K, test_size=test_percent, random_state=0)

    # ---DEFINE PREDICTOR CLASS OBJECTS-----
    # classifiers to train:
    lrc, rfc, lrc_th, rfc_th = define_classifiers()
    # clinical metrics from MR data:
    npv = Predictor(model='Binary', name='NPV', plt_color='blue', feats=None)
    blur = Predictor(model='Continuous', name='NPV_blur', plt_color='blue', feats=None)
    ctd = Predictor(model='Continuous', name='CTD', plt_color='red', feats=None)
    cem240 = Predictor(model='Binary', name='CEM240', plt_color='red', feats=None)

    # ----TRAIN/TEST CLASSIFIERS: Combined data-set--------
    print('Performing training on TRAIN dataset...')
    # train classifiers on combined dataset
    lrc.fit_predict_combined_data(data_dict)
    rfc.fit_predict_combined_data(data_dict)
    lrc_th.fit_predict_combined_data(data_dict)
    rfc_th.fit_predict_combined_data(data_dict)
    npv.fit_predict_combined_data(data_dict)
    blur.fit_predict_combined_data(data_dict)
    ctd.fit_predict_combined_data(data_dict)
    cem240.fit_predict_combined_data(data_dict)


    # ----TRAIN/TEST CLASSIFIERS: CV data-set--------
    print('Performing CV training on full data-set...')
    # train classifiers for each cv fold
    lrc.fit_predict_cv_data(sss, X, y, group, clinical_data=None)
    rfc.fit_predict_cv_data(sss, X, y, group, clinical_data=None)
    lrc_th.fit_predict_cv_data(sss, X, y, group, clinical_data=None)
    rfc_th.fit_predict_cv_data(sss, X, y, group, clinical_data=None)
    npv.fit_predict_cv_data(sss, X, y, group, clinical_data=quad_clinical[:, 0])
    blur.fit_predict_cv_data(sss, X, y, group, clinical_data=quad_clinical[:, 1])
    ctd.fit_predict_cv_data(sss, X, y, group, clinical_data=np.log(quad_clinical[:, 2]))
    cem240.fit_predict_cv_data(sss, X, y, group, clinical_data=quad_clinical[:, 2] >= 240)

    # ----SAVE RESULTS-----------------------------
    # save Predictor objects to file
    pdrs = [lrc, rfc, lrc_th, rfc_th, npv, blur, ctd, cem240]
    for i in pdrs:

        with open(f'{dataroot}Instances/{instance_folder}/{i.name}.pkl', 'wb') as f:
            pickle.dump(i, f)


    with open(f'{dataroot}Instances/{instance_folder}/parameter_dict.pkl', 'wb') as f:
        pickle.dump(instance_parameters, f)

if __name__ == '__main__':

    # --- USER INPUTS ------
    data_dict_folder = '70per_training_cor'  # Folder for combined data dictionaries
    instance_folder = '70per_10Kfolds_cor'
    K = 10
    test_per = 0.3


    global dataroot
    dataroot = '/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/'


    train_classifiers(data_dict_folder, instance_folder, test_per, K)

