#%%
import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import glob
import torch
from pickle import dump, load
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler
import camp.StructuredGridOperators as so
import camp.FileIO as io

class Predictor():

    def __init__(self, model, name, plt_color, feats):
        self.model = model
        self.name = name
        self.plt_color = plt_color
        self.plt_line = None
        self.features = feats

        # variables for combined data set analysis

        self.combined_pred_train = None
        self.combined_pred_test = None
        self.combined_X_train = None
        self.combined_y_train = None
        self.combined_X_test = None
        self.combined_y_test = None
        self.combined_clf_trained = None

        # variables for CV analysis

        self.sss = None
        self.cv_pred_train = []
        self.cv_pred_test = []
        self.cv_X_train = []
        self.cv_y_train = []
        self.cv_X_test = []
        self.cv_y_test = []
        self.cv_subject_train = []
        self.cv_subject_test = []

    def get_X_feats(self, data):

        full_feat_list = feats = ['ctd', 'max', 't2', 'adc']
        feats = self.features
        if feats is None:
            return data

        if (type(data) is list) or (type(data) is tuple):
            data_sub = []
            for i, r in enumerate(data):
                data_feats = data[i].copy()
                data_feats_list = []
                for j, f in enumerate(full_feat_list):
                    if f in feats:
                        data_feats_list.append(data_feats[:, j, None, :, :])

                data_sub.append(torch.cat(data_feats_list, 1))

        else:
            data_feats_list = []
            for j, f in enumerate(full_feat_list):
                if f in feats:
                    data_feats_list.append(data[:, j, None, :, :])

            data_sub = (torch.cat(data_feats_list, 1))

        return data_sub


    def fit_predict_combined_data(self, data_dict):
        """Train classifiers w/ best hyper-parameters on the full TRAIN data-set created in classifer_grid_search"""

        #quad_feats = quad_feats.reshape(quad_feats.shape[0], -1).squeeze()
        X_train = self.get_X_feats(data_dict['X_train'])
        X_test = self.get_X_feats(data_dict['X_test'])

        self.combined_X_train = X_train.reshape(X_train.shape[0], -1).squeeze()
        self.combined_y_train = data_dict['y_train']
        self.combined_X_test = X_test.reshape(X_test.shape[0], -1).squeeze()
        self.combined_y_test = data_dict['y_test']

        print('Learning ' + self.name + '...')
        clf = self.model

        if isinstance(self.model, Pipeline):
            # Train on training data:
            clf.fit(self.combined_X_train, self.combined_y_train)

            # save combined trained clf to object
            self.combined_clf_trained = clf

            # save prediction to object
            self.combined_pred_train = clf.predict_proba(self.combined_X_train)[:, 1]
            self.combined_pred_test = clf.predict_proba(self.combined_X_test)[:, 1]

        elif self.name == 'NPV':
            self.combined_pred_test = data_dict['clinical_test'][:, 0]
            self.combined_pred_train = data_dict['clinical_train'][:, 0]

        elif self.name == 'NPV_blur':
            self.combined_pred_test = data_dict['clinical_test'][:, 1]
            self.combined_pred_train = data_dict['clinical_train'][:, 1]

        elif self.name == 'CTD':
            self.combined_pred_test = np.log(data_dict['clinical_test'][:, 2])  # set-up ctd as log
            self.combined_pred_train = np.log(data_dict['clinical_train'][:, 2])  # set-up ctd as log

        elif self.name == 'CEM240':
            self.combined_pred_test = data_dict['clinical_test'][:, 2] >= 240  # create binary mask
            self.combined_pred_train = data_dict['clinical_train'][:, 2] >= 240  # create binary mask

        else:
            print(f'WARNING: no prediction made for {self.name}.')


    def fit_predict_cv_data(self, cv_iter, X_data, y_data, subject_data, clinical_data=None, balance_test=False):

        self.sss = cv_iter
        clf = self.model

        X_data = self.get_X_feats(X_data)
        X_data = X_data.reshape(X_data.shape[0], -1).squeeze()

        for j, (train_index, test_index) in enumerate(cv_iter.split(X_data, y_data)):
            X_train, y_train, s_train = X_data[train_index], y_data[train_index], subject_data[train_index]
            X_test, y_test, s_test = X_data[test_index], y_data[test_index], subject_data[test_index]

            # flag to see if RandomUndersampling changes results, it doesn't really.
            if balance_test is True:
                rus = RandomUnderSampler(random_state=42)
                test_index, y_test = rus.fit_resample(test_index.reshape(-1, 1), y_test)
                test_index = test_index.reshape(-1)
                X_test, s_test = X_data[test_index], subject_data[test_index]

            # save X, y and subject data to object
            self.cv_X_train.append(X_train)
            self.cv_y_train.append(y_train)
            self.cv_X_test.append(y_test)
            self.cv_y_test.append(y_test)
            self.cv_subject_train.append(s_train)
            self.cv_subject_test.append(s_test)
            # W1 = compute_sample_weight('balanced', y_train)
            # W2 = compute_sample_weight('balanced', y_test)
            # print('Train N =' + str(y_train.shape[0]) + ', Test N = ' + str(y_train.shape[0]) +
            #       '. Train sample weight = ' + str([W1.min(), W1.max()] + ', Test sample weight = ' + str(W2))

            if isinstance(self.model, Pipeline):
                print('Training CV %d for %s...' % (j, self.name))

                # fit predictor
                clf.fit(X_train, y_train)

                # save prediction to object
                self.cv_pred_train.append(clf.predict_proba(X_train)[:, 1])
                self.cv_pred_test.append(clf.predict_proba(X_test)[:, 1])

            else:
                # save clinical predictions to object
                self.cv_pred_train.append(clinical_data[train_index])
                self.cv_pred_test.append(clinical_data[test_index])


    def get_pred_data_combined(self, set):
        # set  = 'train' or 'test'
        # subint = 1, 2, 3, or 4, representing the 4 subjects

        if set == 'train':
            data_pred = self.combined_pred_train
            data_y = self.combined_y_train

        elif set == 'test':
            data_pred = self.combined_pred_test
            data_y = self.combined_y_test

        return data_pred, data_y

    def get_pred_data_cv(self, set, subint=None):
        # set  = 'train' or 'test'
        # subint = 1, 2, 3, or 4, representing the 4 subjects

        if set == 'train':
            data_pred = self.cv_pred_train
            data_y = self.cv_y_train
            subject_data = self.cv_subject_train

        elif set == 'test':
            data_pred = self.cv_pred_test
            data_y = self.cv_y_test
            subject_data = self.cv_subject_test


        if subint is not None:
            subset_pred = []
            subset_y = []
            for f in range(0, len(data_pred)):
                mask = subject_data[f] == subint
                subset_pred.append(data_pred[f][mask])
                subset_y.append(data_y[f][mask])

            return subset_pred, subset_y

        else:

            return data_pred, data_y

    def get_recon_prediction_volume(self, rabbit_dict, r, roi='quad'):

        if roi == 'quad':
            itkfile = f'/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/Data/{r}/{r}_thermal_tissue.nrrd'
        elif roi == 'hist_dilate':
            itkfile = f'/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/Data/{r}/{r}_train_mask.nrrd'

        # load the clf
        clf = self.combined_clf_trained

        # load the labeled data set
        y_label = rabbit_dict[r][roi + '_labels']

        # Load the features for clf fitting or clinical data for the rabbit
        if isinstance(self.model, Pipeline):
            X_data = self.get_X_feats(rabbit_dict[r][roi + '_feats'])
            X_data = X_data.reshape(X_data.shape[0], -1).squeeze()
            y_proba = clf.predict_proba(X_data)[:, 1]

        elif self.name == 'NPV':
            y_proba = rabbit_dict[r]['quad_clinical'][:, 0]

        elif self.name == 'NPV_blur':
            y_proba = rabbit_dict[r]['quad_clinical'][:, 1]

        elif self.name == 'CTD':
            y_proba = np.log(rabbit_dict[r]['quad_clinical'][:, 2])

        elif self.name == 'CEM240':
            y_proba = rabbit_dict[r]['quad_clinical'][:, 2] >= 240

        # generate a 3D mask of the ROI, set prediction values to only voxels in the ROI
        mask = io.LoadITKFile(itkfile)
        pred_out_vol = mask.clone()
        if hasattr(y_proba, 'float'):
            pred_out_vol.data[mask.data.bool()] = y_proba.float()
        else:
            pred_out_vol.data[mask.data.bool()] = torch.tensor(y_proba).float()

        # generate a 3D mask of the ROI, set labels to only voxels in the ROI
        # (must re-load the itkfile because "mask" gets over-written above)
        mask = io.LoadITKFile(itkfile)
        lab_out_vol = mask.clone()
        if hasattr(y_label, 'float'):
            lab_out_vol.data[mask.data.bool()]= y_label.float()
        else:
            lab_out_vol.data[mask.data.bool()] = torch.tensor(y_label).float()

        return pred_out_vol, lab_out_vol

