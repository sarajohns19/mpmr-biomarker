
import tools as tls
import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from skimage import measure
#import CAMP.camp.FileIO as io
#import camp.FileIO as io
import sklearn.metrics as metrics
#import CAMP.camp.StructuredGridOperators as so
#import camp.StructuredGridOperators as so
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.stats import uniform, randint

import tools as tls
import matplotlib
matplotlib.use('module://backend_interagg')
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
plt.ion()

def define_estimators():


    # estimator_dict = {'LRC-unbalanced': {}, 'LRC-undersampled': {}, 'LRC-oversampled': {},
    #                   'RFC-unbalanced': {}, 'RFC-undersampled': {}, 'RFC-oversampled': {},
    #                   'BRFC-unbalanced': {}}
    # estimator_dict = {'RFC-unbalanced': {}, 'RFC-undersampled': {}, 'BRFC-unbalanced': {}}

    estimator_dict = {'LRC-unbalanced': {}, 'RFC-unbalanced': {}}

    estimator_dict['LRC-unbalanced']['pipeline'] = Pipeline(steps=[('scaler', StandardScaler()),
                                                                   ('model', LogisticRegression())])
    estimator_dict['LRC-unbalanced']['param_grid'] = {"model__class_weight": ['balanced'],
                                                      "model__penalty": ["l1", "l2"],
                                                      "model__C": [0.0005, 0.001, 0.005, 0.01, 0.1, 1, 50],
                                                      "model__max_iter": [1000],
                                                      "model__solver": ['liblinear']}

    # estimator_dict['LRC-undersampled']['pipeline'] = Pipeline(steps=[('scaler', StandardScaler()),
    #                                                                  ('sample', RandomUnderSampler()),
    #                                                                  ('model', LogisticRegression())])
    # estimator_dict['LRC-undersampled']['param_grid'] = {"sample__sampling_strategy": ['majority', .5, .75],
    #                                                     "model__penalty": ["l1", "l2"],
    #                                                     "model__C": [0.0005, 0.001, 0.005, 0.01, 0.1, 1, 50],
    #                                                     "model__max_iter": [1000],
    #                                                     "model__solver": ['liblinear']}
    #
    # estimator_dict['LRC-oversampled']['pipeline'] = Pipeline(steps=[('scaler', StandardScaler()),
    #                                                                 ('sample', RandomOverSampler()),
    #                                                                 ('model', LogisticRegression())])
    # estimator_dict['LRC-oversampled']['param_grid'] = {"sample__sampling_strategy": [.25, .5, .75],
    #                                                    "model__penalty": ["l1", "l2"],
    #                                                    "model__C": [0.0005, 0.001, 0.005, 0.01, 0.1, 1, 50],
    #                                                    "model__max_iter": [1000],
    #                                                    "model__solver": ['liblinear']}


    estimator_dict['RFC-unbalanced']['pipeline'] = Pipeline(steps=[('scaler', StandardScaler()),
                                                                   ('model', RandomForestClassifier(n_jobs=16, random_state=0))])  # , n_jobs=16
    estimator_dict['RFC-unbalanced']['param_grid'] = {"model__class_weight": ['balanced', 'balanced_subsample'],
                                                      "model__min_samples_split": [2, 0.005, 0.01, 0.015], # 0.03, 0.05, 0.07, 0.1, 0.13, 0.15, 0.17, .2],
                                                      "model__n_estimators": [10, 50, 100, 150, 200, 250, 300, 350, 400, 450], #, 500, 600, 700, 800],
                                                      "model__max_depth": [5, 10, 20, 30, 40, 50]} #, 60, 70, 80]}

    # estimator_dict['RFC-unbalanced']['param_grid'] = {"model__class_weight": [None, 'balanced', 'balanced_subsample'],
    #                                                   "model__min_samples_split": uniform(0.01, 0.199),
    #                                                   "model__n_estimators": randint(10, 1000),
    #                                                   "model__max_depth": randint(10, 100)}
    #
    # estimator_dict['RFC-undersampled']['pipeline'] = Pipeline(steps=[('scaler', StandardScaler()),
    #                                                                  ('sample', RandomUnderSampler()),
    #                                                                  ('model', RandomForestClassifier(n_jobs=16, random_state=0,
    #                                                                                                   min_samples_split=2))])     # , n_jobs=16
    # estimator_dict['RFC-undersampled']['param_grid'] = {"sample__sampling_strategy": ['majority', .5, .75],
    #                                                  # "model__min_samples_split": uniform(0.01, 0.199),
    #                                                   "model__n_estimators": randint(10, 1000),
    #                                                   "model__max_depth": randint(10, 100)}
    #
    # estimator_dict['BRFC-unbalanced']['pipeline'] = Pipeline(steps=[('scaler', StandardScaler()),
    #                                                                 ('model', BalancedRandomForestClassifier(n_jobs=16, random_state=0,
    #                                                                                                         min_samples_split=2))])
    # estimator_dict['BRFC-unbalanced']['param_grid'] = {"model__sampling_strategy": ['majority', 'all'],
    #                                                   # "model__min_samples_split": uniform(0.01, 0.199),
    #                                                    "model__n_estimators": randint(10, 1000),
    #                                                    "model__max_depth": randint(10, 100)}
    return estimator_dict

def make_hyperparameter_plots(cresults, titlestr, x_var, y_vars, log_x=False, colors=['blue', 'red', 'green'],
                              linstyles=['-', '--', '.-'], markers=['o', '^', 's'], save_str=''):

    # plots score over hyperparameters for test data for CV GridSearch.
    # error bars are for the difference folds

    x_var = 'param_' + x_var
    y_vars = ['param_' + y for y in y_vars]
    for col in y_vars:
        cresults[col] = cresults[col].astype('string')
    y_params = [list(np.unique(cresults[y].values)) for y in y_vars]

    cv_col = [x for x in cresults.columns.values if 'split' in x]
    #titlestr = 'AUC ROC(cv=%d) \n %s' % (len(cv_col), titlestr)
    x_var_string = x_var.split('_')
    x_label = ' '.join(x_var_string[x_var_string.index('')+1:])

    if log_x is True:
        xscale = 'log'
    else:
        xscale = 'linear'

    if len(y_vars) == 1:
        fig = plt.figure(figsize=(5, 5))
        for i, p0 in enumerate(y_params[0]):
            data = cresults.loc[(cresults[y_vars[0]] == p0)]
            plt.errorbar(data[x_var], data['mean_test_score'], yerr=data['std_test_score'],
                         ecolor='black', capsize=3, markersize=6,
                         c=colors[i],
                         ls=linstyles[i],
                         marker=markers[0],
                         label=str(p0))

    elif len(y_vars) == 2:
        fig = plt.figure(figsize=(5, 5))
        for i, p0 in enumerate(y_params[0]):
            for j, p1 in enumerate(y_params[1]):
                data = cresults.loc[(cresults[y_vars[0]] == p0) & (cresults[y_vars[1]] == p1)]
                plt.errorbar(data[x_var], data['mean_test_score'], yerr=data['std_test_score'],
                             ecolor='black', capsize=3, markersize=6,
                             c=colors[i],
                             ls=linstyles[j],
                             marker=markers[0],
                             label=str(p0) + '; ' + str(p1))

    elif len(y_vars) == 3:
        fig = plt.figure(figsize=(5, 5))
        for i, p0 in enumerate(y_params[0]):
            for j, p1 in enumerate(y_params[1]):
                for k, p2 in enumerate(y_params[2]):
                    data = cresults.loc[(cresults[y_vars[0]] == p0) & (cresults[y_vars[1]] == p1) & (cresults[y_vars[2]] == p2)]
                    plt.errorbar(data[x_var], data['mean_test_score'], yerr=data['std_test_score'],
                                 ecolor='black', capsize=3, markersize=6,
                                 c=colors[i],
                                 ls=linstyles[j],
                                 marker=markers[k],
                                 label=str(p0) + '; ' + str(p1) + '; ' + str(p2))

    else:
        raise ValueError('Maximum of 3 parameters can be plotted at once')

    plt.legend(loc='best')
    plt.xscale(xscale)
    plt.ylabel('AUC Score')
    plt.xlabel(x_label)
    plt.title(titlestr)
    plt.ylim(.85, 1)
   # plt.savefig(f'{dataroot}Output/PaperFigs/{save_str}_hyperparams.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{dataroot}Output/{save_str}_hyperparams.svg', dpi=150, bbox_inches='tight', pad_inches=0)

    return fig

def run_grid_search(est_dict, clf_name, X_train, y_train, X_test, CV=5, scorer='roc_auc', verbose=1):

    clf = est_dict[clf_name]['pipeline']
    param_grid = est_dict[clf_name]['param_grid']

    grid_ub = GridSearchCV(clf, param_grid,
                           scoring=scorer,
                           cv=CV,
                           n_jobs=-1,
                           refit=True,
                           verbose=verbose)

    grid_ub.fit(X_train, y_train)
    print(f'Best score: {grid_ub.best_score_} with param: {grid_ub.best_params_}')
    cvresults = pd.DataFrame(grid_ub.cv_results_)

    y_train_proba = grid_ub.predict_proba(X_train)[:, 1]
    y_test_proba = grid_ub.predict_proba(X_test)[:, 1]

    est_dict[clf_name]['CV_results'] = cvresults
    est_dict[clf_name]['CV_best_parameters'] = grid_ub.best_params_
    est_dict[clf_name]['CV_best_estimator'] = grid_ub.best_estimator_
    est_dict[clf_name]['CV_best_AUC'] = grid_ub.best_score_
    est_dict[clf_name]['CV_y_train_proba'] = y_train_proba
    est_dict[clf_name]['CV_y_test_proba'] = y_test_proba

    return est_dict

def run_random_search(est_dict, clf_name, X_train, y_train, X_test, iter= 50, CV=5, scorer='roc_auc', verbose=1):

    clf = est_dict[clf_name]['pipeline']
    param_grid = est_dict[clf_name]['param_grid']

    grid_ub = RandomizedSearchCV(clf, param_grid,
                           n_iter=iter,
                           scoring=scorer,
                           cv=CV,
                           n_jobs=-1,
                           refit=True,
                           verbose=verbose)

    grid_ub.fit(X_train, y_train)
    print(f'Best score: {grid_ub.best_score_} with param: {grid_ub.best_params_}')
    cvresults = pd.DataFrame(grid_ub.cv_results_)

    y_train_proba = grid_ub.predict_proba(X_train)[:, 1]
    y_test_proba = grid_ub.predict_proba(X_test)[:, 1]

    est_dict[clf_name]['CV_results'] = cvresults
    est_dict[clf_name]['CV_best_parameters'] = grid_ub.best_params_
    est_dict[clf_name]['CV_best_estimator'] = grid_ub.best_estimator_
    est_dict[clf_name]['CV_best_AUC'] = grid_ub.best_score_
    est_dict[clf_name]['CV_y_train_proba'] = y_train_proba
    est_dict[clf_name]['CV_y_test_proba'] = y_test_proba

    return est_dict

def compare_rocs(est_dict, y_data_train, y_data_test):

    clist = [x for x in range(0, 10)]
    font = {'family': 'calibri',
            'size': 12}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'monospace'

    axs = (plt.figure(figsize=[12, 6])
           .subplots(1, 2))

    for ax in axs:
        for i, clf in enumerate(est_dict):
            y_train_proba = est_dict[clf]['CV_y_train_proba']
            y_test_proba = est_dict[clf]['CV_y_test_proba']
            tls.plot_ROC(ax, y_train_proba, y_data_train, 'Train-' + clf, matplotlib._cm._tab10_data[clist[i]],
                         line_type='--')
            tls.plot_ROC(ax, y_test_proba, y_data_test, 'Test-' + clf, matplotlib._cm._tab10_data[clist[i]],
                         line_type='-')

        plt.gca().tick_params(labelsize=font['size'])
        ax.legend(prop={'size': font['size']})
        ax.legend(loc='lower right')

        plt.gca()
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f'ROC Analysis: Optimal Estimators')
        ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
        ax.xaxis.set_label_position('top')

    axs[1].set_ylim(.75, 1)
    axs[1].set_xlim(0, .25)
    plt.savefig(f'{dataroot}Output/ROC_best_estimators.svg', dpi=150, bbox_inches='tight', pad_inches=0)


def finetune_rfc(data_dict, estimator_dict, min_sample_splits_vec, max_depth_vec):


    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    for est in estimator_dict:

        print(est + ':')
        font = {'family': 'calibri',
                'size': 12}
        matplotlib.rc('font', **font)
        plt.rcParams['font.family'] = 'monospace'

        axs = (plt.figure(figsize=[12, 6])
               .subplots(1, 2, sharey=True))

        clf = estimator_dict[est]['CV_best_estimator']
        clf.set_params(model__n_estimators=500)

        auc_train = []
        auc_test = []
        for i, val in enumerate(min_sample_splits_vec):

            print('training model for %i/%i min_sample_split values...' % (i, len(min_sample_splits_vec)))
            clf.set_params(model__min_samples_split=val)
            clf.fit(X_train, y_train)

            tpr, fpr, tholds = metrics.roc_curve(y_train, clf.predict_proba(X_train)[:, 1])
            auc_train.append(metrics.auc(tpr, fpr))

            tpr, fpr, tholds = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
            auc_test.append(metrics.auc(tpr, fpr))

        axs[0].plot(min_sample_splits_vec, auc_train, 'b', label='Train')
        axs[0].plot(min_sample_splits_vec, auc_test, 'r', label='Test')
        axs[0].set_xlabel('min_sample_split')
        axs[0].set_ylabel('AUC')
        axs[0].set_title(est)

        clf = estimator_dict['RFC-unbalanced']['CV_best_estimator']
        clf.set_params(model__n_estimators=500)
        clf.set_params(model__min_samples_split=0.1)

        auc_train = []
        auc_test = []
        for i, val in enumerate(max_depth_vec):

            print('training model for %i/%i max_depth values...' % (i, len(max_depth_vec)))
            clf.set_params(model__max_depth=val)
            clf.fit(X_train, y_train)

            tpr, fpr, tholds = metrics.roc_curve(y_train, clf.predict_proba(X_train)[:, 1])
            auc_train.append(metrics.auc(tpr, fpr))

            tpr, fpr, tholds = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
            auc_test.append(metrics.auc(tpr, fpr))

        axs[1].plot(max_depth_vec, auc_train, 'b', label='Train')
        axs[1].plot(max_depth_vec, auc_test, 'r', label='Test')
        axs[1].set_xlabel('max_depth')
        axs[1].set_ylabel('AUC')
        axs[1].set_title(est)
        plt.show()

def create_plots():

    # ---LOAD DICTIONARIES----
    with open(f'{dict_savepath}/gridsearch_dictionary.pkl', 'rb') as f:
        est_dict = pickle.load(f)

    with open(f'{dict_savepath}/data_splits_dictionary.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # ---HYPER-PARAMETER PLOTs---
    # 1. LRC - unbalanced
    clf = 'LRC-unbalanced'
    titl = 'LRC Hyperparameter AUC'
    cvresults = est_dict[clf]['CV_results'].copy()
    cvresults['param_model__class_weight'] = cvresults['param_model__class_weight'].fillna('None')
    make_hyperparameter_plots(cvresults, titl, 'model__C', ['model__penalty'], log_x=True, save_str='LRC')

    # 2. RFC - unbalanced
    clf = 'RFC-unbalanced'
    titl = 'RFC Hyperparameter AUC'
    cvresults = est_dict[clf]['CV_results'].copy()
    cvresults['param_model__class_weight'] = cvresults['param_model__class_weight'].fillna('None')
    cvresults.loc[cvresults['param_model__min_samples_split'] == 2, 'param_model__min_samples_split'] = 0.0001
    # min_samples_split
    cvplt = cvresults.loc[(cvresults['param_model__n_estimators'] == 800) & (cvresults['param_model__max_depth'] == 60)]
    make_hyperparameter_plots(cvplt, titl, 'model__min_samples_split', ['model__class_weight'], save_str='RFC_minsamp')
    # n_estimators
    cvplt = cvresults.loc[
        (cvresults['param_model__min_samples_split'] == 0.0001) & (cvresults['param_model__max_depth'] == 60)]
    make_hyperparameter_plots(cvplt, titl, 'model__n_estimators', ['model__class_weight'], save_str='RFC_estimators')
    # max_depth
    cvplt = cvresults.loc[
        (cvresults['param_model__min_samples_split'] == 0.0001) & (cvresults['param_model__n_estimators'] == 800)]
    make_hyperparameter_plots(cvplt, titl, 'model__max_depth', ['model__class_weight'], save_str='RFC_depth')

    # ## Plot best estimator ROCs
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    compare_rocs(est_dict, y_train, y_test)

    # ________________________________________________________________________________________________________
    ### 9/8/22 no longer use this ###

    #  # 2. LRC - undersampled
    #  clf = 'LRC-undersampled'
    #  cvresults = estimator_dict[clf]['CV_results']
    #  make_hyperparameter_plots(cvresults, clf, 'model__C', ['sample__sampling_strategy', 'model__penalty'], log_x=True)
    #
    #  # 3. LRC - oversampled
    #  clf = 'LRC-oversampled'
    #  cvresults = estimator_dict[clf]['CV_results']
    #  make_hyperparameter_plots(cvresults, clf, 'model__C', ['sample__sampling_strategy', 'model__penalty'], log_x=True)
    #
    #  # 5. BRFC - unbalanced
    #  clf = 'BRFC-unbalanced'
    #  cvresults = estimator_dict[clf]['CV_results']
    #  make_hyperparameter_plots(cvresults, clf, 'model__n_estimators', ['model__sampling_strategy'], linstyles=['none', 'none', 'none'])
    #  make_hyperparameter_plots(cvresults, clf, 'model__max_depth', ['model__sampling_strategy'], linstyles=['none', 'none', 'none'])
    # # make_hyperparameter_plots(cvresults, clf, 'model__min_samples_split', ['model__sampling_strategy'], linstyles=['none', 'none', 'none'])
    #
    #  # 6. RFC - undersampled
    #  clf = 'RFC-undersampled'
    #  cvresults = estimator_dict[clf]['CV_results']
    #  make_hyperparameter_plots(cvresults, clf, 'model__n_estimators', ['sample__sampling_strategy'], linstyles=['none', 'none', 'none'])
    #  make_hyperparameter_plots(cvresults, clf, 'model__max_depth', ['sample__sampling_strategy'], linstyles=['none', 'none', 'none'])
    # # make_hyperparameter_plots(cvresults, clf, 'model__min_samples_split', ['sample__sampling_strategy'], linstyles=['none', 'none', 'none'])

def optimize_classifiers(data_dict, test_percent):

    # --- LOAD COMBINED DATA SPLITS ------
    with open(f'{dict_savepath}/data_splits_dictionary.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']

    # ---5-FOLD CV GRIDSEARCH---
    est_dict = define_estimators()

    for i in est_dict:

        run_grid_search(est_dict, i, X_train, y_train, X_test, CV=5, scorer='roc_auc', verbose=2)

        with open(f'{dict_savepath}/gridsearch_dictionary.pkl', 'wb') as f:
            pickle.dump(est_dict, f)

    return est_dict


if __name__ == '__main__':

    dict_folder = '70per_training_thermal'  # Folder for storing dictionaries


    global dataroot
    dataroot = '/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/'
    global dict_savepath
    dict_savepath = f'{dataroot}Dictionaries/Combined Datasets/{dict_folder}'


    optimize_classifiers()
    create_plots()


    # --- FINE-TUNE PARAMETERS ----
    # Faster hyperparameter plots:
    # min_sample_splits_vec = np.linspace(0.001, 0.4, 25)
    # max_depth_vec = np.linspace(5, 65, 11)
    # finetune_rfc(data_dict, estimator_dict, min_sample_splits_vec, max_depth_vec)




