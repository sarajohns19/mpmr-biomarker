#%%
# NOTE: if helpers are being a bitch, on remote machine, cd to /home/mirl/sjohnson/.pycharm_helpers/
# then enter: tar -xzf helpers.tar.gz.
# restart PyCharm
import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import os
import glob
import torch
import pickle
from pickle import dump, load
import numpy as np
import pandas as pd
from skimage import measure
#import CAMP.camp.FileIO as io
import camp.FileIO as io
import sklearn.metrics as metrics
from scipy.stats import normaltest, ttest_ind
from statannotations.Annotator import Annotator
import seaborn as sns
from skimage.measure import regionprops
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import savemat
#import CAMP.camp.StructuredGridOperators as so
import camp.StructuredGridOperators as so
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_validate

import biomarker_class
from biomarker_class import Predictor

import tools as tls
import matplotlib
matplotlib.use('module://backend_interagg')
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

from collections.abc import Iterable

def load_predictors():
    """Loads predictor class objects from instance folder"""

    print('Loading Predictor objects for all models...')
    instance_path = f'{dataroot}Instances/{instance_folder}'

    # load data output
    with open(f'{instance_path}/LRC-MP.pkl', 'rb') as f:
        lrc = pickle.load(f)
    with open(f'{instance_path}/RFC-MP.pkl', 'rb') as f:
        rfc = pickle.load(f)
    with open(f'{instance_path}/LRC.pkl', 'rb') as f:
        lrc_th = pickle.load(f)
    with open(f'{instance_path}/RFC.pkl', 'rb') as f:
        rfc_th = pickle.load(f)
    with open(f'{instance_path}/NPV.pkl', 'rb') as f:
        npv = pickle.load(f)
    with open(f'{instance_path}/NPV_blur.pkl', 'rb') as f:
        blur = pickle.load(f)
    with open(f'{instance_path}/CTD.pkl', 'rb') as f:
        ctd = pickle.load(f)
    with open(f'{instance_path}/CEM240.pkl', 'rb') as f:
        cem240 = pickle.load(f)

    print('done')
    return lrc, rfc, lrc_th, rfc_th, npv, blur, ctd, cem240

def saveplot(filename, format='png'):
    """Save plot to the Instance folder, optionally to Output folders"""

    plt.savefig(f'{dataroot}Instances/{instance_folder}/Results/{filename}.{format}',
                dpi=150, bbox_inches='tight', pad_inches=0)

    if save_to_output:
        plt.savefig(f'{dataroot}Output/{filename}.{format}',
                dpi=150, bbox_inches='tight', pad_inches=0)

    if save_to_figoutput:
        plt.savefig(f'{dataroot}Output/Paper_Figures/{filename}.{format}',
                    dpi=150, bbox_inches='tight', pad_inches=0)

def plot_ROC(ax, obj, pred_data, label_data, plot_ci=False, th_dict=None, plt=True):
    """Calculate AUC and plot single ROC for one predictor class object. If input data is iterable,
    then the mean ROC will be calculated."""

    # -- inputs ---
    # ax:         matplotlib axes object to plot on
    # obj:        Predictor class object who's data will be plot (for plot color)
    # pred_data:  binary prediction data
    # label_data: binary laebeled data
    # plot_ci:    Boolean, True to plot confidence intervals for multiple folds (default=False)
    # th_dict:    dictionary containing fields "values", "colors", "markers" for threshold markers (default=None)
    # plt:        Boolean, True to plot the ROC for the obj (default=True)
    # --- outputs ---
    # auc_mean:   Average value of interpolated AUCs
    # auc_std:    Standard deviation of interpolated AUCs
    # aucf:       1D array, AUC for each fold

    # get plot parameters from obj
    line_color = obj.plt_color

    if obj.plt_line is None:
        line_type = '-'
    else:
        line_type = obj.plt_line

    if obj.name == 'NPV_blur':
        legendstr = 'NPV'
    else:
        legendstr = obj.name

    tprf = []
    fprf = []
    tholdsf = []
    aucf = []

    # if there are multiple folds, create a list of fpr, tpr, tholds, and auc for each fold.
    if (type(pred_data) is list) or type(pred_data) is tuple:

        for fold in range(0, len(pred_data)):
            fpr, tpr, tholds = metrics.roc_curve(label_data[fold], pred_data[fold])
            tprf.append(tpr)
            fprf.append(fpr)
            tholdsf.append(tholds)
            aucf.append(metrics.auc(fpr, tpr))

        # calculate the mean and std of ROC for all folds
        # interpolate tpr and threshold values along a defined fpr vector
        fpr_mean = np.linspace(0, 1, 100)
        tprf_interp = []
        tholdsf_interp = []
        for fold in range(0, len(pred_data)):
            tpr_interp = np.interp(fpr_mean, fprf[fold], tprf[fold])
            tholds_interp = np.interp(fpr_mean, fprf[fold], tholdsf[fold])
            tpr_interp[0] = 0.0
            #tpr_interp[-1] = 1.0
            tholdsf_interp.append(tholds_interp)
            tprf_interp.append(tpr_interp)

        # calculate mean and standard-dev tpr
        tpr_mean = np.mean(tprf_interp, axis=0)
        tpr_std = 2 * np.std(tprf_interp, axis=0)
        tpr_mean[-1] = 1.0

        # calculate plot upper/lower bounds
        tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
        tpr_lower = tpr_mean - tpr_std

        # calculate mean and standard-dev of AUC
        auc_mean = np.mean(aucf)
        auc_std = np.std(aucf)

        # plot ROC curve
        tpr = tpr_mean
        fpr = fpr_mean

        if plt is True:

            if plot_ci is True:
                # plot mean AUC with confidence intervals
                leg = f'{legendstr} AUC:{auc_mean:.03f} +/- {auc_std:.03f}'
                ax.plot(fpr, tpr, color=line_color, linestyle=line_type, label=leg, linewidth=2)
                ax.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.2, color=line_color)

            else:
                # plot the mean AUC
                leg = f'{legendstr} AUC:{auc_mean:.03f}'
                ax.plot(fpr, tpr, color=line_color, linestyle=line_type, label=leg, linewidth=2)

        if th_dict is not None:
            # plot Markers for threshold values
            values = th_dict['values']
            colors = th_dict['colors']
            markers = th_dict['markers']

            for i in range(0, len(values)):
                tholds_mean = np.mean(tholdsf_interp, axis=0)
                th_fpr = fpr[np.argmin(np.abs(tholds_mean - values[i]))]
                th_tpr = tpr[np.argmin(np.abs(tholds_mean - values[i]))]
                ax.scatter(th_fpr, th_tpr, color=colors[i], marker=markers[i], s=70)

    else:
        # for a single fold of data, calculate ROC once
        fpr, tpr, tholds = metrics.roc_curve(label_data, pred_data)
        auc_mean = metrics.auc(fpr, tpr)
        auc_std = 0

        if plt is True:
            # plot ROC
            leg = f'{legendstr} AUC:{auc_mean:.03f}'
            ax.plot(fpr, tpr, color=line_color, linestyle=line_type, label=leg, linewidth=2)

        if th_dict is not None:
            # plot Markers for threshold values
            values = th_dict['values']
            colors = th_dict['colors']
            markers = th_dict['markers']

            for i in range(0, len(values)):
                th_tpr = tpr[np.argmin(np.abs(tholds - values[i]))]
                th_fpr = fpr[np.argmin(np.abs(tholds - values[i]))]
                ax.scatter(th_fpr, th_tpr, color=colors[i], marker=markers[i], s=70)

    return auc_mean, auc_std, aucf

# FUNCTIONS FOR COMBINED DATASET
def get_optimized_dice(models):
    """Calculate ROC thresholds that maximize Dice score on the training data"""
    # --- inputs ---
    # instance_folder:   string, folder location for training instance
    # models:            list of Predictor objects to optimize
    # --- outputs ---
    # results_df:        Dataframe with optimal thresholds for Predictors

    # set-up dataframe for storing results
    cols = [x.name for x in models[:4]] + \
           [item for sublist in [[x.name + ' train', x.name + ' test'] for x in models[-4:]] for item in sublist]
    results_df = pd.DataFrame(columns=cols,
                               index=['Train Threshold', 'AUC', 'Precision', 'Recall', 'Dice', 'Accuracy'])

    for prd in models:
        re_scale = False  # default, do not need to re-scale from [0, 1]
        if prd.name == 'CTD':
            re_scale = True  # CTD needs to be scaled to [0, 1] to do the optimal threshold calculation
            print('Setting rescale = True for CTD optimization...')

        if prd.name == 'NPV' or prd.name == 'CEM240':
            prd.trained_th = 0.5  # NPV and CEM240 data is already binary

        if not (prd.model == 'Binary'):
            # get training dataset prediction values, find optimum threshold
            pred_data, label_data = prd.get_pred_data_combined('train')
            prd.trained_th = tls.get_dice_thold_v2(pred_data, label_data, rescale=re_scale)

        # save the optimized threshold value to the results_df column for the Predictor
        col_pdr = [c for c in results_df.columns if prd.name == c.split(' ')[0]]
        results_df[col_pdr[0]]['Train Threshold'] = prd.trained_th
        print(f'Optimized dice threshold for {prd.name} is {str(prd.trained_th)}')

        # save the Predictor object with the new "trained_th" attribute
        with open(f'{dataroot}Instances/{instance_folder}/{prd.name}.pkl', 'wb') as f:
            pickle.dump(prd, f)

    return results_df

def plot_ROCs_combined_test(models, results_df, font, titlestr=None):
    """Wrapper function for plotting ROCs of the validation data from multiple Predictors."""
    # --- inputs ---
    # models:       list of Predictor objects to plot ROCs and/or trained threshold values
    # results_df:   Dataframe for saving the AUC scores to
    # font:         dictionary of matplotlib font parameters
    # --- outputs ---
    # results_df:   updated Dataframe with AUC scores

    fig, ax = plt.subplots(1)
    fig.set_figheight(6)

    for prd in models:
        plt_bool = True
        if prd.model == 'Binary':
            plt_bool = False

        if prd.name == 'NPV' or prd.name == 'CEM240':
            th_plt = None
        elif prd.name == 'NPV_blur':
            th_plt = {'values': [prd.trained_th, .5],
                      'colors': [prd.plt_color, 'k'],
                      'markers': ['o', 's']}
        elif prd.name == 'CTD':
            th_plt = {'values': [prd.trained_th, np.log(240)],
                      'colors': [prd.plt_color, 'k'],
                      'markers': ['o', '^']}
        else:
            th_plt = {'values': [prd.trained_th],
                      'colors': [prd.plt_color],
                      'markers': ['o']}

        # get test (validation) dataset predictions, calculate AUC and plot ROC
        y_pred, y_label = prd.get_pred_data_combined('test')
        auc_mean, _, _ = plot_ROC(ax, prd, y_pred, y_label, plot_ci=True, th_dict=th_plt, plt=plt_bool)

        # save AUC score for Predictor in results_df
        col_pdr = [c for c in results_df.columns if prd.name == c.split(' ')[0]]
        results_df[col_pdr[-1]]['AUC'] = auc_mean

    # plot parameters
    plt.gca().tick_params(labelsize=font['size'])
    ax.legend(prop={'size': font['size']})
    ax.legend(loc='lower right')
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(titlestr)

    saveplot('Combined_roc_fig', format='png')
    plt.show()

    return results_df

def calculate_stats_combined(models, dset, results_df):
    """Calculate Dice, Precision, Recall, Accuracy for the training or testing Combined dataset"""
    # --- inputs ---
    # models:      list of Predictor class objects to calculate stats of
    # dset:      "train" or "test" dataset of Combined data
    # results_df:  Dataframe for storing results
    # --- outputs ---
    # results_df:  Dataframe updated with calculated results

    if dset == 'train':
        ind = 0
    elif dset == 'test':
        ind = -1

    for prd in models:
        # get the prediction data from the test/train dataset
        y_pred, y_label = prd.get_pred_data_combined(dset)

        # apply threshold if not already binary (NPV and CEM240)
        if not (prd.model == 'Binary'):
            y_pred = y_pred >= prd.trained_th

        # calculate scores and save them to results_df
        prec, recall, dice, acc = tls.calc_prec_rec_dice(y_pred, y_label)

        col_pdr = [c for c in results_df.columns if prd.name == c.split(' ')[0]]
        results_df[col_pdr[ind]]['Precision'] = prec
        results_df[col_pdr[ind]]['Recall'] = recall
        results_df[col_pdr[ind]]['Dice'] = dice
        results_df[col_pdr[ind]]['Accuracy'] = acc

    return results_df

def combined_stats_barchart(models, results_df, titlestr=None):
    """Make bar-plot of validation dataset scores"""
    # --- inputs ---
    # models:       list of Predictor objects to plot
    # combined_df:  Dataframe with scores to plot
    # ** A bar hatching can be changed to striped from solid fill if the prd.plt_line != None

    model_names = [x.name for x in models]

    # reformat the  results_df columns for plotting
    df_plot = results_df.copy()
    col_drop = [x for x in df_plot.columns if 'train' in x]  # drop columns with training data scores
    rename_k = [x for x in df_plot.columns if 'test' in x]   # create dictionary to rename test-data columns with model names
    rename_v = [x.split(' ')[0] for x in df_plot.columns if 'test' in x]
    rename_dict = {k: v for (k, v) in zip(rename_k, rename_v)}
    df_plot = df_plot.drop(columns=col_drop)
    df_plot = df_plot.rename(columns=rename_dict)
    df_plot = df_plot.loc[:, model_names] # reorder columnns by model list order

    # pivot df
    df_plot = (df_plot.stack()
               .reset_index(name='score')  # rename new column of scores 'scores'
               .rename(columns={'level_1': 'model', 'Unnamed: 0': 'metric'}))  # rename new column 'model'

    # make bar-plot
    fig, ax = plt.subplots(1, figsize=(9, 5))
    bars = sns.barplot(ax=ax,
                       data=df_plot,
                       x="metric",
                       y="score",
                       hue="model",
                       palette=[x.plt_color for x in models],
                       saturation=.8)

    # if prd.plt_line is assigned, then change the hatching/fill of the bar
    n_stat = len(np.unique(df_plot['metric']))
    for i, barobj in enumerate(bars.patches):
        if models[np.floor(i/n_stat).astype('int')].plt_line is not None:
            barobj.set_hatch('/////')
            barobj.set_fill(None)
            barobj.set_edgecolor(models[np.floor(i/n_stat).astype('int')].plt_color)

    # plot parameters
    plt.legend(bbox_to_anchor=(1.02, 0.6), loc='upper left', borderaxespad=0)
    ax.set_title(titlestr)
    ax.set_xlabel('')
    ax.set_ylabel('')

    saveplot('Combined_stats_barchart', format='png')
    plt.show()

# FUNCTIONS FOR CV DATASET
def get_optimized_dice_cv(models):
    """Calculate the optimized ROC threshold on each CV fold of the dataset, by maximizign Dice"""
    # --- inputs ---
    # models:    list of Predictor class objects to process
    # --- outputs ---
    # results_df:   Dataframe containing threhsold for each fold for each Predictor

    # create the empty results dataframe
    cols = [x.name for x in models]
    results_df = pd.DataFrame(columns=cols)

    nfolds = len(models[0].cv_subject_test)

    for prd in models:

        if prd.name == 'NPV':
            th_series = 0.5*np.ones(nfolds)

        elif prd.name == 'CEM240':
            th_series = np.log(240)*np.ones(nfolds) # this value doesn't get used, it's just for reporting so set to 240

        else:
            if prd.name == 'CTD':
                re_scale = True
            else:
                re_scale = False

            print(f'Optimizing {prd.name} thresholds (Rescale={re_scale})')
            y_pred, y_label = prd.get_pred_data_cv('train')

            th_series = []
            for f in range(0, len(y_pred)):
                th = tls.get_dice_thold_v2(y_pred[f], y_label[f], rescale=re_scale)
                th_series.append(th)

        results_df[prd.name] = th_series

    return results_df

def plot_ROCs_cv_test(models, cv_thresh_df, font):
    """"Calculate and plot model ROCs in a separate plot for each subject, with CV error region"""
    # --- inputs ---
    # models         list of Predictor objects to plot
    # cv_thresh_df   Dataframe of optimal thresholds for each CV fold
    # --- outputs ---
    # results_df     Dataframe with AUC values

    # calculate mean threshold value over all CV folds
    cv_thresh_df = cv_thresh_df.mean(axis=0)

    # generate the results_df Dataframe
    cols = ['subject'] + [x.name for x in models]
    results_df = pd.DataFrame(columns=cols)

    # create a subplot for each subject
    for i in range(1, 5):
        fig, ax = plt.subplots(1)
        fig.set_figheight(6)

        # create another Dataframe for this subject only
        sub_results_df = pd.DataFrame(columns=results_df.columns)

        # plt_bool = whether to plot ROC for the model
        # th_plt = parameters for the threshold marker(s)
        for prd in models:
            plt_bool = True
            if prd.model == 'Binary':
                plt_bool = False

            if prd.name == 'NPV' or prd.name == 'CEM240':
                th_plt = None
            elif prd.name == 'NPV_blur':
                th_plt = {'values': [cv_thresh_df[prd.name], .5],
                          'colors': [prd.plt_color, 'k'],
                          'markers': ['o', 's']}
            elif prd.name == 'CTD':
                th_plt = {'values': [cv_thresh_df[prd.name], np.log(240)],
                          'colors': [prd.plt_color, 'k'],
                          'markers': ['o', '^']}
            else:
                th_plt = {'values': [cv_thresh_df[prd.name]],
                          'colors': [prd.plt_color],
                          'markers': ['o']}

            # get model prediction and labeled data
            y_pred, y_label = prd.get_pred_data_cv('test', subint=i)
            # plot the ROC and thresholds
            auc_mean, auc_std, aucf = plot_ROC(ax, prd, y_pred, y_label, plot_ci=True, th_dict=th_plt, plt=plt_bool)

            # save AUC scores to output Dataframe
            sub_results_df[prd.name] = aucf
            sub_results_df['subject'] = pd.Series(np.ones(len(y_pred))*i)

        results_df = pd.concat([results_df, sub_results_df], axis=0)

        # plot properties
        plt.gca().tick_params(labelsize=font['size'])
        ax.legend(prop={'size': font['size']})
        ax.legend(loc='lower right')
        plt.gca()
        ax.set_xlabel('1-Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(f'ROC Analysis: Subject {str(i)}')

        # save plot
        saveplot(f'subject{str(i)}_roc_fig', format='png')
        plt.show()

    return results_df

def plot_ROCs_cv_test_bymodel(models, cv_thresh_df, font):
    """"Calculate and plot subject ROCs in a separate plot for each model, with CV error region"""
    # --- inputs ---
    # models         list of Predictor objects to plot
    # cv_thresh_df   Dataframe of optimal thresholds for each CV fold
    # --- outputs ---
    # results_df     Dataframe with AUC values

    # calculate mean threshold value over all CV folds
    cv_thresh_df = cv_thresh_df.mean(axis=0)

    # plt_bool = whether to plot ROC for the model
    # th_plt = parameters for the threshold marker(s)
    for prd in models:
        plt_bool = True
        if prd.model == 'Binary':
            plt_bool = False

        if prd.name == 'NPV' or prd.name == 'CEM240':
            th_plt = None
        elif prd.name == 'NPV_blur':
            th_plt = {'values': [cv_thresh_df[prd.name], .5],
                      'colors': [prd.plt_color, 'k'],
                      'markers': ['o', 's']}
        elif prd.name == 'CTD':
            th_plt = {'values': [cv_thresh_df[prd.name], np.log(240)],
                      'colors': [prd.plt_color, 'k'],
                      'markers': ['o', '^']}
        else:
            th_plt = {'values': [cv_thresh_df[prd.name]],
                      'colors': [prd.plt_color],
                      'markers': ['o']}

        # create a plot for each model
        fig, ax = plt.subplots(1)
        fig.set_figheight(6)

        # plot ROC for each subject
        for i in range(1, 5):

            # get the model prediction and labeled data
            y_pred, y_label = prd.get_pred_data_cv('test', subint=i)
            # plot the ROC
            auc_mean, auc_std, aucf = plot_ROC(ax, prd, y_pred, y_label, plot_ci=True, th_dict=th_plt, plt=plt_bool)

        # plot parameters
        plt.gca().tick_params(labelsize=font['size'])
        plt.gca()
        ax.set_xlabel('1-Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(f'ROC Analysis:{prd.name}')

        # save plot
        saveplot(f'CV_{prd.name}_roc_fig', format='png')
        plt.show()

def calculate_stats_cv(models, dset, thresholds_df):
    """Calculate Dice, Precision, Recall, Accuracy for the training or testing dataset in each CV fold"""
    # --- inputs ---
    # models:      list of Predictor class objects to calculate stats of
    # dset:      "train" or "test" dataset of Combined data
    # results_df:  Dataframe for storing results
    # --- outputs ---
    # thresholds_df:  Dataframe updated with calculated results

    # create Dataframes for results
    cols = ['subject'] + [x.name for x in models]
    prec_df = pd.DataFrame(columns=cols)
    recall_df = pd.DataFrame(columns=cols)
    dice_df = pd.DataFrame(columns=cols)
    acc_df = pd.DataFrame(columns=cols)

    # create Dataframe for results in each subject, concat later
    for i in range(1, 5):
        sub_prec_df = pd.DataFrame(columns=prec_df.columns)
        sub_recall_df = pd.DataFrame(columns=recall_df.columns)
        sub_dice_df = pd.DataFrame(columns=dice_df.columns)
        sub_acc_df = pd.DataFrame(columns=acc_df.columns)

        for prd in models:
            # get model prediction and label
            y_pred, y_label = prd.get_pred_data_cv(dset, subint=i)

            # if Continuous model prediction, threshold the data using optimized threshold for each CV
            if not (prd.model == 'Binary'):
                th_series = thresholds_df[prd.name]

                for f in range(0, len(y_pred)):
                    y_pred[f] = y_pred[f] >= th_series[f]

            # calculate stats
            prec, recall, dice, acc = tls.calc_prec_rec_dice(y_pred, y_label)

            # store states and create a column for subject number in each Dataframe
            sub_prec_df[prd.name] = prec
            sub_prec_df['subject'] = np.ones(len(y_pred)) * i
            sub_recall_df[prd.name] = recall
            sub_recall_df['subject'] = np.ones(len(y_pred)) * i
            sub_dice_df[prd.name] = dice
            sub_dice_df['subject'] = np.ones(len(y_pred)) * i
            sub_acc_df[prd.name] = acc
            sub_acc_df['subject'] = np.ones(len(y_pred)) * i

        # concatenate stats for all subjects
        prec_df = pd.concat([prec_df, sub_prec_df], axis=0)
        recall_df = pd.concat([recall_df, sub_recall_df], axis=0)
        dice_df = pd.concat([dice_df, sub_dice_df], axis=0)
        acc_df = pd.concat([acc_df, sub_acc_df], axis=0)

    return prec_df.astype('float'), recall_df.astype('float'), dice_df.astype('float'), acc_df.astype('float')

def calculate_one_way_anova(models, stats_list, stat_ind):
    """Perform ANOVA for various stats, for significance across models using CV results"""
    # --- inputs ----
    # models     = list of Predictor objects to calculate between
    # stats_list = list of Dataframes with stats
    # stat_ind   = corresponding column number for each stat in the new output Dataframe
    # ---- outouts ----

    cols = [x.name for x in models]
    subs = range(1, 5)
    normal_check_df = pd.DataFrame(columns=cols, index=stat_ind)
    var_check_df = pd.DataFrame(columns=cols, index=stat_ind)
    anova_p_df = pd.DataFrame(columns=cols, index=stat_ind)

    for j in range(0, len(stats_list)):
        stat_df = stats_list[j]

        for prd in models:
            df = stat_df[['subject', prd.name]]
            df_pivot = df.set_index('subject')
            #df_pivot = df.pivot(columns='subject')

            # 1. Check for normal data:
            normal_flag = 1
            shapiro_p = []
            for i in subs:
                _, p = stats.shapiro(df_pivot[prd.name][i])
                shapiro_p.append(p)

            if (np.asarray(shapiro_p) <= 0.05).any():
                print(f'WARNING [{j}, {prd.name}]: Not all subject data is normal. Shapiro P-values: {shapiro_p}')
                normal_flag = 0

            # 2. Check for homogeneous variance:
            var_flag = 1
            _, p = stats.levene(df_pivot[prd.name][1], df_pivot[prd.name][2], df_pivot[prd.name][3],
                                df_pivot[prd.name][4])

            if p <= 0.05:
                print(
                    f'WARNING [{j}, {prd.name}]: Variance is significantly heterogeneous across groups. Levene P-value: {p}')
                var_flag = 0

            # 3. Run ANOVA
            _, anova_p = stats.f_oneway(df_pivot[prd.name][1], df_pivot[prd.name][2], df_pivot[prd.name][3],
                                        df_pivot[prd.name][4])

            if anova_p <= 0.05:
                print(f'WARNING [{j}, {prd.name}]: ANOVA P-value: {anova_p}')
            if anova_p <= 0.0001:
                anova_p = 0.0001

            normal_check_df[prd.name][j] = normal_flag
            var_check_df[prd.name][j] = var_flag
            anova_p_df[prd.name][j] = anova_p

    return anova_p_df, normal_check_df, var_check_df

# OTHER FUNCTIONS
def calculate_mda_combined(rabbit_dict, models):
    """Calcuate the subject-specific MDA of the Predictor, applied to the entire dataset from each subject"""
    # --- inputs ----
    # rabbit_dict:     dictionary containing rabbit volumes/segmentations
    # models:          list of Predictor class objects to calculated MDA
    # --- outputs ---
    # mda_df:          Dataframe with MDA+/-std as string
    # mda_mean:        Dataframe with mean MDA from each subject
    # mda_std:         Dataframe with standard deviation of MDA from each subject

    # create Dataframes for saving results
    mda_df = pd.DataFrame(index=[1, 2, 3, 4], columns=[x.name for x in models])
    mda_mean = pd.DataFrame(index=[1, 2, 3, 4], columns=[x.name for x in models])
    mda_std = pd.DataFrame(index=[1, 2, 3, 4], columns=[x.name for x in models])

    for prd in models:
        # for each subject
        for i, r in enumerate(rabbit_dict):
            # load the volumetric prediction in the quadricep ROI
            y_pred, ylabel = prd.get_recon_prediction_volume(rabbit_dict, r, roi='quad')

            # threshold the prediction at the combined training data threshold (prd.trained_th)
            th_opt = prd.trained_th
            y_pred_th = y_pred
            y_pred_th.data = (y_pred.data >= th_opt).float()

            # calculate the MDA and save to dataframes
            mda = tls.calc_mean_dist_agreement(y_pred_th, ylabel)
            mda_df[prd.name].iloc[i] = f'{mda[0]: .2f} +/-{mda[1]: .2f}'
            mda_mean[prd.name].iloc[i] = mda[0]
            mda_std[prd.name].iloc[i] = mda[1]

    # convert to float-type from tensor-type
    mda_mean = mda_mean.astype('float')
    mda_std = mda_std.astype('float')

    return mda_df, mda_mean, mda_std

def calculate_vol_diff(rabbit_dict, models):
    """Calculates the percent difference in volume between prediction and label"""
    # --- inputs ---
    # rabbit_dict:     dictionary containing rabbit volumes/segmentations
    # models:          list of Predictor class objects to calculated MDA
    # --- outputs ---
    # vol_diff         Dataframe of the values for each model and subject

    vol_diff = pd.DataFrame(index=[1, 2, 3, 4], columns=[x.name for x in models])

    for prd in models:
        # for each subject
        for i, r in enumerate(rabbit_dict):
            # load the volumetric prediction in the quadricep ROI
            y_pred, ylabel = prd.get_recon_prediction_volume(rabbit_dict, r, roi='quad')

            # threshold the prediction at the combined training data threshold (prd.trained_th)
            th_opt = prd.trained_th
            y_pred_th = y_pred
            y_pred_th.data = (y_pred.data >= th_opt).float()

            # calculate the percent difference in volume and save to dataframe
            predvol = torch.sum(y_pred_th.data)
            labelvol = torch.sum(ylabel.data)
            print('Subject ' + str(i) + prd.name + ' pred: ' + str(predvol) + ', label: ' + str(labelvol))
            per_vol = 100*(predvol-labelvol)/labelvol
            vol_diff[prd.name].iloc[i] = per_vol

    vol_diff = vol_diff.astype('float')
    return vol_diff

def stat_box_plots(models, stat_df, x_col, hue_col, titlestr, cpalette=None, plot_type='box', plt_p=True, plt_pout=False):

    # include only the models input by 'models'
    cols = ['subject'] + [x.name for x in models]
    df_plt = stat_df[cols].copy()
    df_plt['subject'] = df_plt['subject'].astype('int')

    # create color palette
    if cpalette is None:
        cpalette = [x.plt_color for x in models]

    # collapse all model columns into single 'model' column
    df_plt = (df_plt.set_index('subject')
                    .stack()
                    .reset_index(name='score')   # rename new column of scores 'scores'
                    .rename(columns={'level_1': 'model'}))  # rename new column 'model'

    if plot_type == 'box':
        plot_params = {
            'data': df_plt,
            'x': x_col,
            'y': 'score',
            'hue': hue_col,
            'palette': cpalette,
            'fliersize': 0,
            'saturation': .8,
            'linewidth': 1.5,
            'width': .6,
            'medianprops': dict(linewidth=0),
            'boxprops': dict(linewidth=0)
        }

        fig, ax = plt.subplots(1, figsize=(8, 4))
        sns.boxplot(ax=ax, **plot_params)
        plt.legend(bbox_to_anchor=(1.02, 0.6), loc='upper left', borderaxespad=0)
        #ax.set_xticklabels(str(np.unique(df_plt[x_col].unique()).astype('int')), fontsize=12)
        ax.set_title(titlestr, fontsize=14.5)
        ax.set_xlabel(x_col)

    if plot_type == 'bar':
        plot_params = {
            'data': df_plt,
            'x': x_col,
            'y': 'score',
            'hue': hue_col,
            'palette': cpalette,
            'errorbar': 'sd',
            'saturation': .8,
            'errwidth': 2,
            'capsize': 0.0,
            'width': .7
        }

        fig, ax = plt.subplots(1, figsize=(7.5, 5))
        bars = sns.barplot(ax=ax, **plot_params)
        n_sub = len(np.unique(df_plt['subject']))
        for i, barobj in enumerate(bars.patches):
            if models[np.floor(i /n_sub).astype('int')].plt_line is not None:
                barobj.set_hatch('/////')
                barobj.set_fill(None)
                barobj.set_edgecolor(models[np.floor(i / n_sub).astype('int')].plt_color)
        plt.legend(bbox_to_anchor=(1.02, 0.6), loc='upper left', borderaxespad=0)
        #ax.set_xticklabels(np.unique(df_plt[x_col].unique()).astype('int')) #, fontsize=12)
        # ax.set_xticklabels(np.unique(df_plt[x_col].unique()).astype('int'))
        ax.set_title(titlestr) #, fontsize=14.5)
        ax.set_xlabel(x_col)
        ax.set_ylabel('')


    pairs = []
    pvals = []
    if plt_p is True:
        hues = np.unique(df_plt[hue_col])
        cats = np.unique(df_plt[x_col])
        for cat in cats:
            for j in range(len(hues)):
                df_cat = df_plt.loc[df_plt[x_col] == cat]
                for k in range(j + 1, len(hues)):
                    d1 = df_cat['score'].loc[df_cat[hue_col] == hues[j]]
                    d2 = df_cat['score'].loc[df_cat[hue_col] == hues[k]]
                    st, p = ttest_ind(d1, d2, nan_policy='omit')
                    if p <= 0.05:
                        pairs.append([(cat, hues[j]), (cat, hues[k])])
                        pvals.append(p)
                    else:
                        print('N.S.: group = ' + str(cat) + ', hues = ' + str(hues[j]) + ' & ' + str(hues[k]))

        if len(pairs) > 0:
            annotator = Annotator(ax, pairs, **plot_params)
            annotator.configure(line_offset=-5,
                                text_offset=-6,
                                pvalue_thresholds=[(1, 'ns'), (0.05, '*')],
                                verbose=0)
            annotator.set_pvalues(pvals)
            annotator.annotate()

    if plt_pout is True:
        hues = ['LRC', 'RFC']
        cats = np.unique(df_plt[x_col])
        for hue in hues:
            for j in range(len(cats)):
                df_hue = df_plt.loc[df_plt[hue_col] == hue]
                for k in range(j+1, len(cats)):
                    d1 = df_hue['score'].loc[df_hue[x_col] == cats[j]]
                    d2 = df_hue['score'].loc[df_hue[x_col] == cats[k]]
                    st, p = ttest_ind(d1, d2, nan_policy='omit')
                    if p <= 0.05:
                        pairs.append([(cats[j], hue), (cats[k], hue)])
                        pvals.append(p)

        if len(pairs) > 0:
            annotator = Annotator(ax, pairs, **plot_params)
            annotator.configure(line_offset=-5,
                                text_offset=-6,
                             #   pvalue_thresholds=[(1, 'ns'), (0.05, '*')],
                                verbose=0)
            annotator.set_pvalues(pvals)
            annotator.annotate()

        #ax.set_ylim(df_plt['score'].min() - .05, 1.05)

    # plt.savefig(f'{dataroot}Instances/{instance_folder}/Results/bw_{titlestr}_fig.png',
    #             dpi=150, bbox_inches = 'tight', pad_inches = 0)
    #
    # if save_to_output:
    #     plt.savefig(f'{dataroot}Output/bw_{titlestr}_fig.png',
    #                 dpi=150, bbox_inches='tight', pad_inches=0)

    plt.savefig(f'{dataroot}Instances/{instance_folder}/Results/CV_{titlestr}_barchart.png',
                dpi=150, bbox_inches='tight', pad_inches=0)

    if save_to_output:
        plt.savefig(f'{dataroot}Output/CV_{titlestr}_barchart.png',
                    dpi=150, bbox_inches='tight', pad_inches=0)
    if save_to_figoutput:
        plt.savefig(f'{dataroot}Output/Paper_Figures/CV_{titlestr}_barchart.png',
                    dpi=150, bbox_inches='tight', pad_inches=0)
    plt.show()
    return pairs, pvals

def mda_bar_plot(models, df_mean, df_std, x_col, hue_col, titlestr):

    font = {'family': 'sans-serif',
            'serif': 'Helvetica Neue',
            'size': 16}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'monospace'

    cpalette = [x.plt_color for x in models]
    cols = [x.name for x in models]
    df_mean = df_mean[cols].copy()
    df_std = df_std[cols].copy()

    df_mean['subject'] = df_mean.index
    # collapse all model columns into single 'model' column
    df_mean = (df_mean.set_index('subject')
                    .stack()
                    .reset_index(name='score')   # rename new column of scores 'scores'
                    .rename(columns={'level_1': 'model'}))  # rename new column 'model'
    df_std['subject'] = df_std.index
    df_std = (df_std.set_index('subject')
                    .stack()
                    .reset_index(name='score')   # rename new column of scores 'scores'
                    .rename(columns={'level_1': 'model'}))  # rename new column 'model'

    #combined df_mean and df_std
    df_mean['std'] = df_std['score']/2

    # include only the models input by 'models'


    plot_params = {
        'data': df_mean,
        'x': x_col,
        'y': 'score',
        'hue': hue_col,
        'yerr': df_mean['std'],
        'capsize': 0.0,
        'width': 1.2
    }

    cat = "subject"
    subcat = "model"
    val = "score"
    err = 'std'
    df = df_mean

    import colorsys
    import matplotlib.colors as mc
    sat = .9
    cpalette2 = []
    for c in cpalette:
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        cpalette2.append(colorsys.hls_to_rgb(c[0], max(0, min(1, sat * c[1])), c[2]))

    fig, ax = plt.subplots(1, figsize=(7.5, 5))
    # call the function with df from the question
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x + offsets[i], dfg[val].values, width=width,
                label=gr, yerr=dfg[err].values, capsize=0.2, color=cpalette2[i])
    plt.legend(bbox_to_anchor=(1.02, 0.6), loc='upper left', borderaxespad=0)
    plt.xlabel(cat)
    plt.ylabel('[mm]')
    plt.xticks(x, u)
    plt.ylim(0, 8.3)
    plt.title(titlestr)

    plt.savefig(f'{dataroot}Instances/{instance_folder}/Results/MDA_barchart.png',
                dpi=150, bbox_inches='tight', pad_inches=0)

    if save_to_output:
        plt.savefig(f'{dataroot}Output/MDA_barchart.png',
                    dpi=150, bbox_inches='tight', pad_inches=0)
    if save_to_figoutput:
        plt.savefig(f'{dataroot}Output/Paper_Figures/MDA_barchart.png',
                    dpi=150, bbox_inches='tight', pad_inches=0)

    plt.show()

#import importlib
#importlib.reload(tls)

def evaluate_models_combined(instance_folder):
    # matplotlib.rc('font', family='sans-serif')
    # matplotlib.rc('font', serif='Helvetica Neue')
    # matplotlib.rc('text', usetex='false')
    # matplotlib.rcParams.update({'font.size': 15})
    font = {'family': 'sans-serif',
            'serif': 'Helvetica Neue',
            'size': 16}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'monospace'

    # Load the Predictor objects for each model
    lrc, rfc, lrc_th, rfc_th, npv, blur, ctd, cem240 = load_predictors()

    # Set-up dataframe for storing results of combined dataset
    models = [npv, blur, cem240, ctd, lrc_th, rfc_th, lrc, rfc]
    for prd in models:
        prd.plt_line = None


    ### ---COMBINED DATA-SET ANALYSIS------
    #1. Get dice-optimized threshold for each predictor from combined TRAIN dataset
    combined_df = get_optimized_dice(instance_folder, models)

    # 2. Plot ROC of combined TEST data with the optimized threshold
    fig, axs = plt.subplots(1)
    fig.set_figheight(6)
    #blur.plt_color = matplotlib._cm._tab10_data[0]
    ctd.plt_color = matplotlib._cm._tab10_data[3]
    rfc_th.plt_line = '-.'
    lrc_th.plt_line = '-.'
    models = [lrc, rfc, lrc_th, rfc_th, blur, ctd, npv, cem240]
    combined_df = plot_ROCs_combined_test(models, combined_df, font)

    # 3. Calculate stats of TRAIN data with the optimized threshold for classifiers
    models = [lrc, rfc, lrc_th, rfc_th]
    dset = 'train'
    combined_df = calculate_stats_combined(models, dset, combined_df)

    # 4. Calculate stats of TEST data with the optimized threshold
    models = [npv, blur, cem240, ctd, lrc, rfc, lrc_th, rfc_th]
    dset = 'test'
    combined_df = calculate_stats_combined(models, dset, combined_df)

    # 5. Save the combined_df as CSV
    combined_df = combined_df.astype('float')
    combined_df.to_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_results.csv')
    if save_to_output:
        combined_df.to_csv(f'{dataroot}Output/combined_results.csv')

    # 6. Plot the performance stats of combined dataset
    models = [npv, cem240, ctd, lrc, rfc]
    cem240.plt_color = matplotlib._cm._tab10_data[3]
    ctd.plt_color = matplotlib._cm._Paired_data[4]
    #combined_stats_barchart(models, combined_df)

    # 7. Calculate MDA of whole QUADRICEP ROI with the optimized threshold
    with open(f'{dataroot}Instances/{instance_folder}/parameter_dict.pkl', 'rb') as f:
        instance_params = pickle.load(f)
    with open(instance_params['rabbit_dict_file'], 'rb') as f:
        rabbit_dict = pickle.load(f)

    models = [npv, blur, ctd, cem240, lrc, rfc, lrc_th, rfc_th]

    mda_df, mda_mean, mda_std = calculate_mda_combined(rabbit_dict, models)
    mda_df.to_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_mda.csv')
    mda_mean.to_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_mda_mean.csv')
    mda_std.to_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_mda_std.csv')
    if save_to_output:
        mda_df.to_csv(f'{dataroot}Output/combined_mda.csv')
        mda_mean.to_csv(f'{dataroot}Output/combined_mda_mean.csv')
        mda_std.to_csv(f'{dataroot}Output/combined_mda_std.csv')

    # 8. Calculate Vol % difference of whole QUADRICEP ROI with optimized threshold
    with open(f'{dataroot}Instances/{instance_folder}/parameter_dict.pkl', 'rb') as f:
        instance_params = pickle.load(f)
    with open(instance_params['rabbit_dict_file'], 'rb') as f:
        rabbit_dict = pickle.load(f)

    vol_df = calculate_vol_diff(rabbit_dict, models)
    vol_df.to_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_vol_diff.csv')
    if save_to_output:
         vol_df.to_csv(f'{dataroot}Output/combined_vol_diff.csv')

def evaluate_models_cv(instance_folder):

    font = {'family': 'sans-serif',
            'serif': 'Helvetica Neue',
            'size': 16}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'monospace'

    # Load the Predictor objects for each model
    lrc, rfc, lrc_th, rfc_th, npv, blur, ctd, cem240 = load_predictors()

    instance_path = f'{dataroot}Instances/{instance_folder}/Results/'

    if not os.path.exists(f'{instance_path}/cv_thresholds_by_fold.csv'):
        ### ---SUBJECT/CV DATA-SET ANALYSIS----
        # 1. for each CV, calculate the optimal thresholds
        models = [npv, blur, cem240, ctd, lrc, rfc, lrc_th, rfc_th]
        cv_thresholds_df = get_optimized_dice_cv(models)
        cv_thresholds_df.to_csv(f'{instance_path}/cv_thresholds_by_fold.csv')

    else:
        cv_thresholds_df = pd.read_csv(f'{instance_path}/cv_thresholds_by_fold.csv')

    # 2. plot the ROC curves from CV TEST data for each subject
    models = [npv, blur, cem240, ctd, lrc, rfc, lrc_th, rfc_th]
    for prd in models:
        prd.plt_line = None
    rfc_th.plt_line = '-.'
    lrc_th.plt_line = '-.'
    cv_auc_df = plot_ROCs_cv_test(models, cv_thresholds_df, font)
    cv_auc_df.to_csv(f'{instance_path}/cv_auc_by_subject.csv')

    # 3. Calculate stats of TEST data using optimal threshold from each CV-fold
    models = [npv, blur, cem240, ctd, lrc, rfc, lrc_th, rfc_th]
    cv_prec_df, cv_recall_df, cv_dice_df,cv_acc_df = calculate_stats_cv(models, 'test', cv_thresholds_df)
    cv_prec_df.to_csv(f'{instance_path}/cv_prec_by_subject.csv')
    cv_recall_df.to_csv(f'{instance_path}/cv_recall_by_subject.csv')
    cv_dice_df.to_csv(f'{instance_path}/cv_dice_by_subject.csv')
    cv_acc_df.to_csv(f'{instance_path}/cv_acc_by_subject.csv')


        # # read in stats
        # cv_auc_df = pd.read_csv(f'{instance_path}/cv_auc_by_subject.csv')
        # cv_prec_df = pd.read_csv(f'{instance_path}/cv_prec_by_subject.csv')
        # cv_recall_df = pd.read_csv(f'{instance_path}/cv_recall_by_subject.csv')
        # cv_dice_df = pd.read_csv(f'{instance_path}/cv_dice_by_subject.csv')
        # cv_acc_df = pd.read_csv(f'{instance_path}/cv_acc_by_subject.csv')
        # cv_thresholds_df = pd.read_csv(f'{instance_path}/cv_thresholds_by_fold.csv')


    # 4. Calculate ANOVA between subject
    df_list = [cv_auc_df, cv_prec_df, cv_recall_df, cv_dice_df, cv_acc_df]
    df_inds = ['AUC', 'Precision','Recall','Dice', 'Accuracy']
    models = [npv, blur, cem240, ctd, lrc, rfc]

    anovas_df, normal_df, var_df = calculate_one_way_anova(models, df_list, df_inds)

    # 2. Box and whiskers /barcharst plots

    models = [npv, cem240, ctd, lrc_th,  lrc, rfc_th, rfc ]
    cem240.plt_color = matplotlib._cm._tab10_data[3]
    ctd.plt_color = matplotlib._cm._Paired_data[4]
    hue_col = 'model'
    x_col = 'subject'

    df = cv_dice_df
    pairs, pvals = stat_box_plots(models, df, x_col, hue_col, 'Dice', plot_type='bar', plt_p=False, plt_pout=False)

    df = cv_prec_df
    pairs, pvals = stat_box_plots(models, df, x_col, hue_col, 'Precision',  plot_type='bar', plt_p=False, plt_pout=False)

    df = cv_recall_df
    pairs, pvals = stat_box_plots(models, df, x_col, hue_col, 'Recall',  plot_type='bar', plt_p=False, plt_pout=False)


    # 3. ROCs by model in each subject
    models = [npv, blur, cem240, ctd, lrc, rfc]
    ctd.plt_color = matplotlib._cm._tab10_data[3]
    plot_ROCs_cv_test_bymodel(models, cv_thresholds_df, font)

    return anovas_df

def make_slice_contours(instance_folder, rabbit_slices, save_fig=False):
    with open(f'{dataroot}Instances/{instance_folder}/parameter_dict.pkl', 'rb') as f:
        instance_params = pickle.load(f)
    with open(instance_params['rabbit_dict_file'], 'rb') as f:
        rabbit_dict = pickle.load(f)

    # Set some things for the new figure
    # create pink colormap for histology label
    colors = [(0.0, 0.0, 0.0), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]
    cm = LinearSegmentedColormap.from_list('hist_color', colors, N=1)
    contour_width = 1.0
    contour_prune = 9
    ycent = 60
    xcent = 60

    if rabbit_slices is None:
        all_slices = True
        rabbit_slices = []
    else:
        all_slices = False

    lrc, rfc, npv, blur, ctd, cem240 = load_predictors()
    models = [npv, ctd, rfc]
    cem240.plt_color = matplotlib._cm._tab10_data[3]
    ctd.plt_color = matplotlib._cm._Paired_data[3] #4

    for i, r in enumerate(rabbit_dict):

        font = {'family': 'calibri',
                'size': 12}
        matplotlib.rc('font', **font)
        plt.rcParams['font.family'] = 'monospace'

        # load the background T1w image
        if 'acute_t1' not in rabbit_dict[r]:
            data_dir = dataroot + '/Data/'
            t1_volume = io.LoadITKFile(f'{data_dir}{r}/{r}_t1_w_post_file.nii.gz')
            acute_t1 = so.ResampleWorld.Create(rabbit_dict[r]['quad_mask'])(t1_volume)
            rabbit_dict[r]['acute_t1'] = acute_t1.clone()

        # load histology label volume for rabbit
        _, hist_label_vol = npv.get_recon_prediction_volume(rabbit_dict, r, roi='quad')

        # load binary prediction volumes for all models for rabbit
        for prd in models:
            pred_vol, _ = prd.get_recon_prediction_volume(rabbit_dict, r, roi='quad')
            pred_vol.data = pred_vol.data >= prd.trained_th
            setattr(prd, 'pred_vol', pred_vol)
            del pred_vol

        # get list of slices to plot if not input by users
        if all_slices:
            # rabbit_slices.append(list(range(0, rabbit_dict[r]['therm_histology'].shape()[2])))
            rabbit_slices.append(list(range(0, rabbit_dict[r]['quad_mask'].shape()[2])))

        for s in (rabbit_slices[i]):

            print(f'Processing {r} slice {s}:')
            plt.figure()

            # create mask of quadriceps (eval ROI) for plotting
            # mask = rabbit_dict[r]['acute_eval_mask'][0, :, s, :]
            mask = rabbit_dict[r]['quad_mask'][0, :, s, :]

            try:
                properties = regionprops(np.asarray(mask).astype(int))
                ycent, xcent = properties[0].centroid
            except:
                print(f'no COM found in slice {s}')
                ycent, xcent = ycent, xcent

            ymin = ycent + 30
            ymax = ycent - 30
            xmin = xcent - 30
            xmax = xcent + 30

            # rows = np.any(mask.numpy(), axis=1)
            # cols = np.any(mask.numpy(), axis=0)
            # ymin, ymax = np.where(rows)[0][[0, -1]]
            # xmin, xmax = np.where(cols)[0][[0, -1]]
            # ymin -= 10
            # ymax += 10
            # xmin -= 10
            # xmax += 10

            # plot T1w background image
            plt.imshow(rabbit_dict[r]['acute_t1'][0, :, s, :], cmap='gray')
            plt.axis('off')

            # create masked ROI overlay of histology label (in pink)
            hist_label = hist_label_vol[0, :, s, :]
            masked = np.ma.masked_where(hist_label.data.squeeze() == 0, hist_label.data.squeeze())
            plt.imshow(masked, cmap=cm, alpha=0.6)
            # hist_label = rabbit_dict[r]['therm_histology'][0, :, s, :]

            # create contours for each model
            for prd in models:
                seg_slice = prd.pred_vol[0, :, s, :].clone()
                seg_cont = measure.find_contours(seg_slice.cpu().numpy(), 0.5)
                try:
                    for contour in seg_cont:
                        # Prune some of the smaller contours
                        if len(contour) <= contour_prune:
                            continue

                        plt.plot(contour[:, 1], contour[:, 0], color=prd.plt_color, linewidth=contour_width)
                except IndexError:
                    print('No NPV Prediction, ', end='')

            # plt.ylim(ymin, ymax)
            # plt.xlim(xmin, xmax)
            ax = plt.gca()
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(xmin, xmax)
            plt.title(f'{r}_slice_{s}')

            if save_fig:
                plt.savefig(f'{dataroot}Instances/{instance_folder}/Results/{r}_slice_{s}_2Dseg_v3.png',
                            dpi=600, bbox_inches='tight', pad_inches=0)
                if save_to_output:
                    plt.savefig(f'{dataroot}Output/{r}_slice_{s}_2Dseg_v3.png',
                                dpi=600, bbox_inches='tight', pad_inches=0)
            else:
                plt.show()

def save_to_mat(instance_folder):

    with open(f'{dataroot}Instances/{instance_folder}/parameter_dict.pkl', 'rb') as f:
        instance_params = pickle.load(f)
    with open(instance_params['rabbit_dict_file'], 'rb') as f:
        rabbit_dict = pickle.load(f)

    lrc, rfc, lrc_th, rfc_th, npv, blur, ctd, cem240 = load_predictors()
    lrc.name = 'LRC'
    rfc.name = 'RFC'
    models = [npv, ctd, rfc, lrc, blur, cem240]

    for r in rabbit_dict:
        _, hist_label_vol = npv.get_recon_prediction_volume(rabbit_dict, r, roi='quad')

        matdict = {'hist_label': np.squeeze(np.asarray(hist_label_vol.data).astype(int))}
        # load binary prediction volumes for all models for rabbit
        for prd in models:
            pred_vol, _ = prd.get_recon_prediction_volume(rabbit_dict, r, roi='quad')
            pred_vol.data = pred_vol.data >= prd.trained_th
            matdict[prd.name] = np.squeeze(np.asarray(pred_vol.data).astype(int))

        savemat(f'{dataroot}Instances/{instance_folder}/Results/{r}_pred_volumes.mat', matdict)
        if save_to_output:
            savemat(f'{dataroot}/Output/{r}_pred_volumes.mat', matdict)

import colorcet as cc
def make_param_maps(rabbit):
    """Create the 2D parameter maps from one rabbit example"""

    # load the tissue masks and histology label
    quad_mask = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_thermal_tissue.nrrd')
    hist_label = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_hist_label.nii.gz')
    hist_label = so.ResampleWorld.Create(quad_mask)(hist_label)

    # load the parameter volumes
    log_ctd = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_log_ctd_map.nii.gz')
    thermal_vol = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_thermal_vol.nii.gz')
    post_t2 = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_t2_w_post_file.nii.gz')
    pre_t2 = io.LoadITKFile(f'{dataroot}Data/{rabbit}/deform_files/{rabbit}_t2_w_pre_def.nii.gz')
    post_adc = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_adc_post_file.nii.gz')
    pre_adc = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_adc_pre_file.nii.gz')

    # calculate difference in T2w
    post_t2 = so.ResampleWorld.Create(quad_mask)(post_t2)
    pre_t2 = so.ResampleWorld.Create(quad_mask)(pre_t2)
    t2_diff = (post_t2.data.squeeze() - pre_t2.data.squeeze())

    # calculate difference in ADC
    post_adc = so.ResampleWorld.Create(quad_mask)(post_adc)
    pre_adc = so.ResampleWorld.Create(quad_mask)(pre_adc)
    adc_diff = (post_adc.data.squeeze() - pre_adc.data.squeeze())

    # calculate maximum temp projection
    thermal_vol.data[torch.isnan(thermal_vol.data)] = 0.0
    max_vec = thermal_vol.data.max(dim=0)[0]

    # squeeze the binary masks for contour plot
    quad_mask = quad_mask.data.squeeze()
    hist_label = hist_label.data.squeeze()
    log_ctd = log_ctd.data.squeeze()

    #image rotation angle and axis limits (current for 18-062)
    rot = -155  #155
    yl = [83, 53] #[98, 65]
    xl = [25, 89] #[15, 100]
    slices = [8]

    #generate the parameter maps
    bg_map = log_ctd
    clim = [-2, 10]
    cmap = matplotlib.colormaps['Spectral_r']
    for s in slices:
        param_slice(bg_map, s, cmap, clim, hist_label, quad_mask, rot, yl, xl, 'Param2D_ctd2')

    bg_map = max_vec
    clim = [36, 65]
    cmap = matplotlib.colormaps['Spectral_r']
    for s in slices:
        param_slice(bg_map, s, cmap, clim, hist_label, quad_mask, rot, yl, xl, 'Param2D_max2')

    bg_map = t2_diff
    clim = [-250, 250]
    cmap = matplotlib.colormaps['PuOr_r']
    for s in slices:
        param_slice(bg_map, s, cmap, clim, hist_label, quad_mask, rot, yl, xl, 'Param2D_t2w2')

    bg_map = adc_diff
    clim = [-2500, 2500]
    cmap = matplotlib.colormaps['PuOr_r']
    for s in slices:
        param_slice(bg_map, s, cmap, clim, hist_label, quad_mask, rot, yl, xl, 'Param2D_adc2')


def param_slice(bg_map, slice, cmap, clim, hist_bin, quad_bin, rot, ylims, xlims, savetitle):
    """Create single 2D parameter plot"""

    # define the contour colors
    hist_color = 'k' #'(0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
    quad_color = matplotlib._cm._tab10_data[2]

    # rotate the 2D slice
    bg_slice = bg_map.data[:, slice, :].squeeze()
    bg_slice = ndimage.rotate(bg_slice, rot, reshape=False, mode='grid-wrap')

    # plot the 2D slice
    plt.figure()
    im = plt.imshow(bg_slice, cmap=cmap)
    plt.axis('off')

    # rotate and plot HISTOLOGY label contour
    hist_slice = hist_bin.data[:, slice, :].clone()
    hist_slice = ndimage.rotate(hist_slice, rot, reshape = False)
    seg_cont = measure.find_contours(hist_slice, 0.5)
    try:
        for contour in seg_cont:
            plt.plot(contour[:, 1], contour[:, 0], color=hist_color, linewidth=2.5)
    except IndexError:
        print('No NPV Prediction, ', end='')

    # rotate and plot QUAD label contour
    quad_slice = quad_bin.data[:, slice, :].clone()
    quad_slice = ndimage.rotate(quad_slice, rot, reshape=False)
    seg_cont = measure.find_contours(quad_slice, 0.5)
    try:
        for contour in seg_cont:
            plt.plot(contour[:, 1], contour[:, 0], color=quad_color, linewidth=2.5)
    except IndexError:
        print('No NPV Prediction, ', end='')

    # plot parameters
    plt.clim(clim[0], clim[1])   # colorbar axis limit
    #plt.title(f'slice {slice}')
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])  # adjust x- and y-limits
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    ax = plt.gca()
    divider = make_axes_locatable(ax) # make colorbar same height as axis
    cax = divider.append_axes('right', size='2%', pad=0.03)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=13)

    if savetitle is not None:
        saveplot(savetitle, format='png')

    plt.show()


def make_paper_figures(instance_folder):
    font = {'family': 'sans-serif',
            'serif': 'Helvetica Neue',
            'size': 16}
    matplotlib.rc('font', **font)
    plt.rcParams['font.family'] = 'monospace'

    lrc, rfc, lrc_th, rfc_th, npv, blur, ctd, cem240 = load_predictors()

    # ---- COMBINED DATASET -----
    combined_df = pd.read_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_results.csv')
    combined_df = combined_df.set_index('Unnamed: 0')
    roc_models = [lrc, rfc, blur, ctd, npv, cem240]
    per_models = [npv, cem240, ctd, lrc, rfc]

    # 1. Plot ROC of combined TEST data with the optimized threshold
    ctd.plt_color = matplotlib._cm._tab10_data[3]
    cem240.plt_color = 'black'
    rfc_th.plt_line = '-.'
    lrc_th.plt_line = '-.'
    combined_df2 = plot_ROCs_combined_test(roc_models, combined_df, font, titlestr=' ')

    # 2. Plot the performance stats of combined dataset
    combined_df2 = combined_df.drop(index=['Train Threshold', 'AUC', 'Accuracy'])
    cem240.plt_color = matplotlib._cm._tab10_data[3]
    ctd.plt_color = matplotlib._cm._Paired_data[4]
    combined_stats_barchart(per_models, combined_df2, titlestr=' ')

    # ---- CV DATASET -------
    # # read in stats
    instance_path = f'{dataroot}Instances/{instance_folder}/Results/'
    cv_auc_df = pd.read_csv(f'{instance_path}/cv_auc_by_subject.csv')
    cv_prec_df = pd.read_csv(f'{instance_path}/cv_prec_by_subject.csv')
    cv_recall_df = pd.read_csv(f'{instance_path}/cv_recall_by_subject.csv')
    cv_dice_df = pd.read_csv(f'{instance_path}/cv_dice_by_subject.csv')
    cv_acc_df = pd.read_csv(f'{instance_path}/cv_acc_by_subject.csv')
    cv_thresholds_df = pd.read_csv(f'{instance_path}/cv_thresholds_by_fold.csv')

    cv_models = [npv, cem240, lrc, rfc]

    # 3. Plot Dice scores by subject
    hue_col = 'model'
    x_col = 'subject'
    # palette = matplotlib._cm._tab20b_data[0:4]
    # palette = matplotlib._cm._tab20c_data[12:16]
    # palette = sns.color_palette('rocket')
    palette = sns.cubehelix_palette(5, start=.4, rot=-.450, dark=.85, light=.15, hue=1, gamma=.85)
    pairs, pvals = stat_box_plots(per_models, cv_dice_df, hue_col, x_col, ' ', cpalette=palette,
                                  plot_type='bar', plt_p=False, plt_pout=False)

    # 4. AUC curves by subject
    # ctd.plt_color = matplotlib._cm._tab10_data[3]
    # plot_ROCs_cv_test_bymodel(roc_models, cv_thresholds_df, font)

    # --- MDA BARPLOT -----
    mda_mean = pd.read_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_mda_mean.csv')
    mda_std = pd.read_csv(f'{dataroot}Instances/{instance_folder}/Results/combined_mda_std.csv')

    cem240.plt_color = matplotlib._cm._tab10_data[3]
    ctd.plt_color = matplotlib._cm._Paired_data[4]
    # models2 = [npv, cem240, ctd, lrc, rfc]
    mda_bar_plot(per_models, mda_mean, mda_std, 'subject', 'model', ' ')

    # --- PARAMETER MAPS ----
    make_param_maps('18_062')



#%%

if __name__ == '__main__':

    global dataroot
    dataroot = '/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/'
    global save_to_output
    global save_to_figoutput
    global instance_folder

    instance_folder = '70per_10Kfolds_cor'  # location of data to load
    save_to_output = True
    save_to_figoutput = True

    if not os.path.exists(f'{dataroot}Instances/{instance_folder}/Results/'):
        os.mkdir(f'{dataroot}Instances/{instance_folder}/Results/')

    #evaluate_models_combined(instance_folder)
    evaluate_models_cv(instance_folder)
    #make_paper_figures(instance_folder)
    #save_to_mat(instance_folder)


    # save_to_mat(instance_folder)
    # wanted_keys = ['18_062']  # The keys you want
    # rabbit_dict2 = dict((k, rabbit_dict[k]) for k in wanted_keys if k in rabbit_dict)
    # slices = [[13], [8], [6], [6]]
    # make_slice_contours(instance_folder, slices, save_fig=True)

    make_param_maps('18_062')








