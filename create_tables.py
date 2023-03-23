
import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import os
import glob
import torch
import pickle
from pickle import dump, load
import numpy as np
import pandas as pd

def cv_to_range(cv_df, index_name, out_df, fmt= '.2f'):

    col_keep = out_df.columns
    if len(cv_df) > 4:
        sub_df = cv_df.groupby(by='subject').mean().drop('Unnamed: 0', axis=1)
    else:
        sub_df = cv_df

    sub_df = sub_df[col_keep]
    for col in col_keep:
        out_df[col][index_name] = f'({sub_df[col].min(): {fmt}} - {sub_df[col].max(): {fmt}})'

    return out_df

def cv_to_average(cv_df, index_name, out_df, fmt= '.2f'):

    col_keep = out_df.columns
    if len(cv_df) > 4:
        sub_df = cv_df.groupby(by='subject').mean().drop('Unnamed: 0', axis=1)
    else:
        sub_df = cv_df

    sub_df = sub_df[col_keep]
    for col in col_keep:
        out_df[col][index_name] = f'{sub_df[col].mean(): {fmt}} +/- {sub_df[col].std(): {fmt}}'

    return out_df

if __name__ == '__main__':

    global dataroot
    dataroot = '/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/'
    global save_to_output
    global save_to_figoutput
    global instance_folder

    instance_folder = '70per_10Kfolds_cor'  # location of data to load
    save_to_output = True

    instance_path = f'{dataroot}Instances/{instance_folder}/Results/'
    # load dfs
    combined_df = pd.read_csv(f'{instance_path}/combined_results.csv')
    combined_df = combined_df.set_index('Unnamed: 0')

    cv_auc_df = pd.read_csv(f'{instance_path}/cv_auc_by_subject.csv')
    cv_prec_df = pd.read_csv(f'{instance_path}/cv_prec_by_subject.csv')
    cv_recall_df = pd.read_csv(f'{instance_path}/cv_recall_by_subject.csv')
    cv_dice_df = pd.read_csv(f'{instance_path}/cv_dice_by_subject.csv')
    cv_acc_df = pd.read_csv(f'{instance_path}/cv_acc_by_subject.csv')
    cv_thresholds_df = pd.read_csv(f'{instance_path}/cv_thresholds_by_fold.csv')

    mda_mean = pd.read_csv(f'{instance_path}/combined_mda_mean.csv')
    mda_mean = mda_mean.drop('Unnamed: 0', axis=1)

    vol_df = pd.read_csv(f'{instance_path}/combined_vol_diff.csv')

    col_keep = ['NPV', 'CEM240', 'CTD', 'LRC-MP','RFC-MP']
    range_df = pd.DataFrame(columns=col_keep, index=['Dice', 'Precision', 'Recall', 'MDA [mm]', '% Volume Difference'])
    range_df = cv_to_range(cv_dice_df, 'Dice', range_df)
    range_df = cv_to_range(cv_prec_df, 'Precision', range_df)
    range_df = cv_to_range(cv_recall_df, 'Recall', range_df)
    range_df = cv_to_range(mda_mean, 'MDA [mm]', range_df, fmt='.1f')
    range_df = cv_to_range(vol_df, '% Volume Difference', range_df)

    avg_df = pd.DataFrame(columns=col_keep, index=['Dice', 'Precision', 'Recall', 'MDA [mm]', '% Volume Difference'])
    avg_df = cv_to_average(cv_dice_df, 'Dice', avg_df)
    avg_df = cv_to_average(cv_prec_df, 'Precision', avg_df)
    avg_df = cv_to_average(cv_recall_df, 'Recall', avg_df)
    avg_df = cv_to_average(mda_mean, 'MDA [mm]', avg_df, fmt='.1f')
    avg_df = cv_to_average(vol_df, '% Volume Difference', avg_df)

    range_df.to_csv(f'{dataroot}Output/Range_scores_table.csv')
    avg_df.to_csv(f'{dataroot}Output/Average_scores_table.csv')

    range_df.to_csv(f'{dataroot}Output/Paper_Figures/Range_scores_table.csv')
    avg_df.to_csv(f'{dataroot}Output/Paper_Figures/Average_scores_table.csv')