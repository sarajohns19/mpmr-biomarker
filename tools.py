import glob
import torch
import numpy as np
import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import sklearn.metrics as metrics
#import CAMP.camp.StructuredGridOperators as so
import camp.StructuredGridOperators as so
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import camp.FileIO as io

import matplotlib
matplotlib.use('module://backend_interagg')
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()



np.random.seed(31415)



def get_fmeasure_thold(data, label):

    fpr, tpr, thold = metrics.roc_curve(label.squeeze(), data.squeeze(), drop_intermediate=False)
    return fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], thold[np.argmax(tpr - fpr)]


def get_dice_thold(data, label):
    test_dce = []
    for t in range(1000):
        _, _, dice, _ = calc_prec_rec_dice((data >= t / 1000.0), label)
        test_dce.append(dice)

    return np.argmax(test_dce) / 1000.0

def get_dice_thold_v2(pred_data, label_data, rescale=False):

    if rescale is True:
        pred_data_temp = (pred_data.clone() - pred_data.min()) / (
                    pred_data.max() - pred_data.min())  # must be between 0 and 1

        ## from tls.get_dice_thold ##
        test_dce = []
        for t in range(1000):
            _, _, dice, _ = calc_prec_rec_dice((pred_data_temp >= t / 1000.0), label_data)
            test_dce.append(dice)

        th_temp = np.argmax(test_dce) / 1000.0
        #############################
        th = ((th_temp * (pred_data.max() - pred_data.min())) + pred_data.min()).item()

    else:
        test_dce = []
        for t in range(1000):
            _, _, dice, _ = calc_prec_rec_dice((pred_data >= t / 1000.0), label_data)
            test_dce.append(dice)

        th = np.argmax(test_dce) / 1000.0

    return th


def get_j_thold(data, label):
    test_j = []
    for t in range(1000):
        j = calc_j_score((data >= t / 1000.0), label)
        test_j.append(j)

    return np.argmax(test_j) / 1000.0


def calc_prec_rec_dice(pred, label):
    np.seterr(invalid='ignore')

    if (type(pred) is list) or (type(pred) is tuple):
        dice = []
        prec = []
        reca = []
        acc = []
        for f in range(0, len(pred)):
            # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
            TP = np.logical_and(pred[f] == 1, label[f] == 1).sum()

            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = np.logical_and(pred[f] == 0, label[f] == 0).sum()

            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = np.logical_and(pred[f] == 1, label[f] == 0).sum()

            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = np.logical_and(pred[f] == 0, label[f] == 1).sum()

            dice.append((2 * TP) / float(((2 * TP) + FP + FN)))
            prec.append(TP / float((TP + FP)))
            reca.append( TP / float((TP + FN)))
            acc.append((TP + TN) / float(TP + TN + FP + FN))

    else:
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.logical_and(pred == 1, label == 1).sum()

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.logical_and(pred == 0, label == 0).sum()

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.logical_and(pred == 1, label == 0).sum()

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.logical_and(pred == 0, label == 1).sum()

        dice = (2 * TP) / float(((2 * TP) + FP + FN))
        prec = TP / float((TP + FP))
        reca = TP / float((TP + FN))
        acc = (TP + TN) / float(TP + TN + FP + FN)

    return prec, reca, dice, acc


def calc_j_score(pred, label):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.float(np.logical_and(pred == 1, label == 1).sum())

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.float(np.logical_and(pred == 0, label == 0).sum())

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.float(np.logical_and(pred == 1, label == 0).sum())

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.float(np.logical_and(pred == 0, label == 1).sum())

    if TP == 0 and FN == 0:
        ft = 0.0
    else:
        ft = (TP / (TP + FN))

    if TN == 0 and FP == 0:
        st = 0.0
    else:
        st = (TN / (TN + FP))

    acc = (ft + st) - 1

    return acc


def get_segmentation_boundaries(seg):
    # Get the boundary indices
    grad_vol = so.Gradient.Create(dim=3)(seg)
    grad_vol.data = (grad_vol.data != 0.0).float()

    # Intersect the gradient with the segmentation because we are using central difference gradient
    grad_vol = grad_vol * seg
    boundary_vol = seg.clone()
    boundary_vol.data = grad_vol.data.max(dim=0, keepdim=True)[0]

    return boundary_vol


def calc_mean_dist_agreement(pred_label, target_label):
    # Get the boundaries of the labels
    pred_bounds = get_segmentation_boundaries(pred_label)
    target_bounds = get_segmentation_boundaries(target_label)

    if pred_bounds.max() == 0.0:
        return torch.tensor(0.0), torch.tensor(0.0)

    # Get the indicies of label
    pred_inds = torch.nonzero(pred_bounds.data.squeeze(), as_tuple=False).float()
    target_inds = torch.nonzero(target_bounds.data.squeeze(), as_tuple=False).float()

    distance_mat = ((pred_inds.permute(1, 0).unsqueeze(0) - target_inds.unsqueeze(2)) ** 2).sum(1).sqrt()
    min_dist_vector = torch.cat([distance_mat.min(dim=0)[0], distance_mat.min(dim=1)[0]], dim=0)
    mean_dist_agreement = min_dist_vector.mean()
    std_dist_agreement = min_dist_vector.std()

    return mean_dist_agreement, std_dist_agreement


def calc_signed_distance(pred_label, target_label):
    # Get the boundaries of the labels
    pred_bounds = get_segmentation_boundaries(pred_label)
    target_bounds = get_segmentation_boundaries(target_label)

    # Get the boundary pixels that are within the target segmentation
    internal = torch.logical_and(pred_bounds.data, target_label.data)
    external = torch.logical_and(torch.logical_xor(pred_bounds.data, target_label.data), pred_bounds.data)

    sign_vol = pred_bounds.clone()
    sign_vol.data[internal] = -1.0
    sign_vol.data[external] = 1.0

    # Get the indicies of label
    signs = sign_vol.data[pred_bounds.data != 0.0]
    pred_inds = torch.nonzero(pred_bounds.data.squeeze(), as_tuple=False).float()
    target_inds = torch.nonzero(target_bounds.data.squeeze(), as_tuple=False).float()

    distance_mat = ((pred_inds.permute(1, 0).unsqueeze(0) - target_inds.unsqueeze(2)) ** 2).sum(1).sqrt()
    min_dist_vector = distance_mat.min(dim=0)[0]
    signed_distance_vector = signs * min_dist_vector

    return signed_distance_vector


def _get_old_data(feats=None, patch=True):
    data_dir = '/home/sci/blakez/ucair/AcuteBiomarker/ClassifierData/Patch3X3/'
    rabbit_list = [x.split('/')[-1].split('_te')[0] for x in sorted(glob.glob(f'{data_dir}18_*_test_label*'))]

    train_feats = []
    train_labels = []

    test_feats = []
    test_labels = []

    if not feats:
        feats = ['ctd', 'max', 't2', 'adc']

    full_feat_list = ['ctd', 'max', 't2', 'adc', 't1']

    for rabbit in rabbit_list:
        train_labels.append(torch.load(f'{data_dir}{rabbit}_train_labels.pt').long())
        test_labels.append(torch.load(f'{data_dir}{rabbit}_test_labels.pt').long())

        train_feat = torch.load(f'{data_dir}{rabbit}_train_features.pt')
        test_feat = torch.load(f'{data_dir}{rabbit}_test_features.pt')

        temp_train_feats = []
        temp_test_feats = []
        # if train_feat.shape[1] != len(full_feat_list):
        #     exit('Wrong number of features in the data...')
        for i, f in enumerate(full_feat_list):
            if f in feats:
                if patch:
                    temp_train_feats.append(train_feat[:, i, None, :, :])
                    temp_test_feats.append(test_feat[:, i, None, :, :])
                else:
                    temp_train_feats.append(train_feat[:, i, None, 1, 1])
                    temp_test_feats.append(test_feat[:, i, None, 1, 1])
        train_feats.append(torch.cat(temp_train_feats, 1))
        test_feats.append(torch.cat(temp_test_feats, 1))

    return train_feats, train_labels, test_feats, test_labels

def get_data_roi(data_dir, roi='quad', feats=None):
    data_dir = data_dir + 'ProcessedData/'
    rabbit_list = [x.split('/')[-1].split('_qu')[0] for x in sorted(glob.glob(f'{data_dir}18_*_quad_label*'))]

    data_feats = []
    data_labels = []
    data_clinical = []

    if not feats:
        feats = ['ctd', 'max', 't2', 'adc']

    full_feat_list = ['ctd', 'max', 't2', 'adc', 't1']

    for rabbit in rabbit_list:
        # load torch 1D arrays of train/test labels
        if roi == 'hist_dilate':
            data_labels.append(torch.load(f'{data_dir}{rabbit}_train_labels.pt').long())
            data_clinical.append(torch.load(f'{data_dir}{rabbit}_train_clinical.pt'))
            data_feat = torch.load(f'{data_dir}{rabbit}_train_features.pt')

        elif roi == 'quad':
            data_labels.append(torch.load(f'{data_dir}{rabbit}_quad_labels.pt').long())
            data_clinical.append(torch.load(f'{data_dir}{rabbit}_quad_clinical.pt'))
            data_feat = torch.load(f'{data_dir}{rabbit}_quad_features.pt')

        temp_feats = []
        for i, f in enumerate(full_feat_list):
            if f in feats:
                temp_feats.append(data_feat[:, i, None, :, :])

        data_feats.append(torch.cat(temp_feats, 1))

    return data_feats, data_labels, data_clinical


def get_data(data_dir, feats=None):
    data_dir = data_dir + 'ProcessedData/'
    rabbit_list = [x.split('/')[-1].split('_qu')[0] for x in sorted(glob.glob(f'{data_dir}18_*_quad_label*'))]

    train_feats = []
    train_labels = []
    train_clinical = []

    test_feats = []
    test_labels = []
    test_clinical = []

    if not feats:
        feats = ['ctd', 'max', 't2', 'adc']

    full_feat_list = ['ctd', 'max', 't2', 'adc', 't1']

    for rabbit in rabbit_list:
        # load torch 1D arrays of train/test labels
        train_labels.append(torch.load(f'{data_dir}{rabbit}_train_labels.pt').long())
        test_labels.append(torch.load(f'{data_dir}{rabbit}_quad_labels.pt').long())

        # load torch 1D arrays of train/test feat
        train_feat = torch.load(f'{data_dir}{rabbit}_train_features.pt')
        test_feat = torch.load(f'{data_dir}{rabbit}_quad_features.pt')

        temp_train_feats = []
        temp_test_feats = []
        # if train_feat.shape[1] != len(full_feat_list):
        #     exit('Wrong number of features in the data...')
        for i, f in enumerate(full_feat_list):
            if f in feats:
                temp_train_feats.append(train_feat[:, i, None, :, :])
                temp_test_feats.append(test_feat[:, i, None, :, :])

        train_feats.append(torch.cat(temp_train_feats, 1))
        test_feats.append(torch.cat(temp_test_feats, 1))


    # old_train_f, old_train_l, old_test_f, old_test_l = _get_old_data(['ctd', 'max', 't2', 'adc'])
    #
    # if old_train_f != train_feats:
    #     print('Diff Train Feats...', end='')
    # if old_train_l != train_labels:
    #     print('Diff Train Feats...', end='')
    # if old_test_f != test_feats:
    #     print('Diff Train Feats...', end='')
    # if old_test_l != test_labels:
    #     print('Diff Train Feats...', end='')

    return train_feats, train_labels, test_feats, test_labels

def get_data_clinical(data_dir):
    data_dir = data_dir + 'ProcessedData/'
    rabbit_list = [x.split('/')[-1].split('_qu')[0] for x in sorted(glob.glob(f'{data_dir}18_*_quad_label*'))]

    train_clinical = []
    test_clinical = []

    for rabbit in rabbit_list:
        # load torch 1D arrays of clinical metric data
        train_clinical.append(torch.load(f'{data_dir}{rabbit}_train_clinical.pt'))
        test_clinical.append(torch.load(f'{data_dir}{rabbit}_quad_clinical.pt'))

    return train_clinical, test_clinical

def split_data(data, labels, test_percent):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_percent, random_state=0)
    for train_index, test_index in sss.split(data, labels):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

    return train_data, train_labels, test_data, test_labels, train_index, test_index

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    from matplotlib.patches import Patch
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        #xlim=[0, 10000],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    plt.show()

def logistic_regression(train_data, train_labels, test_data):

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_data, train_labels)
    train_proba = clf.predict_proba(train_data)[:, 1]
    test_proba = clf.predict_proba(test_data)[:, 1]

    return train_proba, test_proba


def random_forest(train_data, train_labels, test_data):

    clf = RandomForestClassifier(max_depth=21, n_estimators=700, n_jobs=16)
    clf.fit(train_data, train_labels)
    train_proba = clf.predict_proba(train_data)[:, 1]
    test_proba = clf.predict_proba(test_data)[:, 1]

    return train_proba, test_proba

def cross_validate_clf(clf, X, y, k):

    print('Running ' + str(k) + '-fold cross-validation for ' + str(clf.named_steps) + '...')

    kf = StratifiedKFold(n_splits=k)
    scoring = ['balanced_accuracy', 'f1', 'roc_auc']
    scores = cross_validate(clf, X, y, cv=kf, scoring=scoring, return_train_score=True)
    scores_df = pd.DataFrame(data=scores)
    scores_df.loc['mean'] = scores_df.mean()
    scores_df.loc['std'] = scores_df.std()

    print('Train Dice: ' + str(scores['train_f1']))
    print('Test Dice: ' + str(scores['test_f1']))
    print('Train Dice =  %.2f +/-%.2f' % (scores['train_f1'].mean(), scores['train_f1'].std()))
    print('Test Dice =  %.2f +/-%.2f' % (scores['test_f1'].mean(), scores['test_f1'].std()))

    return scores_df

# def train_clf(clf, X, y, test_percent):
#
#     print('Training model: ' + str(clf.named_steps) + '...')
#
#     train_feats, train_labels, test_feats, test_labels, train_index, test_index = split_data(X, y, test_percent)
#     clf.fit(train_feats, train_labels)
#     train_proba = clf.predict_proba(train_feats)[:, 1]
#     test_proba = clf.predict_proba(test_feats)[:, 1]
#
#     W = compute_sample_weight('balanced', train_labels)
#     print('train sample weights: ' + str(W.min()) + ', ' + str(W.max()))
#     W = compute_sample_weight('balanced', test_labels)
#     print('test sample weights: ' + str(W.min()) + ', ' + str(W.max()))
#
#     return train_proba, train_labels, test_proba, test_labels, clf, train_index, test_index

def train_clf(clf, train_data, train_labels, test_data):

    print('Training model: ' + str(clf.named_steps) + '...')

   # train_feats, train_labels, test_feats, test_labels, train_index, test_index = split_data(X, y, test_percent)
    clf.fit(train_data, train_labels)
    train_proba = clf.predict_proba(train_data)[:, 1]
    test_proba = clf.predict_proba(test_data)[:, 1]

    return train_proba, test_proba, clf

def plot_ROC(ax, pred_data, label_data, legendstr, linecolor, line_type='-'):

    tpr, fpr, tholds = metrics.roc_curve(label_data, pred_data)
    if legendstr is None:
        ax.plot(tpr, fpr, color=linecolor, linestyle=line_type)
    else:
        leg = f'{legendstr} AUC:{metrics.auc(tpr, fpr):.03f}'
        ax.plot(tpr, fpr, label=leg, color=linecolor, linestyle=line_type)

    return metrics.auc(tpr, fpr)

def plot_ROC_optimal(ax, pred_data, label_data, marker_color, marker_type, th=None, rescale=False):

    tpr, fpr, tholds = metrics.roc_curve(label_data, pred_data)

    if th is None:
        if rescale is True:
            pred_data_temp = (pred_data.clone() - pred_data.min()) / (pred_data.max() - pred_data.min())  # must be between 0 and 1
            th_temp = get_dice_thold(pred_data_temp, label_data)
            th = ((th_temp * (pred_data.max() - pred_data.min())) + pred_data.min()).item()
        else:
            th = get_dice_thold(pred_data, label_data)

    th_fpr = tpr[np.argmin(np.abs(tholds - th))]
    th_tpr = fpr[np.argmin(np.abs(tholds - th))]
    ax.scatter(th_fpr, th_tpr, color=marker_color, marker=marker_type, s=60)
    return th

def concat_subject_data(rabbit_dict, mask_name, data_name):

    tensor_list = []
    for i, r in enumerate(rabbit_dict):

        # These are masks with the original shape
        data_mask = rabbit_dict[r][mask_name].data.bool().clone()
        data = rabbit_dict[r][data_name].data[data_mask]
        tensor_list.append(data)

    data_cat = torch.cat(tensor_list)

    return data_cat