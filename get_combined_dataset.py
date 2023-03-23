import tools as tls
import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import os
import glob
import torch
import pickle
import numpy as np
import camp.FileIO as io


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
    trainROI_feats, trainROI_labels, quadROI_feats, quadROI_labels = tls.get_data(dataroot,
                                                                                  feats=feats)  # feats=['ctd', 'max', 't2', 'adc'])
    trainROI_clinical, quadROI_clinical = tls.get_data_clinical(dataroot)

    if data_mask == 'quad':
        return quadROI_feats, quadROI_labels, quadROI_clinical

    if data_mask == 'hist_dilate':
        return trainROI_feats, trainROI_labels, trainROI_clinical

def create_rabbit_dict():
    data_dir = dataroot + 'Data/'

    rabbit_list = [x.split('/')[-1] for x in sorted(glob.glob(f'{data_dir}/18_*'))]
    rabbit_dict = {f'{x}': {} for x in rabbit_list}

    print('===> Loading volumes for each subject ... ')

    # Load the necessary volumes
    for i, r in enumerate(rabbit_list):
        print(f'=> Suject {r}: Loading masks ... ', end='')
        train_mask = io.LoadITKFile(f'{data_dir}{r}/{r}_train_mask.nrrd')
        eval_mask = io.LoadITKFile(f'{data_dir}{r}/{r}_thermal_tissue.nrrd')

        rabbit_dict[r]['hist_dilate_mask'] = train_mask.clone()
        rabbit_dict[r]['quad_mask'] = eval_mask.clone()
        print('done')

    with open(dict_savepath + 'rabbit_dictionary.pkl', 'wb') as f:
        pickle.dump(rabbit_dict, f)

    return rabbit_dict

def recon_prediction(pred, rabbit, region='quad'):
    # Load the target volume
    data_dir = dataroot + 'Data/'

    if region == 'hist_dilate':
        mask = rabbit_dict[rabbit]['hist_dilate_mask']
        #mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_train_mask.nrrd')
    else:
        mask = rabbit_dict[rabbit]['quad_mask']
        #mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_thermal_tissue.nrrd')

    out_vol = mask.clone()
    if hasattr(pred, 'float'):
        out_vol.data[mask.data.bool()] = pred.float()
    else:
        out_vol.data[mask.data.bool()] = torch.tensor(pred).float()

    return out_vol

def get_data_splits(rabbit_dict, test_percent, features=None, save_subject_splits=True):

    if not features:
        features = ['ctd', 'max', 't2', 'adc']

    #quadROI_feats, quadROI_labels, quadROI_clinical = compile_data(data_mask='quad')
    quadROI_feats, quadROI_labels, quadROI_clinical = tls.get_data_roi(dataroot, roi='quad', feats=features)

    print(f'Generating training data ... ', end='')
    # TRAINING DATA
    # Get 30% of quadricep data from each rabbit, concatenate into single training data set.
    # Stratified sampling.

    data_dict = {}

    all_train_feats = []
    all_train_labels = []
    all_test_feats = []
    all_test_labels = []
    all_train_clinical = []
    all_test_clinical = []

    for i, r in enumerate(rabbit_dict):
        quad_feats = quadROI_feats[i].clone()
        #quad_feats = quad_feats.reshape(quad_feats.shape[0], -1).squeeze()
        quad_labels = quadROI_labels[i].clone()
        quad_clinical = quadROI_clinical[i].clone()

        # for each subject, extract the test/train data and concatenate into full dataset
        train_feats, train_labels, test_feats, test_labels, train_index, test_index = tls.split_data(quad_feats, quad_labels, test_percent)

        all_train_feats.append(train_feats)
        all_train_labels.append(train_labels)
        all_train_clinical.append(quad_clinical[train_index, :])
        all_test_feats.append(test_feats)
        all_test_labels.append(test_labels)
        all_test_clinical.append(quad_clinical[test_index, :])

        # for each subject, save the linearized features, labels, clinical data, and test/train indices
        if save_subject_splits:

            rabbit_dict[r]['quad_feats'] = quad_feats
            rabbit_dict[r]['quad_labels'] = quad_labels
            rabbit_dict[r]['quad_clinical'] = quad_clinical
            rabbit_dict[r]['train_index'] = train_index
            rabbit_dict[r]['test_index'] = test_index
            # save volumetric mask of train and test indices
            # train_mask_linear = torch.tensor(np.zeros(quad_labels.shape))
            # inds = rabbit_dict[r]['train_index']
            # train_mask_linear[inds] = 1
            # rabbit_dict[r]['train_mask'] = recon_prediction(train_mask_linear, r, region='quad')
            # test_mask_linear = torch.tensor(np.zeros(quad_labels.shape))
            # inds = rabbit_dict[r]['test_index']
            # test_mask_linear[inds] = 1
            # rabbit_dict[r]['test_mask'] = recon_prediction(test_mask_linear, r, region='quad')

    # save test/train data from all subjects to data_dict
    data_dict['X_train'] = torch.cat(all_train_feats, 0)
    data_dict['y_train'] = torch.cat(all_train_labels, 0)
    data_dict['X_test'] = torch.cat(all_test_feats, 0)
    data_dict['y_test'] = torch.cat(all_test_labels, 0)
    data_dict['clinical_train'] = torch.cat(all_train_clinical, 0)
    data_dict['clinical_test'] = torch.cat(all_test_clinical, 0)
    data_dict['features'] = features

    # save the data_dict and rabbit_dictionaries
    with open(f'{dict_savepath}/rabbit_dictionary.pkl', 'wb') as f:
        pickle.dump(rabbit_dict, f)
    with open(f'{dict_savepath}/data_splits_dictionary.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    print('done')
    return rabbit_dict, data_dict

if __name__ == '__main__':

    global dataroot
    dataroot = '/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/'

    # where to store estimator results and generated dictionaries
    global dict_savepath
    dict_folder = '70per_training_cor'  # Folder for storing dictionaries
    test_per = .3  # Test % of total data

    dict_savepath = f'{dataroot}Dictionaries/Combined Datasets/{dict_folder}/'
    if not os.path.exists(dict_savepath):
        os.mkdir(dict_savepath)

    rabbit_dict = create_rabbit_dict()
    get_data_splits(rabbit_dict, 0.3, save_subject_splits=True)


