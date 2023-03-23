import sys
sys.path.append('/home/mirl/sjohnson/CAMP/CAMP')
import os
import glob
import torch
#import CAMP.camp.FileIO as io
#import CAMP.camp.StructuredGridOperators as so
import camp.FileIO as io
import camp.StructuredGridOperators as so

import matplotlib
matplotlib.use('module://backend_interagg')
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
from scipy.io import savemat

device = 'cuda:0'


def process_data(data_dir, rabbit, quad_mask, mask_region='train'):
    # Run for each rabbit

    # load .nii.gz (Seg3D exports) files for all labels and MR contrasts
    print('Loading data files for ' + rabbit + '...')
    thermal_vol = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_thermal_vol.nii.gz')
    post_t2 = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_t2_w_post_file.nii.gz')
    pre_t2 = io.LoadITKFile(
        f'{dataroot}Data/{rabbit}/deform_files/{rabbit}_t2_w_pre_def.nii.gz')
    pre_t1 = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_t1_w_pre_file.nii.gz')
    post_t1 = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_t1_w_post_file.nii.gz')
    post_adc = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_adc_post_file.nii.gz')
    pre_adc = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_adc_pre_file.nii.gz')
    hist_label = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_hist_label.nii.gz')
    npv_label = io.LoadITKFile(f'{dataroot}Data/{rabbit}/{rabbit}_day0_npv_file.nrrd')

    # Create boolean mask for analysis ROI (quad_mask)
    quad_mask_bool = quad_mask.data.squeeze().bool()

    ### Preprocess MRI features for training classifiers ###

    # resample T2, ADC, and T1, maps onto quadricep ROI of rabbit (linear interpolation)
    post_t2 = so.ResampleWorld.Create(quad_mask)(post_t2)
    pre_t2 = so.ResampleWorld.Create(quad_mask)(pre_t2)

    post_adc = so.ResampleWorld.Create(quad_mask)(post_adc)
    pre_adc = so.ResampleWorld.Create(quad_mask)(pre_adc)

    post_t1 = so.ResampleWorld.Create(quad_mask)(post_t1)
    pre_t1 = so.ResampleWorld.Create(quad_mask)(pre_t1)

    # calculate MRI difference maps
    t2_diff = (post_t2.data.squeeze() - pre_t2.data.squeeze())
    t1_diff = (post_t1.data.squeeze() - pre_t1.data.squeeze())
    adc_diff = (post_adc.data.squeeze() - pre_adc.data.squeeze())

    # set NaNs in MRTI data to 0.0
    thermal_vol.data[torch.isnan(thermal_vol.data)] = 0.0

    # pad and unfold the MRI data (aka: make the 3x3 voxel patches)
    p = 3
    pad_vec = tuple([p // 2] * 2 + [0] * 2 + [p // 2] * 2 + [0] * 2)

    # MRTI:
    thermal_pad = torch.nn.functional.pad(thermal_vol.data.squeeze(), pad_vec)
    thermal_unfold = thermal_pad.unfold(1, p, 1).unfold(3, p, 1).contiguous()
    # thermal_view = thermal_unfold.reshape(list(thermal_vol.data.shape) + [-1]).permute(0, -1, 1, 2, 3).contiguous()
    # put time dimension in 2nd dim
    thermal = thermal_unfold.clone()[:, quad_mask_bool, :, :].permute(1, 0, 2, 3)

    # T1w/T2/ADC
    t1_pad = torch.nn.functional.pad(t1_diff, pad_vec[0:6])
    t1_unfold = t1_pad.unfold(0, p, 1).unfold(2, p, 1).contiguous()
    # t2_view = t2_unfold.reshape(list(t2_diff.shape) + [-1]).permute(-1, 0, 1, 2).contiguous()
    t1_view = t1_unfold.clone()[quad_mask_bool, :, :].squeeze().unsqueeze(1) # add back the "time" dimension

    t2_pad = torch.nn.functional.pad(t2_diff, pad_vec[0:6])
    t2_unfold = t2_pad.unfold(0, p, 1).unfold(2, p, 1).contiguous()
    # t2_view = t2_unfold.reshape(list(t2_diff.shape) + [-1]).permute(-1, 0, 1, 2).contiguous()
    t2_view = t2_unfold.clone()[quad_mask_bool, :, :].squeeze().unsqueeze(1)

    adc_pad = torch.nn.functional.pad(adc_diff, pad_vec[0:6])
    adc_unfold = adc_pad.unfold(0, p, 1).unfold(2, p, 1).contiguous()
    # adc_view = adc_unfold.reshape(list(adc_diff.shape) + [-1]).permute(-1, 0, 1, 2).contiguous()
    adc_view = adc_unfold.clone()[quad_mask_bool, :, :].squeeze().unsqueeze(1)

    # calculate CTD and maximum temperature
    rcem = torch.ones_like(thermal.squeeze())
    rcem[thermal >= 43.0] *= 0.5
    rcem[thermal < 43.0] *= 0.25
    ctd = (rcem.pow((43.0 - torch.clamp(thermal, thermal.min(), 95.0))) * 4.5/60).sum(1, keepdim=True)
    ctd = torch.clamp(ctd, ctd.min(), 1000000)

    max_vec = thermal.max(1, keepdim=True)[0]

    # generate the MRI features tensor
    feats = torch.cat([ctd, max_vec, t2_view, adc_view, t1_view], 1)


    ### Preprocess Histology label for training classifiers  ###

    # resample hist label onto quadricep ROI of rabbit. Reset to binary mask
    hist_label = so.ResampleWorld.Create(quad_mask)(hist_label)
    hist_label.data = (hist_label.data >= 0.5).long()

    # get voxels in the ROI
    hist_mask = hist_label.data[:, quad_mask_bool].squeeze()


    ## Preprocess clinical metrics (CTD, NPV) for comparison to classifiers

    # resample NPV label onto quadricep ROI of rabbit. Reset to binary mask
    npv_label = so.ResampleWorld.Create(quad_mask)(npv_label)
    npv_blur = so.Gaussian.Create(1, 3, 1, 3)(npv_label)
    npv_label.data = (npv_label.data >= 0.5).float()

    # get voxels in the ROI
    npv_label = npv_label.data[:, quad_mask_bool].squeeze()
    npv_blur = npv_blur.data[:, quad_mask_bool].squeeze()

    # calculate the CTD data for the quad ROI
    rcem = torch.ones_like(thermal_vol.data.squeeze())
    rcem[thermal_vol.data >= 43.0] *= 0.5
    rcem[thermal_vol.data < 43.0] *= 0.25
    ctd = (rcem.pow((43.0 - torch.clamp(thermal_vol.data, thermal_vol.data.min(), 95.0))) * 4.5/60).sum(0, keepdim=True)
    ctd = torch.clamp(ctd, ctd.min(), 1000000)
    ctdnp = {'CTD': ctd.detach().cpu().numpy()}
    savemat(f'{dataroot}Output/{rabbit}_ctd_volume.mat', ctdnp)
    ctd_data = ctd.data[:, quad_mask_bool].squeeze()

    # generate clinical metrics tensor
    clinical = torch.stack([npv_label, npv_blur, ctd_data], 1)

    print('Saving data files for ' + rabbit + '...')
    out_path = '/'.join(data_dir.split('/')[:-2] + ['']) + 'ProcessedData/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    torch.save(feats, f'{out_path}/{rabbit}_{mask_region}_features.pt')
    torch.save(hist_mask, f'{out_path}/{rabbit}_{mask_region}_labels.pt')
    torch.save(clinical, f'{out_path}/{rabbit}_{mask_region}_clinical.pt')


if __name__ == '__main__':
    #global dataroot
    dataroot = '/v/raid10/users/sjohnson/Papers/2022_NoncontrastBiomarker/'
    data_dir = dataroot + 'Data/'
    rabbit_list = [x.split('/')[-1] for x in sorted(glob.glob(f'{data_dir}/18_*'))]

    norm_features = []
    labels = []

    for rabbit in rabbit_list:
        train_mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_train_mask.nrrd')
        eval_mask = io.LoadITKFile(f'{data_dir}{rabbit}/{rabbit}_thermal_tissue.nrrd')
        process_data(data_dir, rabbit, train_mask, mask_region='train')
        process_data(data_dir, rabbit, eval_mask, mask_region='quad')

