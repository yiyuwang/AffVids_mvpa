#!/usr/bin/env python
# coding: utf-8


# time stamp
import time
from datetime import datetime
start_time = time.time()

# dd/mm/YY H:M:S
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("start time: ", dt_string)

import sys


array_id_position = 1
job_array_id = sys.argv[array_id_position]
print ("Job_array_id %i: %s" % (array_id_position, job_array_id))


SEED = job_array_id

import glob
import os
import numpy as np
import pandas as pd
import random
import nibabel as nib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# for parallel processing
from joblib import Parallel, delayed, cpu_count
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

# stats
from scipy import linalg, ndimage, stats
from scipy.stats import norm

#scikit learn
from sklearn import linear_model
from sklearn.utils import check_random_state
from sklearn.metrics import make_scorer
from sklearn import decomposition

# nifti handling
from nilearn.input_data import NiftiMasker
from nilearn import decoding
import nilearn.masking as masking
from nilearn.masking import apply_mask
from nilearn.image import new_img_like, load_img, get_data, concat_imgs,threshold_img

# searchlihgt
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn.image.resampling import coord_transform

# plotting modules
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_img, plot_roi, show
import warnings
warnings.filterwarnings('ignore')

# directories
glm_dir = '/fmri_results/model2_OneRegPerVid_VisReg/1stLvl/'
res_dir = '/AffVids/fmri_results/SG12_permutation/'
       
if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
        
# masks:
mask_path ='/masks/FSL_binary_MNI152_T1_3mm_brain.nii.gz'
mask = nib.load(mask_path)


#load behavioral data
behavdata_dir =  '/BehavData/'

zratings = glob.glob(behavdata_dir +'AffVids_novel_interpolated_rating_zscored.csv')
zratings = pd.read_csv(zratings[0],index_col=0).reset_index()
zratings = zratings.sort_values(by=['sub_id','run']).drop(['index'], axis =1).reset_index(drop=True)


# subjects information
subjects_str = ['04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','23','25','26','28','29'] 
subjects = list(range(4,20))+[23,25,26,28,29]
Nsub = len(subjects)
print("subjects in this analysis:")
print(subjects_str)
print(f"**** n = {Nsub} *****" )


video_list = list(range(1,37))


def get_video_lists(vcat):
    
    if vcat == 'Heights':
        videos = list(range(1,13))
    elif vcat == 'Social':
        videos = list(range(13,25))
    elif vcat == 'Spiders':
        videos = list(range(25,37))
    else: # v_cat == 'Situation_General'
        videos = list(range(1,37))

    return videos

def get_category(vn):
    if vn < 13:
        cat = 'Heights'
        cn = 1
    elif vn > 24:
        cat = 'Spiders'
        cn = 3
    else:
        cat = 'Social'
        cn = 2
    return [cat,cn]


# copied from scipy : remove returning p value so that this function can be converted to a scorer in scikit-learn
def pearsonr(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')

    if n < 2:
        raise ValueError('x and y must have length at least 2.')

    x = np.asarray(x)
    y = np.asarray(y)

    if (x == x[0]).all() or (y == y[0]).all():
        return np.nan

    dtype = type(1.0 + x[0] + y[0])

    if n == 2:
        return dtype(np.sign(x[1] - x[0])*np.sign(y[1] - y[0]))

    xmean = x.mean(dtype=dtype)
    ymean = y.mean(dtype=dtype)

    xm = x.astype(dtype) - xmean
    ym = y.astype(dtype) - ymean

    normxm = linalg.norm(xm)
    normym = linalg.norm(ym)

    threshold = 1e-13
    if normxm < threshold*abs(xmean) or normym < threshold*abs(ymean):
        print('values close to the mean')

    r = np.dot(xm/normxm, ym/normym)
    r = max(min(r, 1.0), -1.0)
    return r


class GroupIterator(object):
    def __init__(self, n_features, n_jobs):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        split = np.array_split(np.arange(self.n_features), self.n_jobs)
        for list_i in split:
            yield list_i



# make lassopcr searchlight a function:
def my_lassopcr_searchlight(list_i, list_rows,X_train,X_test,train_y, test_y, thread_id):

    # check if the voxel index (list_i) is the same lenth as input data (list_rows)
    if len(list_rows) != len(list_i):
        raise ValueError('Voxel index does not equal to input data size!!!!')

    sl_scores=np.zeros(len(list_rows))
    sl_rmse = np.zeros(len(list_rows))
    

    for i, row in enumerate(list_rows):
        
        n_components=min(len(X_test[:, row]), len(X_train[:, row]))
        pca_fit = decomposition.PCA(n_components = n_components) # n_comp = number of testing sample (smaller than training sample)
        pca_train_x = pca_fit.fit_transform(X_train[:, row])

        pca_test_x = pca_fit.transform(X_test[:, row])

        # run LASSO
        clf = linear_model.Lasso(alpha=alpha,max_iter=5000)
        clf.fit(pca_train_x, train_y)

        prediction_test =clf.predict(pca_test_x)

        r_value = pearsonr(test_y, prediction_test)
        sl_scores[i] =r_value

        rmse = np.sqrt(np.mean((prediction_test-test_y)**2))
        sl_rmse[i] = rmse
    
    sl_scores_combined = [list(a) for a in zip(sl_scores, sl_rmse)]
    return sl_scores_combined


# function to randomly select 12 videos out of all the videos:
SEED = job_array_id
import random
def select_12_videos(SEED):
  
    """
    Randomly picks three unique elements from a list.
    
    Parameters:
        seed (int): Random seed for reproducibility.
        
    Returns:
        list: List of four randomly chosen videos from each , unique elements from the input list.
    """
    random.seed(SEED)
    # Randomly pick three unique items
    heights_vids = random.sample(list(range(1,13)), 4)
    social_vids = random.sample(list(range(13,25)), 4)
    spiders_vids = random.sample(list(range(25,37)), 4)
    
    return heights_vids + social_vids + spiders_vids

    
# model information
my_radius = 15
k_fold = 3
n_jobs = 16
alpha = 0.01 #LASSO regularization term

print(f'kfold split pre-generated:')
cv_train_subjects = [[4, 5, 6, 7, 8, 9, 10, 14, 17, 19, 23, 25, 28, 29],
 [4, 5, 6, 7, 11, 12, 13, 15, 16, 18, 19, 25, 26, 29],
 [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 26, 28]]

cv_test_subjects = [[11, 12, 13, 15, 16, 18, 26],
 [8, 9, 10, 14, 17, 23, 28],
 [4, 5, 6, 7, 19, 25, 29]]



# load nifti data into data
data = []
nii_file_list = []
nifti_masker = NiftiMasker(mask_img=mask)

for sidx in range(len(subjects)):
    s_str = subjects_str[sidx]
    s = subjects[sidx]
    print('running subject: ' + s_str)


    for v in zratings[zratings.sub_id == s].video_number:
        v = int(v)
        filename =glm_dir + f'{s_str}/sub-{s_str}_run-*_beta_video-{v}_gm_visreg.nii.gz'
        filename =glob.glob(filename)
        if filename:
            nii_file_list.append(filename[0])
            #data.append(nib.load(filename[0]).get_fdata()[:,:,:,0])
            data.append(nib.load(filename[0]).get_fdata())

data = np.moveaxis(data, 0, -1)



# prepare cv:
#############################################
# change the category before submitting jobs
train_cat = 'Situation_General'
test_cat = 'Spiders'
#############################################

print("training Category: ", train_cat)
print("Testing Category: ", test_cat)

# randomly select videos here:
testing_videos = get_video_lists(test_cat)
training_videos = select_12_videos(SEED)
print(training_videos)
print(testing_videos)


# compute searchlight coordinates
process_mask_img = mask
process_mask, process_mask_affine = masking._load_mask_img(
    process_mask_img)
process_mask_coords = np.where(process_mask != 0)
process_mask_coords = coord_transform(
    process_mask_coords[0], process_mask_coords[1],
    process_mask_coords[2], process_mask_affine)
process_mask_coords = np.asarray(process_mask_coords).T




for cv_iter in range(1,k_fold+1):
    print("cv: ", cv_iter)
    train_subjects = cv_train_subjects[cv_iter-1]
    test_subjects = cv_test_subjects[cv_iter-1]


    train_index = list(zratings[zratings.sub_id.isin(train_subjects) & zratings.video_number.isin(training_videos)].index)
    test_index = list(zratings[zratings.sub_id.isin(test_subjects) & zratings.video_number.isin(testing_videos)].index)
    
    # behavioral data (y)
    train_y = zratings.iloc[train_index].fear
    test_y = zratings.iloc[test_index].fear
    
    # brain data (x)
    train_x = data[:,:,:,train_index]
    test_x = data[:,:,:,test_index]


    new_affine = mask.affine.copy()
    train_img = new_img_like(mask, train_x, affine=new_affine)

    test_img = new_img_like(mask, test_x, affine=new_affine)


    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("training: applying mask and get affinity: ", dt_string)
    X_train, A_train = _apply_mask_and_get_affinity(
                process_mask_coords, train_img, my_radius, True, mask_img=mask)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("testing: applying mask and get affinity: ", dt_string)


    X_test, A_test = _apply_mask_and_get_affinity(
                process_mask_coords, test_img, my_radius, True, mask_img=mask)



    this_cv_scores = []
    this_cv_rmse = []

    print('starting searchlight')
    group_iter = GroupIterator(A_train.shape[0], n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter('ignore', ConvergenceWarning)
        sl_scores_combined = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(my_lassopcr_searchlight)(
                list_i,
                A_train.rows[list_i],
                X_train,
                X_test,
                train_y,
                test_y, 
                thread_id)
            for thread_id, list_i in enumerate(group_iter))

    sl_scores_combined = np.array(sl_scores_combined)

    sl_scores = sl_scores_combined[:,:,0]
    reshaped_sl_scores = np.reshape(sl_scores, (sl_scores.shape[0] *sl_scores.shape[1]))
    scores_3D = np.zeros(process_mask.shape)
    scores_3D[process_mask] = reshaped_sl_scores


    sl_rmse = sl_scores_combined[:,:,1]
    reshaped_sl_rmse = np.reshape(sl_rmse,(sl_rmse.shape[0]*sl_rmse.shape[1]))
    rmse_3D = np.zeros(process_mask.shape)
    rmse_3D[process_mask] = reshaped_sl_rmse

    # #save the scores
    this_cv_scores.append(scores_3D)
    this_cv_rmse.append(rmse_3D)

    this_cv_scores = np.moveaxis(this_cv_scores, 0, -1)
    this_cv_rmse = np.moveaxis(this_cv_rmse, 0, -1)


    # #save the scores to img
    this_cv_affine = mask.affine


    this_cv_scores_img = new_img_like(mask, this_cv_scores, affine=this_cv_affine)
    this_cv_rmse_img = new_img_like(mask, this_cv_rmse, affine=this_cv_affine)

    print('saving: iteration' + str(cv_iter))
    nib.save(this_cv_scores_img,res_dir + f'cv{cv_iter}_kfold3_searchlight_pearsonr_train_{train_cat}_test_{test_cat}_{job_array_id}.nii.gz')
    nib.save(this_cv_rmse_img,res_dir + f'cv{cv_iter}_kfold3_searchlight_rmse_train_{train_cat}_test_{test_cat}_{job_array_id}.nii.gz')


    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    iter_line = f"iteration {cv_iter} finished at {dt_string}"
    print(iter_line)

        