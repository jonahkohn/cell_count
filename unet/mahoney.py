import numpy as np
from unet import utils
import math

# from unet import sim_measures

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

# from unet.metrics import recall, precision, f1, mcor
# from unet.losses import weighted_bce_dice_loss

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from scipy.misc import imsave
from skimage.measure import label, regionprops
from skimage.transform import resize, downscale_local_mean
from skimage import img_as_uint

import sys
import os

# import cv2

import tifffile

def roiSaveMask(msk, ch_name, output_folder, filetype = 'tif'):
    # path = './train_data/' + ch_name
    path = output_folder + '/masks'
    if not os.path.isdir(path):
        os.makedirs(path)
    # img_path = output_folder + ch_name
    

    outfile = os.path.join(output_folder, 'masks/' + ch_name + '_mask.' + filetype)
    imsave(outfile, msk)#np.squeeze(msk, axis=2))


def split_2d_img(np_img, x_dim=1024, y_dim=1024, x_dim_loc=1, y_dim_loc=0, diff_out_size=False, output_x_dim=1024, output_y_dim=1024):
    img_x_dim = np_img.shape[x_dim_loc]
    img_y_dim = np_img.shape[y_dim_loc]
    print("IMG SHAPE: ", img_x_dim, img_y_dim)

    n_slice_x = math.ceil(img_x_dim/x_dim)
    n_slice_y = math.ceil(img_y_dim/y_dim)

    x_grid_size = int(img_x_dim/n_slice_x)
    y_grid_size = int(img_y_dim/n_slice_y)

    x_overlap = int(((n_slice_x*x_dim)-img_x_dim)/(n_slice_x-1))
    y_overlap = int(((n_slice_y*y_dim)-img_y_dim)/(n_slice_y-1))

    print("OVERLAP x,y:", x_overlap, y_overlap)

    coord_list = []

    last_x = 0
    slice_num = 0
    
    for x_slice in range(n_slice_x):
        if x_slice == 0:
            x_coord_start = last_x
        else:
            x_coord_start = last_x-(x_overlap)
        x_coord_end = x_coord_start+x_dim
        if x_coord_end > img_x_dim:
            x_coord_end = img_x_dim
        last_x = x_coord_end
        # print(x_coord_start, x_coord_end)

        last_y = 0
        for y_slice in range(n_slice_y):
            if y_slice == 0:
                y_coord_start = last_y
            else:
                y_coord_start = last_y-(y_overlap)
            y_coord_end = y_coord_start+y_dim
            if y_coord_end > img_y_dim:
                y_coord_end = img_y_dim
            last_y = y_coord_end
            # print(y_coord_start, y_coord_end)
            slice_img = np_img[y_coord_start:y_coord_end,x_coord_start:x_coord_end]
            if slice_img.shape[:2] != (y_dim, x_dim):
                slice_img = resize(slice_img, (y_dim, x_dim, 1))
            # print(slice_img.shape)
            if diff_out_size:
                slice_img = resize(slice_img, (output_y_dim, output_x_dim, 1))
            slice_coord_dict = {'slice_n':slice_num, 'x_start':x_coord_start, 'x_end':x_coord_end, 'y_start':y_coord_start, 'y_end':y_coord_end, 'slice_img':slice_img, 'mask':None}
            coord_list.append(slice_coord_dict)
            slice_num = slice_num + 1


    return(coord_list)


def all_the_kings_men(peices, x_dim, y_dim, dtype=int, diff_out_size=False):
    temp_img = np.zeros((y_dim, x_dim))#, dtype=dtype)
    temp_mask = np.zeros((y_dim, x_dim))
    
    for p in peices:
        x_start = p['x_start']
        x_end = p['x_end']
        y_start = p['y_start']
        y_end = p['y_end']
        x_size = x_end-x_start
        y_size = y_end-y_start
        part_of_img = p['slice_img']
        early_mask = p['mask']

        # print(part_of_img.shape, early_mask.shape, y_size, x_size)

        
        if part_of_img.shape != (y_size, x_size, 1):
            # print("GOTTA RESHAPE!")
            part_of_img = resize(part_of_img, (y_size, x_size, 1))
            early_mask = resize(early_mask, (y_size, x_size, 1))

        temp_img[y_start:y_end, x_start:x_end] = part_of_img[:,:,0]
        temp_mask[y_start:y_end, x_start:x_end] = early_mask[:,:,0]
    
    # print('PUT TOGETHER IMG: ', temp_img.shape, np.max(temp_img))
    # print('PUT TOGETHER MASK: ', temp_mask.shape, np.max(temp_mask))
    return(temp_img, temp_mask)


def merge_channels(list_of_arrays):
    combined_array = list_of_arrays[0]
    # n_ch = len(list_of_arrays)
    for arr in list_of_arrays[1:]:
        combined_array = np.maximum(combined_array, arr)
    return(combined_array)

