from __future__ import division

import cv2
import numpy as np
import os
import logging
import math
import sys
from tqdm import tqdm

from multiprocessing import Pool
from joblib import Parallel, delayed

IMAGE_SIZE = 101

hog_win_size = 8
hog_block_size = 8
hog_block_stride = 4
hog_cell_size = 4
hog_nbins = 9

feat_win_size = 8
feat_win_stride = 6


def _extract_hog_patch(patch, hog_win_size, hog_block_size, hog_block_stride, hog_cell_size, hog_nbins):
    """
    Extract HOG (Histogram of Oriented Gradients) from an image patch.
    See http://www.learnopencv.com/histogram-of-oriented-gradients/

    Parameters
    ----------
    patch : Numpy Array
        Image patch to extract HOG
    hog_win_size : tuple(int,int)
        HOG window size. It is supposed to have the same shape od patch.shape
    hog_block_size : tuple(int,int)
        HOG block_size
    hog_block_stride : tuple(int,int)
        HOG block stride
    hog_cell_size : tuple(int,int)
        HOG cell size
    hog_nbins : int
        Number of bins on HOG histogram

    Returns
    ----------
    hog_feature_vector : Numpy Array
        Extracted hog features

    """
    hog_descriptor = cv2.HOGDescriptor(
        (hog_win_size, hog_win_size), (hog_block_size, hog_block_size),
        (hog_block_stride, hog_block_stride), (hog_cell_size, hog_cell_size),
        hog_nbins)
    return hog_descriptor.compute(patch).reshape((1, -1)).ravel()


def _extract_hog(img, x_start, y_start, feat_win_size, hog_win_size, hog_block_size, hog_block_stride, hog_cell_size,
                 hog_nbins):
    """
    Extract HOG (Histogram of Oriented Gradients) from an image patch given the patch localtion.
    See http://www.learnopencv.com/histogram-of-oriented-gradients/

    Parameters
    ----------
    img : Numpy Array
        Image extract HOG
    x_start : int
        Initial x-coordenate of patch
    y_start : int
        Initial y-coordenate of patch
    feat_win_size : tuple(int,int)
         Patch shape (width,height)
    hog_win_size : tuple(int,int)
        HOG window size. It is supposed to have the same shape od patch.shape
    hog_block_size : tuple(int,int)
        HOG block_size
    hog_block_stride : tuple(int,int)
        HOG block stride
    hog_cell_size : tuple(int,int)
        HOG cell size
    hog_nbins : int
        Number of bins on HOG histogram

    Returns
    ----------
    patch_features : Numpy Array
        Extracted hog features

    """
    x_end = x_start + feat_win_size
    y_end = y_start + feat_win_size

    #logging.info("x_start {} x_end {} y_start {} y_end {}".format(x_start,x_end,y_start,y_end))

    patch = img[y_start:y_end, x_start:x_end]
    patch_features = _extract_hog_patch(
        patch, hog_win_size, hog_block_size, hog_block_stride, hog_cell_size, hog_nbins)

    return patch_features


def extract_hog(image):
    """
    Extract HOG feaures from an image

    See http://www.learnopencv.com/histogram-of-oriented-gradients/

    Parameters
    ----------
    image : Numpy Array
        Input image
    prediction : bool
        Switch feature window stride whether a prediction will be performed

    Returns
    ----------
    hog_feature_vector : Numpy Array
        Extracted hog features
    """
    hog_num_features = int(
        (math.pow(hog_block_size / hog_block_stride, 2)) * hog_nbins)

    

    features_foward = Parallel(n_jobs=3)(
        delayed(_extract_hog)(image, i, j, feat_win_size, hog_win_size, hog_block_size,
                              hog_block_stride, hog_cell_size, hog_nbins) for i in
        range(0, (IMAGE_SIZE - feat_win_size + 1), feat_win_stride) for j in
        range(0, (IMAGE_SIZE - feat_win_size + 1), feat_win_stride))

    
    features_backward = Parallel(n_jobs=3)(
        delayed(_extract_hog)(image, i, j, feat_win_size, hog_win_size, hog_block_size,
                              hog_block_stride, hog_cell_size, hog_nbins) for i in
        range((IMAGE_SIZE - feat_win_size),0, -feat_win_stride) for j in
        range((IMAGE_SIZE - feat_win_size),0, -feat_win_stride))

    features = features_foward+features_backward
    #logging.info(type(features))

    return np.array(features)


def extract_lbp(image):
    pass


def extract_dct(image):
    pass


def extract_features(image):
    pass


def split_mask(image):

    mask = []

    #Forward
    for i in range(0, (IMAGE_SIZE - feat_win_size + 1), feat_win_stride):
        for j in range(0, (IMAGE_SIZE - feat_win_size + 1), feat_win_stride):
            mask_patch = image[i:(i + feat_win_size), j:(j + feat_win_size)]
            mask.append(mask_patch)

    #Backward
    for i in range((IMAGE_SIZE - feat_win_size),0, -feat_win_stride):
        for j in range((IMAGE_SIZE - feat_win_size),0, -feat_win_stride):
            mask_patch = image[i:(i + feat_win_size), j:(j + feat_win_size)]
            mask.append(mask_patch)

    return np.array(mask)

# from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument('feature',defa)


logging.basicConfig(stream=sys.stderr, level=logging.INFO)


if __name__ == "__main__":

    dataset_folder = '../data/train/'
    dataset_images_folder = os.path.join(dataset_folder,'images')
    dataset_masks_folder = os.path.join(dataset_folder,'masks')    
    output_dir = os.path.join(dataset_folder,'features','hog')
    has_labels = True
    phase = "train"
    
    #with open('../data/train/mini_evalset.txt') as f:
    #with open('../data/train/evalset.txt') as f:
    with open('../data/train/trainset.txt') as f:
        dataset_filenames = [i.strip() for i in f.readlines()]

    labels = []
    features = []

    for filename in tqdm(dataset_filenames):
        logging.info("Extracting features from: {}".format(filename))

        input_path = os.path.join(dataset_images_folder, filename)
        input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        input_features = extract_hog(input_image) # (1152, 36)

        if has_labels:
            mask_path = os.path.join(dataset_masks_folder, filename)
            input_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8)
            mask_target = split_mask(input_mask) # (1152, 8, 8)
               
        labels.append(mask_target)
        features.append(input_features)

    if has_labels:
        labels_npy = np.array(labels)
        labels_npy = np.reshape(labels_npy,(labels_npy.shape[0]*labels_npy.shape[1],labels_npy.shape[2]*labels_npy.shape[3]))
        logging.info("labels_npy shape: {}".format(labels_npy.shape))        
        np.save(os.path.join(output_dir,'{}_labels.npy'.format(phase)),labels_npy)
        

    features_npy = np.array(features)    
    features_npy = np.reshape(features_npy,(features_npy.shape[0]*features_npy.shape[1],features_npy.shape[2]))
    logging.info("features_npy shape: {}".format(features_npy.shape))    
    np.save(os.path.join(output_dir,'{}_features.npy'.format(phase)),features_npy)
    
        
