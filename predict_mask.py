from __future__ import division

import numpy as np
import pickle
import cv2
import math

from extract_classical_features import extract_hog
IMAGE_SIZE = 101

hog_win_size = 8
hog_block_size = 8
hog_block_stride = 4
hog_cell_size = 4
hog_nbins = 9

feat_win_size = 8
feat_win_stride = 6


def predict_sklearn_model(model,image):

    features = extract_hog(image)

    mask_patches = np.argmax(np.swapaxes(np.array(model.predict_proba(features)),0,1),axis=2)
    mask_patches = np.reshape(mask_patches,(mask_patches.shape[0],int(math.sqrt(mask_patches.shape[1])),int(math.sqrt(mask_patches.shape[1]))))

    k = 0
    mask = np.zeros((IMAGE_SIZE,IMAGE_SIZE))

    #Forward
    for i in range(0, (IMAGE_SIZE - feat_win_size + 1), feat_win_stride):
        for j in range(0, (IMAGE_SIZE - feat_win_size + 1), feat_win_stride):
            mask[i:(i + feat_win_size), j:(j + feat_win_size)] += mask_patches[k]            
            k+=1

    #Backward
    for i in range((IMAGE_SIZE - feat_win_size),0, -feat_win_stride):
        for j in range((IMAGE_SIZE - feat_win_size),0, -feat_win_stride):
            mask[i:(i + feat_win_size), j:(j + feat_win_size)] += mask_patches[k]            
            k+=1

    mask = (255*(mask/2)).astype(np.uint8)
    cv2.imwrite("teste.png",mask)
    print(mask.shape)


if __name__ == '__main__':

    model_name = '../models/rf_hog'
    model = pickle.load(open(model_name,'rb'))

    image_path = '../data/train/images/0a1742c740.png'
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    predict_sklearn_model(model,image)


