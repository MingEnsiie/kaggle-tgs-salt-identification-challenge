from __future__ import division

import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

IMAGE_SIZE = 101

def convert_rle(mask):
    
    image = np.zeros((IMAGE_SIZE*IMAGE_SIZE))

    for i in range(0,len(mask),2):
        start_i = mask[i]-1 # 1-based indices
        end_i = start_i + mask[i+1]
        image[start_i:end_i] = 1

    image = np.transpose(np.reshape(image,(IMAGE_SIZE,IMAGE_SIZE)))
    return image


if __name__ == '__main__':

    dataset_dir = '../data/'    
    #image_list_file = os.path.join(dataset_dir,'test_set.txt')    
    image_list_file = os.path.join(dataset_dir,'mini_test_set.txt')    
    images_dir = os.path.join(dataset_dir,'predictions_test_set','cnn_v1')
    submission_file = os.path.join(dataset_dir,'predictions_test_set','mini_predictions_cnn_v1.txt')
    

    sub_df = pd.read_csv(submission_file)

    for row_id,row in sub_df.iterrows():        
        rle_mask = [int(i) for i in row['rle_mask'].split(' ')[1:-1]]
        mask = convert_rle(rle_mask)
        mask = (mask*255).astype(np.uint8)
        cv2.imwrite(os.path.join(dataset_dir,'predictions_test_set','check','{}.png'.format(row['id'])),mask)