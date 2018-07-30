from __future__ import division

import os
import numpy as np
import cv2
from tqdm import tqdm



if __name__ == '__main__':

    dataset_dir = '../data/'
    
    #images_dir = os.path.join(dataset_dir,'predictions_test_set','check')
    images_dir = os.path.join(dataset_dir,'predictions_test_set','cnn_v1')
    
    #image_list_file = os.path.join(dataset_dir,'predictions_test_set','list_check.txt')
    image_list_file = os.path.join(dataset_dir,'predictions_test_set','list_cnn_v1.txt')
        
    
    #output_dir = os.path.join(dataset_dir,'predictions')
    output_dir = os.path.join(dataset_dir,'predictions_test_set','cnn_v1_morph')
    
    
    with open(image_list_file) as f:
        image_list = [l.strip() for l in f.readlines()]

    for image_name in tqdm(image_list):
        
        image_path = os.path.join(images_dir,image_name)    
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)        

        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        erode = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel,iterations=1)
    
        cv2.imwrite(os.path.join(output_dir,image_name),erode)


