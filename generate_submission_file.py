from __future__ import division

import os
import numpy as np
import cv2
from tqdm import tqdm


def get_mask(image):
    
    count = 1
    mask = ''
    init = 1
    SALT_PIXEL = 1
    for i in range(1,len(image)):
        curr_pixel = image[i]
        prev_pixel = image[i-1]
        
        if curr_pixel != prev_pixel:
            if curr_pixel == SALT_PIXEL:
                init=i
                count=1
            else:
                mask+= '{} {} '.format(init,count)
                count=0
            
        else:
            if curr_pixel == SALT_PIXEL:
                count+=1


    return mask

if __name__ == '__main__':

    dataset_dir = '../data/'    
   
    
    # images_dir = os.path.join(dataset_dir,'predictions_test_set','cnn_v1')
    # output_file = os.path.join(dataset_dir,'predictions_test_set','predictions_cnn_v1.txt')
    
    image_list_file = os.path.join(dataset_dir,'test_set.txt')    
    images_dir = os.path.join(dataset_dir,'predictions_test_set','cnn_v1_morph')
    #output_file = os.path.join(dataset_dir,'predictions_test_set','predictions_cnn_v1_morph.txt')
    output_file = os.path.join(dataset_dir,'predictions_test_set','predictions_cnn_v1_morph_neg.txt')

    with open(image_list_file) as f:
        image_list = [l.strip() for l in f.readlines()]

    with open(output_file,'w') as f:
        
        f.write('id,rle_mask\n')
        
        for image_name in tqdm(image_list):
            
            image_path = os.path.join(images_dir,image_name)
            
            image = (cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)/255).astype(np.uint8).transpose()
            image = np.reshape(image,(image.shape[0]*image.shape[1]))

            mask = get_mask(image)

            f.write('{}, {}\n'.format(image_name.split('.')[0],mask))


