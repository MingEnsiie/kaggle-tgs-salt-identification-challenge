from __future__ import division

import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)



if __name__ == '__main__':

    dataset_dir = '../data/'
    images_dir = os.path.join(dataset_dir,'images')
    #image_list_file = os.path.join(dataset_dir,'evalset.txt')
    image_list_file = os.path.join(dataset_dir,'test_set.txt')
    #output_dir = os.path.join(dataset_dir,'predictions')
    output_dir = os.path.join(dataset_dir,'predictions_test_set','cnn_v1')
    model_path = './trained_models/v1/1532944184'

    model = tf.contrib.predictor.from_saved_model(model_path,'predicted_mask')

    with open(image_list_file) as f:
        image_list = [l.strip() for l in f.readlines()]

    for image_name in tqdm(image_list):
        #tf.logging.info('Processing: {}'.format(image_name))
        image_path = os.path.join(images_dir,image_name)
        
        with open(image_path,'rb') as f:
            image_bytes = f.read()
        
        prediction = np.squeeze(model({'image':[image_bytes]})['output'])
        mask = np.argmin(prediction,axis=2)
        image_mask = (mask*255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir,image_name),image_mask)


