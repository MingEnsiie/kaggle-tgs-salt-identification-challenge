import cv2
import numpy as np
import os


def extract_hog(image):
    pass


def extract_lbp(image):
    pass


def extract_dct(image):
    pass


def extrac_features(image):
    pass


def train_model():
    pass

def iou(mask1,mask2):
    pass

if __name__ == "__main__":

    dataset_images_folder = '../data/train/images'
    dataset_masks_folder = '../data/train/masks'

    with open('../data/train/train_images.txt') as f:
        dataset_filenames = [i.strip() for i in f.readlines()]
        dataset_images_path = [os.path.join(dataset_images_folder,i) for i in dataset_filenames]
        dataset_masks_path = [os.path.join(dataset_masks_folder,i) for i in dataset_filenames]

    
    
