#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import argparse
import json
import os

dataset_path = '../data/train/features/hog/'
model_path = '../models/'

model_name = 'rf_hog'

data = np.load(os.path.join(dataset_path,"train_features.npy"))
labels = np.load(os.path.join(dataset_path,"train_labels.npy"))


rfc = RandomForestClassifier(n_estimators = 5,max_features=18,n_jobs=3,max_depth=5)
rfc.fit(data,labels)

import pickle
pickle.dump(rfc,open(os.path.join(model_path,model_name),"wb"))