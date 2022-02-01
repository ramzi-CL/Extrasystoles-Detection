# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:11:30 2022

@author: RamziAbdelhafidh
"""


import pandas as pd
from preprocessing_utils import *
from hots_utils import *
from constants import *
from sklearn.svm import SVC
import pickle


# load data
df = load_arrhythmia_files(MIT_FOLDER_PATH)

#filter ecg signals
df = filter_df_ecgs(data=df, fs=FS, sig_col_name='signal')


# segmentation & add RR
df = generate_templates(data=df, 
                        normal_beats=NORMAL_BEATS, 
                        extrasystole_beats=EXTRASYSTOLE_BEATS, 
                        fs=FS)

# normalization
df = minmax_scale(df)


# train test split
df_train, df_test = train_test_split(df, DS_TRAIN_NAMES, DS_TEST_NAMES)


# classifier construction
x_train = df_train.drop(columns=['record_name', 'template', 'label'], axis=1)
y_train = df_train['label']
clf = SVC(**SVC_PARAMS, class_weight='balanced')
clf.fit(x_train, y_train)


# save HOTS centers
import pickle
with open('centers.pkl','wb') as f:
    pickle.dump(centers, f)

# save classifier
with open('classifier.pkl','wb') as f:
    pickle.dump(clf, f)