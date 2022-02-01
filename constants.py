# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:15:46 2022

@author: RamziAbdelhafidh
"""

MIT_FS = 360
CHRONOLIFE_FS = 200
MIT_FOLDER_PATH = 'mit-bih-arrhythmia-database-1.0.0'
NORMAL_BEATS = ['N', 'A', 'L', 'R', 'e', 'F']
EXTRASYSTOLE_BEATS = ['V', 'a']
DS_TRAIN_NAMES = ['101', '106', '108', '109', '112', '114', '115', '116',
                  '118', '119', '122', '124', '201', '203', '205', '207',
                  '208', '209', '215', '220', '223', '230']
DS_TEST_NAMES = ['100', '103', '105', '111', '113', '117', '121', '123',
                 '200', '202', '210', '212', '214', '219', '221', '222',
                 '228', '231', '232', '233', '234', '213']
MIT_HOTS_PARAMS = {'level_step': 0.025,
                   'nb_chan': 80,
                   'level_min': -1,
                   'level_max': 1,
                   'n_layers': 1,
                   'delta_chan': 5,
                   'nb_centers': 50,
                   'tau': 0.7,
                   'fast': True,
                   'density': True,
                   'fs': MIT_FS}
CHR_HOTS_PARAMS = {'level_step': 0.025,
                   'nb_chan': 80,
                   'level_min': -1,
                   'level_max': 1,
                   'n_layers': 1,
                   'delta_chan': 5,
                   'nb_centers': 50,
                   'tau': 0.7,
                   'fast': True,
                   'density': True,
                   'fs': CHRONOLIFE_FS}
SVC_PARAMS = {'C': 50.0,
              'gamma': 5.0,
              'kernel': 'rbf',
              'degree': 3}
MLP_PARAMS = {'hidden_layer_sizes': (35, ),
              'alpha': 0.0001,
              'activation': 'relu',
              'solver': 'adam',
              'learning_rate': 'constant',
              'max_iter': 200}
KNN_PARAMS = {'leaf_size': 30,
              'n_neighbors': 5,
              'algorithm': 'auto',
              'metric': 'manhattan',
              'weights': 'uniform',
              'p': 2}
