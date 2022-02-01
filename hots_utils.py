# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:57:01 2022

@author: RamziAbdelhafidh
"""

import numpy as np
import pandas as pd
from old_hots_functions import context_of_signal, centers_from_context, \
    features_classification, features_creation
from level_crossing import events_creation, levels_creation



def events_construction(signals, level_min, level_max, level_step, fs):
    levels = levels_creation(level_min=level_min, 
                             level_max=level_max, 
                             level_step=level_step)
    train_events = [events_creation(np.linspace(0, len(record)/fs, len(record)),
                                    record,
                                    [0]*len(record),
                                    levels)
                    for record in signals]
    return train_events


def create_centers(train_events, n_layers, nb_chan, delta_chan, 
                   nb_centers, tau, fast):
    train_final_context = []
    for i in range(len(list(train_events))):
        train_final_context.append(context_of_signal(events=list(train_events)[i],
                                                     n_layers=n_layers,
                                                     nb_chan=nb_chan,
                                                     delta_chan=delta_chan,
                                                     nb_centers=nb_centers,
                                                     tau=tau,
                                                     adaptative=False,
                                                     rr_indicators=None,
                                                     fast=fast))

    all_centers = centers_from_context(
        np.concatenate(train_final_context), nb_centers)
    return all_centers


def convert_old_polarities(old_events, n_layers, nb_chan, delta_chan, 
                           nb_centers, tau, fast, centers):
    final_events = []

    for i in range(len(list(old_events))):
        current_events, fc = features_classification(events=list(old_events)[i],
                                                     n_layers=n_layers,
                                                     nb_chan=nb_chan,
                                                     delta_chan=delta_chan,
                                                     nb_centers=nb_centers,
                                                     tau=tau,
                                                     adaptative=False,
                                                     rr_indicators=None,
                                                     fast=fast,
                                                     all_centers=np.array(
                                                         [centers]),
                                                     events_clf='kmeans')
        final_events.append(current_events)
    return final_events


def convert_events_to_histograms(events, nb_centers, n_layers, density):
    histograms = []
    for i in range(len(events)):
        sig_events = events[i]
        histograms.append(features_creation(
            sig_events, nb_centers*n_layers, density=density))
    histograms = [
        feature for sublist in histograms for feature in sublist]
    return histograms


def add_features_to_df(histograms, data):
    histograms = np.array(histograms)
    for i in range(histograms.shape[1]):
        data['hots_feature_'+str(i)] = histograms[:, i]
    return data


def generate_hots_train_features(df_train, level_step, nb_chan,
                                 level_min, level_max, n_layers,
                                 delta_chan, nb_centers, tau,
                                 fast, density, fs):
    # train events construction
    train_signals = [list(signal) for signal in df_train['template'].values]
    train_events = events_construction(train_signals, level_min, level_max,
                                       level_step, fs)

    # centers creation
    centers = create_centers(train_events, n_layers, nb_chan, delta_chan,
                             nb_centers, tau, fast)

    # Converting old polarities to new polarities
    train_final_events = convert_old_polarities(train_events, n_layers,
                                                nb_chan, delta_chan,
                                                nb_centers, tau,
                                                fast, centers)

    # Converting new polarities to histograms
    train_histograms = convert_events_to_histograms(train_final_events,
                                                    nb_centers, n_layers,
                                                    density)

    # add histograms to training dataframe
    df_train = add_features_to_df(train_histograms, df_train)

    return df_train, centers


def generate_hots_test_features(df_test, centers, level_step, nb_chan,
                                level_min, level_max, n_layers,
                                delta_chan, nb_centers, tau,
                                fast, density, fs):
    # test events construction
    test_signals = [list(signal) for signal in df_test['template'].values]
    test_events = events_construction(test_signals, level_min, level_max,
                                      level_step, fs)

    # Converting old polarities to new polarities
    test_final_events = convert_old_polarities(test_events, n_layers,
                                               nb_chan, delta_chan,
                                               nb_centers, tau,
                                               fast, centers)

    # Converting new polarities to histograms
    test_histograms = convert_events_to_histograms(test_final_events,
                                                   nb_centers, n_layers,
                                                   density)

    # add histograms to test dataframe
    df_test = add_features_to_df(test_histograms, df_test)

    return df_test
