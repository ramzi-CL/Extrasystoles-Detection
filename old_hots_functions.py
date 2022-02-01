# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:03:53 2022

@author: RamziAbdelhafidh
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def context_compute(current_event, nb_chan, tau, delta_chan, buff_channel, fast=False):
    """ Compute spatio-temporal context for each event
        Input: current_event, nb_chan, tau, delta_chan, buff_channel, fast
        current_event: triplet (ti,pi,ci)
                    ti: time
                    pi: polarity
                    ci: channel
            nb_chan: total number of channels sampling events data
            tau: time constant of the exponential kernel
            delta_chan: window of channels surrounding event to compute context
            buff_channel : Number of centers for the Kmeans classification
            fast : Compute context takes less polarities into account  
            Output: final_context 
    """
    # For each event compute the spatio-temporal context
    # Context is computed using inverse time difference between the event
    # And the surrounding events (around the same channel +/- delta_chan)

    t_ev = current_event[0]
    p_ev = int(current_event[1])
    chan_ev = int(current_event[2])
    buff_channel[chan_ev, p_ev] = t_ev
    final_context = []
    min_chan = chan_ev-delta_chan
    max_chan = chan_ev+delta_chan
    boarder = False
    if fast:
        polarities_range = [p_ev]
    else: 
        polarities_range = range(len(buff_channel[0, :]))     
    # Deal with event at the channel limits
    # Add zeros to the context above or below the limit chan
    # All contexts must have the same dimension
    for p_ev in polarities_range:
        if min_chan < 0:
            add_zeros = np.zeros((np.abs(min_chan), 1))
            min_chan = 0
            context = list(add_zeros)
            context.extend(buff_channel[min_chan:max_chan, p_ev])
            context = np.array(context)
            boarder = True
        if max_chan >= nb_chan:
            add_zeros = np.zeros((max_chan-nb_chan+1, 1))
            max_chan = nb_chan-1
            context = list(buff_channel[min_chan:max_chan, p_ev])
            context.extend(add_zeros)
            context = np.array(context)
            boarder = True            
        if not boarder:
            context = buff_channel[min_chan:max_chan, p_ev]
            context = np.array(context)
       
        # Compute the spatio-temporal context
        context[context > 0] = np.exp(-(t_ev-context[context > 0].astype(np.float64))/tau)
        final_context.extend(context)
    return final_context


  
def features_learning(events, n_layers, nb_chan=30, delta_chan=5,\
                      nb_centers=5, tau=0.4, adaptative=False,\
                      rr_indicators=None, fast=False,\
                      events_clf='kmeans'):
    """ Extract features by defining spatio-temporal context for each event

    Parameters
    -------
    events : 4xN matrix for N events(ti,pi,ci,si)
        ti: time
        pi: polarity
        ci: channel
        si: signal id
    n_layers: Number of layers for HOTS algorithm
    nb_chan: Total number of channels sampling events data
    nb_centers : Number of centers for the Kmeans/GMM clustering
    delta_chan : window of channels surrouding event to compute context
    tau : exponential decay
    all_centers : centers found in features_extraction_learning
    fast : Compute context takes less polarities into account

    Returns
    -------
    Events containing final polarities
        ti: time
        pi: new polarity
        ci: channel
        si: signal id

    """
        
    # Define the variable current_events with the input events
    # Set negative polarity to 0 to facilitate future processes
    current_events = events.copy()
    current_events[current_events[:, 1] < 0, 1] = 0
    all_centers = []
    
    for j in range(n_layers):
        # For each layer define the number of polarities and define size of buff_channel
        # Set value of the channels (delta_chan) and time (tau) window for each
        # layer
    
        nb_polarities = len(np.unique(current_events[:, 1]))
        buff_channel = np.zeros((nb_chan, nb_polarities))
       
        delta_chan = (j+1)*delta_chan
        nb_centers = (j+1)*nb_centers
                
        if fast:
            final_context = np.zeros((len(current_events)-nb_chan, delta_chan*2)) 
        else:   
            final_context = np.zeros((len(current_events)-nb_chan, delta_chan*2*nb_polarities))
            
            
        #Initialize buff_channel with the first nb_channel even
        for k in range(nb_chan): 
            if np.isnan(current_events[k, 1]):
                continue
            t_ev = current_events[k, 0]    
            p_ev = int(current_events[k, 1])
            chan_ev = int(current_events[k, 2])
            buff_channel[chan_ev, p_ev] = t_ev       
            
        for i in range(nb_chan, len(current_events)):
            t_ev = current_events[i, 0]
            
            if adaptative:
                id_sig = current_events[i, 3]
                tau = rr_indicators[int(id_sig)]
            
            context = context_compute(current_events[i, :], nb_chan, tau,\
                                      delta_chan, buff_channel, fast)
           
            try:
                final_context[i-nb_chan, :] = context
            except:
                pass
    
        #Clustering using Guassian Mixture or Kmeans
        if events_clf == 'gmm':
            gaussian_mix = GaussianMixture(n_components=nb_centers)
            gaussian_mix.fit(np.array(final_context))
            all_centers.append(gaussian_mix)
            labels = gaussian_mix.predict(final_context)
        elif events_clf == 'kmeans':
            kmeans = KMeans(nb_centers, init = 'k-means++', random_state= 0, n_init= 10, verbose=0).fit(np.array(final_context))
            centers = kmeans.cluster_centers_
            labels =  kmeans.labels_ 
            all_centers.append(centers)
        current_events[nb_chan:, 1] = labels
        
    return current_events, all_centers, final_context


def context_of_signal(events, n_layers, nb_chan=50, delta_chan=5,\
                      nb_centers=5, tau=0.4, adaptative=False,\
                      rr_indicators=None, fast=False):
        
    # Define the variable current_events with the input events
    # Set negative polarity to 0 to facilitate future processes
    current_events = events.copy()
    current_events[current_events[:, 1] < 0, 1] = 0
    all_centers = []
    
    for j in range(n_layers):
        # For each layer define the number of polarities and define size of buff_channel
        # Set value of the channels (delta_chan) and time (tau) window for each
        # layer
    
        nb_polarities = len(np.unique(current_events[:, 1]))
        buff_channel = np.zeros((nb_chan, nb_polarities))
       
        delta_chan = (j+1)*delta_chan
        nb_centers = (j+1)*nb_centers
                        
        if fast:
            final_context = np.zeros((len(current_events)-nb_chan, delta_chan*2)) 
        else:   
            final_context = np.zeros((len(current_events)-nb_chan, delta_chan*2*nb_polarities))
            
            
        #Initialize buff_channel with the first nb_channel even
        for k in range(nb_chan): 
            if np.isnan(current_events[k, 1]):
                continue
            t_ev = current_events[k, 0]    
            p_ev = int(current_events[k, 1])
            chan_ev = int(current_events[k, 2])
            buff_channel[chan_ev, p_ev] = t_ev       
            
        for i in range(nb_chan, len(current_events)):
            t_ev = current_events[i, 0]
            
            if adaptative:
                id_sig = current_events[i, 3]
                tau = rr_indicators[int(id_sig)]
            
            context = context_compute(current_events[i, :], nb_chan, tau,\
                                      delta_chan, buff_channel, fast)
           
            try:
                final_context[i-nb_chan, :] = context
            except:
                pass
    return final_context

def centers_from_context(final_context, nb_centers):
    kmeans = MiniBatchKMeans(nb_centers, init = 'k-means++', random_state= 0, n_init= 10, verbose=0, batch_size=20).fit(np.array(final_context))
    centers = kmeans.cluster_centers_
    return centers









def context_classif(context, centers):
    """ Compute the minimum distance between the incoming context and the centers found during training

            Input: context, centers 
                    context: 1xN matrix context of incomming event with N channels
                    centers: Centers computed by Kmeans during training 
                                    MxN matrix for M centers (classes) and N channels

            Output: polarity
                    polarity: Integer corresponding to the class
    """
    context = np.array(context)
    centers = np.array(centers)

    try:
        polarity = np.argmin(np.sqrt(((context-centers)**2).sum(axis=1)))
    except AttributeError:
        context = np.array(list(context)).astype('float')

        polarity = np.argmin(np.sqrt(((context-centers)**2).sum(axis=1)))

    return polarity

def features_classification(events, n_layers, nb_chan=50, delta_chan=5,\
                            nb_centers=5, tau=0.4, adaptative=False,\
                            rr_indicators=None, fast=False,\
                            all_centers=None, events_clf='kmeans'):
    """ Extract features by defining spatio-temporal context for each event

    Parameters
    -------
    events : 4xN matrix for N events(ti,pi,ci,si)
        ti: time
        pi: polarity
        ci: channel
        si: signal id
    n_layers: Number of layers for HOTS algorithm
    nb_chan: Total number of channels sampling events data
    nb_centers: Number of centers for the Kmeans/GMM clustering
    delta_chan: window of channels surrouding event to compute context
    tau: exponential decay
    all_centers: centers found in features_extraction_learning
    fast: Compute context takes less polarities into account

    Returns
    -------
    Events containing final polarities
        ti: time
        pi: new polarity
        ci: channel
        si: signal id
    """
    
    current_events = events.copy()
    current_events[current_events[:, 1] < 0, 1] = 0

    for j in range(n_layers):
        nb_polarities = len(np.unique(current_events[:, 1]))
        buff_channel = np.zeros((nb_chan, nb_polarities))
        delta_chan = (j+1)*delta_chan
        nb_centers = (j+1)*nb_centers
        
        if fast:
            final_context = np.zeros((len(current_events)-nb_chan, delta_chan*2)) 
        else:   
            final_context = np.zeros((len(current_events)-nb_chan, delta_chan*2*nb_polarities))
            
        for k in range(nb_chan):
            # Initialize buff_channel with the first nb_channel events
            if np.isnan(current_events[k, 1]):
                continue
            t_ev = current_events[k, 0]
            p_ev = int(current_events[k, 1])
            chan_ev = int(current_events[k, 2])
            buff_channel[chan_ev, p_ev] = t_ev    
            
        for i in range(nb_chan, len(current_events)):
            t_ev = current_events[i, 0]
            
            if adaptative:
                id_sig = current_events[i, 3]
                tau = rr_indicators[int(id_sig)]
            
            context = context_compute(current_events[i, :], nb_chan, tau, delta_chan, buff_channel, fast)
            final_context[i-nb_chan, :] = context

            if events_clf == 'gmm':
                gaussian_mix = all_centers[j]
                pol = gaussian_mix.predict(np.array(context).reshape(1, -1))
            elif events_clf == 'kmeans':
                centers = all_centers[j]
                pol = context_classif(final_context[i-nb_chan, :], centers)

            current_events[i, 1] = pol
        
    return current_events, final_context














def wrap_events(events):
    
    list_events = []
    index_id_s = []
    
    ids = np.unique(events[:,3])
    imin = 0
    for id_ in ids[1:]:
        index_id = np.argwhere(events[:,3] == id_)[:,0][0]
        index_id_s.append(index_id)
        imax = index_id-1
        list_events.append(events[imin:imax,:])
        imin = index_id
    list_events.append(events[imin:,:])
    
    index_id_s = np.array(index_id_s)
    i_single_event = np.argwhere(index_id_s[1:] - index_id_s[:-1] == 1)
    if len(i_single_event) > 0:
        print('!!! WARNING !!! Single event per ecg signal')
        
    return list_events

def features_creation(all_events, nb_centers_f, density=True):
    """ Create a list of signatures (features vector) from a matrix of events
    	
    Parameters
    -------
    all_events: Nx4 matrix matrix for N events(ti,pi,ci,si)
            ti: time
            pi: polarity
            ci: channel
            si: signal id
    nb_centers_f: Number of features used to discriminate events
        
    Returns
    -------
    features : list of  m signatures (features vector). Each signature represents one signal.
        
    """    
    list_events = wrap_events(all_events)
    features = []
    for i in range(len(list_events)):
        events = list_events[i]
        features.append(np.histogram(events[:, 1], range(nb_centers_f+1), density=density)[0])
    
    return features















def classifier(features_train, y_train, features_test, clf='cn', class_weight=None):
    """ Create a list of signatures (features vector) from a matrix of events

    Parameters
    -------
    features_train: Training vectors
    y_train: class labels
    features_test: Test vectors
    name : name of the classifier 
        "svm" uses Support Vector Classification
        "cn" find the closest signature

    Returns
    -------
    pred : Class labels for samples in test_features
    rate : (Sensitivity : recognition rate of the class "A", Specificity : recognition rate of the class "N")
    """

    pred = []
    if(clf == "svm"):
        svc = SVC(class_weight=class_weight, probability=True)
        parameters = {'gamma' : np.linspace(1, 20, 10), 
                      'C'     : np.linspace(1,20,10), 
                      'kernel': ['rbf']}
        clf = GridSearchCV(svc, 
                           parameters, 
                           cv=5, 
                           n_jobs=-1, 
                           return_train_score=True, 
                           scoring='roc_auc')
#        clf = svm.SVC(C=C,gamma=gamma)
        clf.fit(features_train, y_train)                                      
        pred = clf.best_estimator_.predict(features_test)
        prob = clf.best_estimator_.predict_proba(features_test)[:, 1]
        
    if(clf == "knn"):
        svc = KNeighborsClassifier()
        parameters = {'n_neighbors': [5, 21, 31, 41],
                      'weights': ['uniform'],
                      'metric': ['manhattan'],
                      'leaf_size': [30]}
        clf = GridSearchCV(svc, 
                           parameters,
                           cv=5, 
                           n_jobs=-1, 
                           return_train_score=True)
#        clf = svm.SVC(C=C,gamma=gamma)
        clf.fit(features_train, y_train)                                      
        pred = clf.best_estimator_.predict(features_test)
        prob = clf.best_estimator_.predict_proba(features_test)[:, 1]
        
    if(clf=='rf'):
        randomforest = RandomForestClassifier(class_weight=class_weight)
        parameters = {'n_estimators': range(10, 110, 10), 
                      'max_depth': range(5, 15, 2), 
                      'max_leaf_nodes': range(10, 50, 5)}
        clf = RandomizedSearchCV(estimator=randomforest, 
                                 param_distributions=parameters, 
                                 cv=5, 
                                 n_jobs=-1, 
                                 return_train_score=True, 
                                 scoring='accuracy')
        clf.fit(features_train, y_train)
        pred = clf.best_estimator_.predict(features_test)
        prob = clf.best_estimator_.predict_proba(features_test)[:, 1]
        
    if(clf == "lgr"):
        clf = LogisticRegression()
        clf.fit(features_train, y_train)                                      
        pred = clf.predict(features_test)
    
    
    
    if(clf == "cn"):
        for i in range(len(features_test)):
            index = context_classif(np.array(features_test[i]), np.array(features_train))
            pred.append(y_train[index])
        pred = np.array(pred)

    if(clf == "mlp"):
        mlp = MLPClassifier(max_iter = 1000, early_stopping=False)
        parameters = {'alpha': 10.0 ** -np.arange(3, 7), 
                      'hidden_layer_sizes':[(25,), (30,), (20,), (35,)], 
                      'learning_rate': ['constant'], 
                      'activation': ['relu'],
                      'solver': ['adam']}
        clf = RandomizedSearchCV(estimator=mlp, 
                                 param_distributions=parameters, 
                                 cv=5, 
                                 n_jobs=-1, 
                                 n_iter=10, 
                                 return_train_score=True, 
                                 verbose=0)
        clf.fit(features_train, y_train)                                      
        pred = clf.predict(features_test)
        prob = clf.predict_proba(features_test)[:, 1]
        

    return pred, pd.DataFrame(clf.cv_results_), prob