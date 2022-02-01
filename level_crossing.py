# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:01:30 2022

@author: RamziAbdelhafidh
"""

import numpy as np

def events_creation(times, sig, ids, levels):
    """ Create +/- events

    Parameters
    -------
    times: time of the signal
    sig: amplitude of the signal
    ids: signal id
    levels: Resampled amplitude axis

    Returns
    -------
    Event created 4xN matrix for N events(ti,pi,ci,si)
        ti: time
        pi: polarity
        ci: channel
        si: signal id

    """
    events = []
    ind_level = np.argsort(np.abs(sig[0]-levels))
    ind_level = np.max(ind_level[0:2])
    id_sig = ids[0]
    event_count = 0
    for i in range(1, len(sig)):
        new_id_sig = ids[i]

        if id_sig != new_id_sig:
            if event_count < 2:
                for k in range(abs(id_sig - new_id_sig)):
                    print('!!! WARNING !!! Levels step is too low OR signal flat',
                          ids[i-(k+1)], ')')
                    # Fake event to avoid 0 event per ecg signal
                    events.append([times[i-(k+1)], 1, levels[0], ids[i-(k+1)]])
                    events.append([times[i-(k+1)], 1, levels[0], ids[i-(k+1)]])

            event_count = 0
            id_sig = new_id_sig

        current_val = sig[i]
        level_plus = levels[ind_level]
        level_moins = levels[ind_level-1]
        if current_val > level_plus:
            current_ind = ind_level
            ind_level = np.argsort(np.abs(current_val-levels))
            level_crossed = np.min(ind_level[0:2])
            ind_level = np.max(ind_level[0:2])
            for inter_level in range(current_ind, level_crossed+1):
                event_count += 1
                events.append([times[i], 1, inter_level, ids[i]])
        elif current_val < level_moins:
            current_ind = ind_level-1
            ind_level = np.argsort(np.abs(current_val-levels))
            level_crossed = np.max(ind_level[0:2])
            ind_level = np.max(ind_level[0:2])
            for inter_level in range(current_ind, level_crossed-1, -1):
                event_count += 1
                events.append([times[i], -1, inter_level, ids[i]])
        else:
            level_crossed = ind_level # ind_level - 1 ?

    if event_count < 2:
        print('!!! WARNING !!! Levels step is too low OR signal is flat (id', ids[i-(k+1)], ')')
        # Fake event to avoid 0 event per ecg signal
        events.append([times[i-(k+1)], 1, levels[0], ids[i-(k+1)]])
        events.append([times[i-(k+1)], 1, levels[0], ids[i-(k+1)]])

    events = np.array(events)
    events[events[:, 1] < 0, 1] = 0

    return events

def level_crossing_fusion(time_sig, sig, step, level_min,
                          level_max, sig_num, show_levels=True, linear=False,
                          n_interp=20, n_levels=200):
    """ Resample the signal along amplitude axis and create +/- events
    Input: t, sig, step, level_max
    time_sig: time of the signal
    sig: amplitude of the signal
    step: steps between two levels
    level_min: lowest value of a level
    level_max: highest value of a level
    linear: if True signal is normalized between 0 and 1,
    and resampled in n_levels
    show_levels: (optional) Boolean to dispay figures
    Output: Event created 3xN matrix for N events(ti,pi,ci, si)
            ti: time
            pi: polarity
            ci: channel
            si: sig_num
    """
#    time_sig = a
#    sig = b 
#    level_max = 0.2E6
#    level_min = -0.2E6
#    step = 10000
    if linear:
        new_t = np.linspace(time_sig[0], time_sig[-1], len(time_sig)*n_interp)
        sig = np.interp(new_t, time_sig, sig)
        time_sig = new_t.copy()
        # Signal between [0,1]
        sig = (sig-np.min(sig))/(np.max(sig)-np.min(sig))
        # Define levels
        levels = np.linspace(np.min(sig), np.max(sig), n_levels)
    else:
        if level_min < 0:
            levels_positives = np.arange(0, level_max, step)
            levels_negatives = -levels_positives[1:]
            levels = np.hstack((levels_negatives[::-1], levels_positives))
        else:
            levels = np.arange(level_min, level_max, step)

   
        
    #Initialisation: Find first level
    events = []
    data0 = sig[0]
    ind_level = np.argsort(np.abs(data0-levels))

    level_plus = levels[np.max(ind_level[0:2])]
    level_plus = levels[np.min(ind_level[0:2])]

    ind_level = np.max(ind_level[0:2])
#    sig_num = 3
    #Detect cross leveling
    for i in range(1, len(sig)):
        current_val = sig[i]

        level_plus = levels[ind_level]
        level_moins = levels[ind_level-1]
        if current_val > level_plus:
            level_crossed = ind_level
            
            events.append([time_sig[i], 1, level_crossed, sig_num])
            ind_level = ind_level+1
        if current_val < level_moins:
            level_crossed = ind_level-1
            events.append([time_sig[i], -1, level_crossed, sig_num])
            ind_level = ind_level-1
        else:
            level_crossed = ind_level
    if show_levels:
        events = np.array(events)

        events = events[np.abs(events[:, 1]) > 0]
        color_ = np.array(events[:, 1])
        import matplotlib.pyplot as plt
        
        fig, ax2 = plt.subplots()
       
      #  plt.plot(time_sig, sig)
        for level in enumerate(levels):
            plt.plot([time_sig[0], time_sig[-1]], [level[1], level[1]], "g")
        plt.title("Level-crossing")
        
        for level in enumerate(levels):
            ax2.plot([time_sig[0], time_sig[-1]], [level[1], level[1]], "g")
            ax2.scatter(events[:, 0], levels[np.int32(events[:, 2])], c=color_)


    return np.array(events)


def levels_creation(level_min=-2, level_max=2, level_step=0.03):
    """ Create a resampled amplitude axis to define events

    Parameters
    ----------
    level_step: Level's resolution
    level_min: Level's minimum amplitude
    level_max: Level's maximum amplitude

    Returns
    -------
    levels: Resampled amplitude axis
    """
    level_max = level_max
    level_min = level_min
    level_step = level_step
    levels = np.arange(level_min, level_max, level_step)

    return levels
