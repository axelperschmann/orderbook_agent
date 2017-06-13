import numpy as np
import pandas as pd
from scipy import signal

def safe_list_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default
    except KeyError:
        return default

def gauss_window(actions, a_idx, std):
    low = len(actions) - 1 - a_idx
    upp = low + len(actions)
    
    window = signal.gaussian((len(actions)*2)-1, std=std)
    window = window[low:upp]
    window = window / np.sum(window)
    
    return window

def add_features_to_orderbooks(orderbooks, hist):
    direction = orderbooks[-1].get_center() / orderbooks[0].get_center()
    if direction > 1.004:
        direction_disc = 2.
    elif direction > 1.002:
        direction_disc = 1.
    elif direction < 0.996:
        direction_disc = -2.
    elif direction < 0.998:
        direction_disc = -1.
    else:
        direction_disc = 0.
    
    for ob in orderbooks:
        ob.norm_factor = 1
        ob.features = {'spread': ob.asks.index[0]-ob.bids.index[0]}
        
        ts = pd.to_datetime(ob.timestamp)        
        ts_prev = pd.to_datetime(ob.timestamp) - pd.Timedelta('60Min')
    
        market_features = hist.loc[ts_prev:ts, :]
        
        for lookahead in [15,30,45]:
            ob.features["future{}".format(lookahead)] = market_features['future{}'.format(lookahead)][-1]
            ob.features["future{}_disc".format(lookahead)] = market_features['future{}_disc'.format(lookahead)][-1]
        ob.features['spread_disc'] = market_features['spread_disc'][-1]
        
        ob.features['direction'] = direction_disc
        ob.features['direction_float'] = direction
    return orderbooks

def load_and_preprocess_historyfiles(histfiles):
    hist = pd.DataFrame()
    for file in histfiles:
        data = pd.read_csv(file, index_col=0)
        hist = pd.concat([hist, data])
    hist.set_index(keys=pd.to_datetime(hist.index), inplace=True)

    # remove dublicates and unneeded columns
    hist = hist[~hist.index.duplicated(keep='first')]
    hist.drop(["id", 'isFrozen'], axis=1, inplace=True)

    # fill gaps with last observation
    idx = pd.date_range(hist.index[0], hist.index[-1], freq="1min")
    hist = hist.sort_index().reindex(index=idx.sort_values(), method='ffill')
    
    # add more features
    hist['spread'] = hist.lowestAsk - hist.highestBid

    for lookahead in [15, 30, 45]:
        hist["future{}".format(lookahead)] = (hist['last'].shift(-lookahead) / hist['last']) - 1.

    return hist