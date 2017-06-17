from tqdm import tqdm
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

def discretize_hist_feature(hist, feature, test_start_date=None, bins=5):
    df = hist.copy()
    
    if test_start_date is None:
        df_train = df
    else:
        # discretize features only by looking at quantiles of historic data.
        # Avoids lookahead!
        df_train = df[df.index<test_start_date]
    
    splits = [n/bins for n in range(1, bins)]
    quantile = df_train[feature].quantile(splits)

    bin_borders = [-np.inf] + list(quantile.values) + [np.inf]
    
    idx = df.columns.get_loc(feature)
    df.insert(idx+1, '{}_disc'.format(feature), pd.cut(df[feature], bins=bin_borders, labels=False).astype(float))
    
    return df

def addMarketFeatures_toSamples(samples, hist, market_features, 
                                state_variables=None, period_length=15):
    df = samples.copy()
    
    for i, f in tqdm(enumerate(market_features)):
        print(f)
        f_n = "{}_n".format(f)
        if f in df.columns:
            df.drop(f, inplace=True, axis=1)
        if f_n in df.columns:
            df.drop(f_n, inplace=True, axis=1)

        df.insert(loc=2+i, column=f, 
                  value=hist.loc[df.timestamp, f].values,
                 allow_duplicates=True)

        df.insert(loc=df.shape[1], column=f_n,
                  value=hist.loc[df.timestamp + pd.Timedelta(minutes=period_length), f].values,
                  allow_duplicates=True)
        
        #'hist' does not necessarily contain timestamps consistent 
        # with the timestamp of the 'new_state'. Thus, dropna():
        df.dropna(inplace=True)

        if state_variables is not None:
            if f not in state_variables:
                state_variables.append(f)

        if df.isnull().any().any()==True:
            raise ValueError("samples contains nan-values. Propably 'ts' or 'ts_next' of some samples were not present in 'hist'")

    return df


def add_features_to_orderbooks(orderbooks, hist, features=None, reset_features=False):
    
    # if features is None:
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
        ob.norm_factor = 1.
        ts = pd.to_datetime(ob.timestamp)

        if features is not None:
            if reset_features or not hasattr(ob, 'features'):
                ob.features = {}

            for feat in features:
                ob.features[feat] = hist.loc[ts, feat]

        else:
            if reset_features:
                ob.features = {}
            ob.features['spread'] = ob.asks.index[0]-ob.bids.index[0]
            
            ts = pd.to_datetime(ob.timestamp)        
            ts_prev = pd.to_datetime(ob.timestamp) - pd.Timedelta('60Min')
            
            market_features = hist.loc[ts_prev:ts, :]
            
            for lookahead in [15,30,45]:
                ob.features["future{}".format(lookahead)] = market_features['future{}'.format(lookahead)][-1]
                ob.features["future{}_disc".format(lookahead)] = market_features['future{}_disc'.format(lookahead)][-1]
            ob.features['spread_disc'] = market_features['spread_disc'][-1]
            
        ob.features['direction'] = direction
        ob.features['direction_disc'] = direction_disc
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