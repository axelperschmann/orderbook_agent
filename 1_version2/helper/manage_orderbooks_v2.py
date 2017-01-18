from tqdm import tqdm
import pandas as pd
import numpy as np
import gzip
import json
import math
import matplotlib.pyplot as plt
import time
from datetime import datetime
from IPython.display import display

def get_ask(orderbook):
    assert isinstance(orderbook, dict)
    assert 'asks' in orderbook.keys() and isinstance(orderbook['asks'], pd.DataFrame) and len(orderbook['asks']) > 0
    return orderbook['asks'].index.values[0]

def get_bid(orderbook):
    assert isinstance(orderbook, dict)
    assert 'bids' in orderbook.keys() and isinstance(orderbook['bids'], pd.DataFrame) and len(orderbook['bids']) > 0
    return orderbook['bids'].index.values[0]

def get_center(orderbook):
    assert isinstance(orderbook, dict)
    assert 'asks' in orderbook.keys() and isinstance(orderbook['asks'], pd.DataFrame) and len(orderbook['asks']) > 0
    assert 'bids' in orderbook.keys() and isinstance(orderbook['bids'], pd.DataFrame) and len(orderbook['bids']) > 0
    
    return log_mean(orderbook['asks'].index.values[0], orderbook['bids'].index.values[0])

def log_mean(x, y):
    assert isinstance(x, int) or isinstance(x, float), 'Bad value: {}'.format(x)
    assert isinstance(y, int) or isinstance(y, float), 'Bad value: {}'.format(y)
    
    if x == y:
        return x
    return (x - y) / (math.log(x) - math.log(y))



def load_orderbook_snapshot(infile, verbose=True, first_line=None, last_line=None):
    assert isinstance(infile, str)
    assert isinstance(verbose, bool)
    assert (isinstance(first_line, int) and first_line>=0) or not first_line
    assert (isinstance(last_line, int) and last_line>1) or not last_line
    
    data = []
    with open(infile, "r+") as f:
        read_data = f.readlines()

    line_range = range(len(read_data)) 
    if last_line:
        line_range = line_range[:last_line]
    if first_line:
        line_range = line_range[first_line:]
       
    for line in tqdm(line_range):
        dictionary = json.loads(read_data[line])
        
        
        bids = pd.DataFrame.from_dict(dictionary['bids'])[::-1]
        asks = pd.DataFrame.from_dict(dictionary['asks'])
        
        # Must undo float to string conversion, which was necessary to prevent rounding
        # issues, occuring when DataFrame.to_dict() is called
        bids.index = bids.index.astype(float)
        bids['Amount'] = bids.Amount.values.astype(float)
        asks.index = asks.index.astype(float)
        asks['Amount'] = asks.Amount.values.astype(float)
        
        orderbook = {
            'timestamp': dictionary['timestamp'],
            'bids': bids,
            'asks': asks
        }
        
        bids = orderbook['bids']  #.sort_index()
        data.append(orderbook)

    if verbose:
        print("Loaded {} orderbooks from file '{}'.".format(len(data), infile))
    return data


def extract_orderbooks_for_one_currencypair(datafiles, currency_pair, outfile, overwrite=True, range_factor=None
, pricelevel_precision=2, verbose=True):
    assert len(datafiles)>0
    assert isinstance(currency_pair, str)
    assert isinstance(outfile, str)
    assert isinstance(overwrite, bool)
    assert isinstance(range_factor, float) or isinstance(range_factor, int) or not range_factor
    if range_factor:
        assert range_factor > 1, "range_factor must be larger than 1, not '{}'".format(range_factor)
    assert isinstance(pricelevel_precision, int)
    assert pricelevel_precision>=0
    assert isinstance(verbose, bool)
    
    if overwrite:
        filemode = "wb"
        if verbose:    
            print("Orderbook content will be written to '{}'".format(outfile))
    else:
        filemode = "ab"
        if verbose:    
            print("Orderbook content will be appended to '{}'".format(outfile))
    
    with open(outfile, filemode) as f_out:
            
        for fullpath in tqdm(datafiles):
            with gzip.open(fullpath, 'r') as f_in:
                df = json.load(f_in)
                if df['orderbook_' + currency_pair].keys()[0] == 'error':
                    continue
                
                timestamp = df['timestamp'][:16] # cut off milliseconds
                df = df['orderbook_' + currency_pair]
            if df.keys()[0] == 'error':
                # Ignore empty, erroneous orderbooks.
                # First time this message occured: 'orderbook_USDT_BTC', u'2017-01-10T12:34:02.126242'
                print("Skipped {} at t={} do to contained 'error' message.".format('orderbook_' + currency_pair, timestamp))
                continue
                
            # extract all ask orders
            asks = pd.DataFrame(df['asks'], columns=['Price', 'Amount'])
            asks['Price'] = pd.to_numeric(asks['Price']).round(decimals=pricelevel_precision)
            asks = asks.groupby('Price', as_index=False).sum()
            asks = asks.set_index(asks.Price.values).drop("Price", axis=1)
            
            # extract all bid orders
            bids = pd.DataFrame(df['bids'], columns=['Price', 'Amount'])
            bids = bids.append(pd.DataFrame([[705.45, 0.4]], columns=['Price', 'Amount']))
            
            bids['Price'] = pd.to_numeric(bids['Price']).round(decimals=pricelevel_precision)
            bids = bids.groupby('Price', as_index=False).sum()[::-1]
            bids = bids.set_index(bids.Price.values).drop("Price", axis=1)

            if asks.index.values[0] == bids.index.values[0]:
                # Due to rounding issues (parameter pricelevel_precision) it can happen that ask and bid
                # are equal (=zero spread). We must take care of this problem by 'matching' (=fulfilling)
                # corresponding orders.
                asks_vol = asks.Amount.values[0]
                bids_vol = bids.Amount.values[0]
            
                if asks_vol > bids_vol:
                    asks.iloc[0] -= bids_vol
                    bids = bids.drop(bids.index[0])
                else:
                    bids.iloc[0] -= asks_vol
                    asks = asks.drop(asks.index[0])

            center = round(log_mean(asks.index.values[0], bids.index.values[0]),
                           pricelevel_precision+2)

            if range_factor:
                # limited price range relative to center_log or norm_Price
                asks = asks[asks.index <= center * range_factor].dropna()
                bids = bids[bids.index >= center / range_factor].dropna()
            
            # must convert floats to unicode to prevent precision loss when
            # executing DataFrame.to_dict() :
            # e.g. index: 706.17 -> 706.16999999999996
            # e.g. Amount: 0.5283784 -> 0.052837839999999997
            
            asks.index = asks.index.map(unicode)
            asks['Amount'] = asks.Amount.map(unicode)
            bids.index = bids.index.map(unicode)
            bids['Amount'] = bids.Amount.map(unicode)            

            obj = {'asks': asks.to_dict(),
                   'bids': bids.to_dict(),
                   'timestamp': timestamp}

            f_out.write(json.dumps(obj) + "\n")
    
    if verbose:
        if overwrite:
            print("Successfully created file '{}'".format(outfile))
        else:
            print("Successfully appended {} lines to file '{}'".format(len(datafiles), outfile))