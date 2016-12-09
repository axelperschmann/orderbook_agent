from tqdm import tqdm
import pandas as pd
import numpy as np
import gzip
import json
import math
import matplotlib.pyplot as plt

def discretize_orderbook(data, range_factor=1.3, num_samples=51):
    assert isinstance(data, pd.DataFrame)
    assert isinstance(num_samples, int)
    assert num_samples > 1, "num_samples must be larger than 1, not '{}'".format(num_samples)
    assert num_samples%2 == 1, "please select uneven number of samples, not '{}'".format(num_samples)
    assert isinstance(range_factor, float) or isinstance(range_factor, int)
    assert range_factor > 1, "range_factor must be larger than 1, not '{}'".format(range_factor)

    
    sample_idx = np.logspace(-1, 1, base=range_factor, num=num_samples)
    indices = []
    for p in sample_idx:
        # sample full orderbook for every specified norm_Price
        if p < 1:
            # print("len", data[data.norm_Price>p].index[0])
            indices.append(data[data.norm_Price>p].index[0])
        elif p > 1:
            # print("len", len(data[data.norm_Price<p]))
            # print("x", data[data.norm_Price<p].index[-1])
            indices.append(data[data.norm_Price<p].index[-1])
        else:
            indices.append(data[data.norm_Price==p].index[0])
            
    new_data = data.iloc[indices].copy()
    new_data.reset_index(inplace=True, drop=True)
    return new_data

def log_mean(x, y):
    assert isinstance(x, int) or isinstance(x, float)
    assert isinstance(y, int) or isinstance(y, float)
    
    if x == y:
        return x
    return (x - y) / (math.log(x) - math.log(y))

def place_order(orderbook, amount, trade_history=None, limit=None, verbose=True):
    assert isinstance(orderbook, pd.DataFrame)
    assert isinstance(amount, float) or isinstance(amount, int)
    assert amount != 0
    assert isinstance(limit, float) or isinstance(limit, int) or not limit
    if limit:
        assert limit > 0
    assert isinstance(verbose, bool)
    
    df = orderbook.copy()
    
    if not trade_history:
        trade_history = {}
    trade_summary = {}
    info = {}
    
    info['cashflow'] = 0
    info['amount_fulfilled'] = 0
    info['limit'] = limit
    
    if amount > 0:
        # buy from market
        order_type = 'ask'
    elif amount < 0:
        # sell to market
        order_type = 'bid'
    # reduce orderbook to relevant type of orders only
    df = df[df.Type==order_type]
    
    # adjust dataFrame to our own trade_history. Simulate influence caused by our previous trades.
    for hist in trade_history.keys():
        pd_row = df[df.Price == float(hist)]
        new_amount = (pd_row.Amount - trade_history[hist])

        df.loc[pd_row.index, 'Amount']  = new_amount
        df.loc[pd_row.index, 'Volume']  = new_amount * pd_row.Price  # unneccesary?!
        # not modified, but not of interest for now: VolumeAcc and norm_Price
    df = df[df.Amount > 0]
    
    if order_type == 'ask':
        # buy from market
        ask = df.iloc[0].Price
        for pos in range(len(df)):
            order = df.iloc[pos]
            if limit and order.Price > limit:
                # Price limit exceeded, stop trading now!
                break
                    
            if amount - order.Amount >= 0:
                purchase_amount = order.Amount
            else:
                purchase_amount = amount

            # Fullfill trade
            if str(order.Price) in trade_summary.keys():
                trade_summary[str(order.Price)] += purchase_amount
            else:
                trade_summary[str(order.Price)] = purchase_amount
                
            if str(order.Price) in trade_history.keys():
                trade_history[str(order.Price)] += purchase_amount
            else:
                trade_history[str(order.Price)] = purchase_amount
            
            info['cashflow'] -= purchase_amount * order.Price
            info['amount_fulfilled'] += purchase_amount
            amount -= purchase_amount
            info['worst_price'] = order.Price
            if amount == 0:
                break
        info['slippage'] = (ask * info['amount_fulfilled']) + info['cashflow']
        
    elif order_type == 'bid':
        # sell to market
        bid = df.iloc[-1].Price
        for pos in range(len(df)-1, 0, -1):
            order = df.iloc[pos]
            if limit and order.Price < limit:
                # Price limit exceeded, stop trading now!
                break

            if amount + order.Amount <= 0:
                sell_amount = - order.Amount
            else:
                sell_amount = amount
                
            # Fullfill trade
            if str(order.Price) in trade_summary.keys():
                trade_summary[str(order.Price)] += sell_amount
            else:
                trade_summary[str(order.Price)] = sell_amount
            info['cashflow'] -= sell_amount * order.Price
            info['amount_fulfilled'] += sell_amount
            
            amount -= sell_amount
            info['worst_price'] = order.Price
            if amount == 0:
                break
        info['slippage'] = abs((bid * info['amount_fulfilled']) + info['cashflow'])

    info['amount_unfulfilled'] = amount
    info['trade_summary'] = trade_summary
    info['trade_history'] = trade_history
    
    if verbose:
        if order_type == 'ask':
            print("Bought {:.4f}/{:.4f} shares for {}".format(info['amount_fulfilled'],
                                                          info['amount_fulfilled']+info['amount_unfulfilled'],
                                                          info['cashflow']))
        elif order_type == 'bid':
            print("Sold {:.4f}/{:.4f} shares for {}".format(info['amount_fulfilled'],
                                                          info['amount_fulfilled']+info['amount_unfulfilled'],
                                                          info['cashflow']))
            
    return info


def extract_orderbooks_for_one_currencypair(datafiles, currency_pair, outfile, overwrite=True, range_factor=None
, num_samples=None, float_precision=2, verbose=True):
    assert len(datafiles)>0
    assert isinstance(currency_pair, str)
    assert isinstance(outfile, str)
    assert isinstance(overwrite, bool)
    assert isinstance(num_samples, int) or not num_samples, "num_samples must be an integer or None, not {}".format(type(num_samples))
    if num_samples:
        assert range_factor, 'Please specify range_factor for discretization of orderbook'
        assert num_samples > 1, "num_samples must be larger than 1, not '{}'".format(num_samples)
        assert num_samples%2 == 1, "please select uneven number of samples, not '{}'".format(num_samples)
    assert isinstance(range_factor, float) or isinstance(range_factor, int) or not range_factor
    if range_factor:
        assert range_factor > 1, "range_factor must be larger than 1, not '{}'".format(range_factor)
    assert isinstance(float_precision, int)
    assert float_precision>0
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

            # extract all ask orders
            price  = [round(float(x[0]), float_precision) for x in df['orderbook_' + currency_pair]['asks']]
            lowest_ask = price[0]
            amount = [float(x[1]) for x in df['orderbook_' + currency_pair]['asks']]
            volume = [round(float(x[0]), float_precision) * float(x[1]) for x in df['orderbook_' + currency_pair]['asks']]
            asks = pd.DataFrame({'Amount': pd.Series(amount),
                                 'Price': price,
                                 'Volume':volume})
            # group by rounded Price
            asks = asks.groupby('Price').sum()
            asks['Type'] = 'ask'
            asks['VolumeAcc'] = 0
            asks.reset_index(inplace=True)

            # extract all bid orders
            price  = [round(float(x[0]), float_precision) for x in df['orderbook_' + currency_pair]['bids']]
            highest_bid = price[0]
            amount = [float(x[1]) for x in df['orderbook_' + currency_pair]['bids']]
            volume = [round(float(x[0]), float_precision) * float(x[1]) for x in df['orderbook_' + currency_pair]['bids']]
            bids = pd.DataFrame({'Amount': pd.Series(amount),
                                 'Price': price,
                                 'Volume':volume})
            # group by rounded Price
            bids = bids.groupby('Price').sum()
            bids['Type'] = 'bid'
            bids['VolumeAcc'] = 0
            bids.reset_index(inplace=True)

            # compute log_mean (center between lowest_ask and highest_bid)
            center_log = log_mean(lowest_ask, highest_bid)
            # spread = lowest_ask - highest_bid

            center = pd.DataFrame({'Amount': 0,
                                 'Price': center_log,
                                 'Type':'center',
                                 'Volume':0,
                                 'VolumeAcc': 0}, index=[0])


            # concat ask, center and bid DataFrames
            df2 = pd.concat([asks, bids, center])
            df2 = df2.sort_values("Price")
            df2.index.rename(df['timestamp'], inplace=True)
            df2.reset_index(inplace=True, drop=True)

            # compute accumulated order volume
            df2['VolumeAcc'].loc[df2.Type=='bid'] = (df2[df2.Type=='bid'].Volume)[::-1].cumsum().values[::-1]
            df2['VolumeAcc'].loc[df2.Type=='ask'] = (df2[df2.Type=='ask'].Volume).cumsum().values

            # normalization
            df2['norm_Price'] = df2.Price / center_log

            if num_samples:
                # discretize orderbook
                df2 = discretize_orderbook(data=df2, range_factor=range_factor, num_samples=num_samples)
            if range_factor:
                # limited price range relative to center_log or norm_Price
                df2 = df2[(df2['norm_Price'] <= range_factor) & (df2['norm_Price'] >= range_factor**-1)]
            
            obj = {'dataframe': df2.to_json(double_precision=15), 'timestamp': df['timestamp']}

            f_out.write(json.dumps(obj) + "\n")
    if verbose:
        if overwrite:
            print("Successfully created file '{}'".format(outfile))
        else:
            print("Successfully appended {} lines to file '{}'".format(len(datafiles), outfile))


def load_orderbook_snapshot(infile, verbose=True):
    assert isinstance(infile, str)
    assert isinstance(verbose, bool)
    
    timestamps = []
    data = []
    with open(infile, "r+") as f:
        read_data = f.readlines()

    for line in tqdm(range(len(read_data))):
        dictionary = json.loads(read_data[line])
        df = pd.read_json(dictionary['dataframe'] , precise_float=True)
        df.sort_index(inplace=True)

        data.append(df)
        timestamps.append( dictionary['timestamp'])

    if verbose:
        print("Loaded Orderbooks: {}".format(len(data)))
    return data, timestamps

def plot_orderbook(data, title, normalized=False, range_factor=None, outfile=None, fileformat='svg'):
    assert isinstance(data, pd.DataFrame)
    assert isinstance(title, str) or isinstance(title, unicode)
    assert isinstance(normalized, bool)
    assert isinstance(range_factor, float) or isinstance(range_factor, int) or not range_factor
    if range_factor:
        assert range_factor > 1, "range_factor must be larger than 1, not '{}'".format(range_factor)
    assert isinstance(outfile, str) or not outfile
    if outfile:
        if outfile[-(len(fileformat)+1):] != '.{}'.format(fileformat):
            outfile = "{}.{}".format(outfile, fileformat)
    assert isinstance(fileformat, str)
    
    
    plt.figure(figsize=(16,8))
    if normalized:
        if range_factor:
            xlim = (1./range_factor, range_factor)
        else:
            xlim = (0, data.norm_Price.values[-1])
        
        bids_lim = data[data.norm_Price>xlim[0]].VolumeAcc.values[0]
        asks_lim = data[data.norm_Price<xlim[1]].VolumeAcc.values[-1]
        y_factor = asks_lim + bids_lim
        
        asks_x = data[data.Type == 'ask'].norm_Price.values
        asks_y = data[data.Type == 'ask'].copy().VolumeAcc / y_factor
        # added .copy(), was it's absence the reason for a warning?
        # A value is trying to be set on a copy of a slice from a DataFrame
        
        bids_x = data[data.Type == 'bid'].norm_Price.values
        bids_y = data[data.Type == 'bid'].copy().VolumeAcc / y_factor    # added .copy()
        
        plt.ylim((0,1))
        
    else:
        center = data[data.Type == 'center'].Price.values[0]        
        if range_factor:
            xlim = (center/range_factor, center*range_factor)
        else:
            xlim = (0, data.Price.values[-1])
        
        bids_lim = data[data.Price>xlim[0]].VolumeAcc.values[0]
        asks_lim = data[data.Price<xlim[1]].VolumeAcc.values[-1]
        y_factor = asks_lim + bids_lim
        
        asks_x = data[data.Type == 'ask'].Price.values
        asks_y = data[data.Type == 'ask'].VolumeAcc
        bids_x = data[data.Type == 'bid'].Price.values
        bids_y = data[data.Type == 'bid'].VolumeAcc
        
        plt.ylim((0,y_factor))

    plt.plot(bids_x, bids_y, color='g', label='VolumeAcc Bid')
    plt.plot(asks_x, asks_y, color='r', label='VolumeAcc Ask')
    plt.fill_between(bids_x, bids_y, 0, color='g', alpha=0.1)
    plt.fill_between(asks_x, asks_y, 0, color='r', alpha=0.1)
    
    plt.xlim(xlim)
    plt.suptitle("{} - center: {:1.4f}".format(title, data[data.Type=='center'].Price.values[0]))
    plt.legend()
    
    if outfile:
        plt.savefig(outfile, format='svg')
        print("Successfully saved'{}'".format(outfile))
        plt.close()
    else:
        plt.show()
        plt.close()
