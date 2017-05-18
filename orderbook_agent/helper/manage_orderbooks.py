from tqdm import tqdm
import pandas as pd
import numpy as np
import gzip
import json
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from datetime import datetime
from IPython.display import display
import itertools

from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import OrderbookTradingSimulator

class OrderbookEpisodesGenerator(object):
    
    def __init__(self, filename, episode_length, slicerange=None):
        assert isinstance(filename, str), "Parameter 'filename' must be of type 'str' and should point to an existing file."
        assert isinstance(episode_length, int) and episode_length>1, "Parameter 'episode_length' must be of type 'int' and larger than 1."
        
        self.filename = filename
        self.episode_length = episode_length
        self.slice = slicerange
    
    def __len__(self):
        with open(self.filename) as f:
            count = sum(1 for line in f)
        
        count = int(count/self.episode_length)
        
        if self.slice is not None:
            count = min(count, self.slice.stop)
            count -= self.slice.start
            count /= self.slice.step
        
        return math.ceil(count)

    def __getitem__(self, index):
        assert isinstance(index, (int, slice))

        if isinstance(index, int):
            start = index * self.episode_length
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self)
            step = index.step or 1

            if start < 0:
                start = len(self) + start
            if stop < 0:
                stop = len(self) + stop

            return OrderbookEpisodesGenerator(filename=self.filename, episode_length=self.episode_length, slicerange=slice(start, stop, step))
            # return [self[idx] for idx in range(start, stop, step)]
        
        with open(self.filename, "r+") as f:
            for _ in range(start):
                 next(f)
            episode = []
            for e in range(self.episode_length):
                line = next(f)
                container = convert_line_to_OrderbookContainer(line)

                episode.append(container)
            return episode

    def __iter__(self):
        with open(self.filename, "r+") as f:
            if self.slice is not None:
                for i in range(self.slice.start * self.episode_length):
                    next(f)

            counter = 0
            while True:
                if self.slice is not None:
                    if counter % self.slice.step != 0:
                        counter += 1
                        for i in range(self.episode_length):
                            next(f)
                        continue
                    if counter == self.slice.stop - self.slice.start:
                        break
                counter += 1

                episode = []
                for e in range(self.episode_length):
                    line = next(f)
                    container = convert_line_to_OrderbookContainer(line)

                    episode.append(container)


                #if len(episode) == self.episode_length:
                yield episode
                # else:
                #     raise StopIteration


def orderbooks_difference(orderbook1, orderbook2):
    bids_diff = self.bids.subtract(other.bids, axis=1, fill_value=0)
    asks_diff = self.asks.subtract(other.asks, axis=1, fill_value=0)

    return OrderbookContainer(timestamp=other.timestamp,
                              bids = bids_diff[bids_diff != 0].dropna(),
                              asks = asks_diff[asks_diff != 0].dropna())


def plot_Q(model, V, T, actions, STATE_DIM=2, outfile=None, outformat=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t in range(T):
        xs = np.arange(V)
        
        ys = np.zeros(V)
        for v in range(V):
            state = np.array([t, v])
            qval = model.predict(state.reshape(1, STATE_DIM))
            ys[v] = actions[np.argmin(qval)]
        
        ax.bar(xs, ys, zs=t, zdir='y', alpha=0.5)

    ax.set_xlabel("time remaining")
    ax.set_ylabel("shares remaining")
    ax.set_zlabel("aggression level")
    plt.title("Q function")
    if outfile:
        if outfile[-3:] != outformat:
            outfile = "{}.{}".format(outfile, outformat)
        plt.savefig(outfile, format=outformat)
        print("Successfully saved '{}'".format(outfile))
    else:
        plt.show()
    plt.close()

def convert_actions_to_limit(episode_windows, actions, lim_stepsize=0.1):
    init_center = episode_windows[0].get_center()
    lim_increments = init_center * (lim_stepsize / 100)

    period_length = int(len(episode_windows) / len(actions))
    decision_points = [period_length*t for t in range(int(len(episode_windows)/period_length))]
    
    limits = []
    for t, timepoint in enumerate(decision_points):
        ob_now = episode_windows[timepoint]
        price_incScale = int(round(ob_now.get_center()/lim_increments, 3))
        limits.append(lim_increments * (price_incScale + actions[t]))
    print("limits", limits)
    return limits


def plot_episode(episode_windows, volume=100, figsize=(8,6), ylim=None,
                 outfile=None, outformat='pdf', intervals=1, kind="avg",
                 legend=True, limits=None, actions=None, lim_stepsize=0.1, vlines=None, hlines=None):
    assert isinstance(episode_windows, list)
    assert type(episode_windows[0]).__name__ == "OrderbookContainer"
    assert isinstance(volume, (int, float))
    assert volume != 0, "parameter 'volume' must not be 0"
    assert isinstance(figsize, tuple) and len(figsize) == 2
    assert (isinstance(ylim, tuple) and len(ylim) == 2) or ylim is None
    assert isinstance(outfile, str) or outfile is None
    assert isinstance(outformat, str)
    assert (isinstance(intervals, int) and intervals > 0) or intervals is None
    assert isinstance(kind, str) and kind in ['avg', 'worst']
    assert isinstance(legend, bool)
    assert isinstance(limits, list) or limits is None
    volume = abs(volume)
    
    initial_center = episode_windows[0].get_center()
    center = []
    ask = []
    bid = []
    price_ask = {}
    price_bid = {}
    for i in range(intervals):
        price_ask[i] = []
        price_bid[i] = []

    timestamps = []

    if isinstance(actions, list) and lim_stepsize is not None:
        limits = convert_actions_to_limit(episode_windows, actions=actions)

    if limits is not None:
        period_length = int(len(episode_windows) / len(limits))
        limit_idx = 0
        limits_y = []
        fig, axs = plt.subplots(nrows=3, figsize=figsize)
        ax = axs[0]
    else:
        fig, ax = plt.subplots(nrows=1, figsize=figsize)

    for o, ob in enumerate(episode_windows):
        if (limits is not None) and (o%period_length == 0):
            #limits_y = limits_y + list(np.repeat(ob.get_ask() * (1. + (limits[limit_idx]/100.)), period_length))
            limits_y = limits_y + list(np.repeat(limits[limit_idx], period_length))
            limit_idx += 1
            #print("x", ob.get_ask() * (1. + (limits[limit_idx]/100.)))

        center.append(ob.get_center())
        ask.append(ob.get_ask())
        bid.append(ob.get_bid())

        for i in range(intervals):
            volume_fraction = 1.*(i+1)/intervals * volume
            if kind == 'avg':
                # Show current average price, when buying specified volume.
                price_ask[i].append(ob.get_current_price(volume_fraction)[0] / volume_fraction)
                price_bid[i].append(ob.get_current_price(-volume_fraction)[0] / volume_fraction)
            elif kind == 'worst':
                # Show current worst price, when buying specified volume.
                price_ask[i].append(ob.get_current_price(volume_fraction)[1])
                price_bid[i].append(ob.get_current_price(-volume_fraction)[1])
        timestamps.append(datetime.strptime(ob.timestamp, '%Y-%m-%dT%H:%M'))
    
    if limits is not None:
        ax.plot(timestamps, limits_y, color='grey', drawstyle='steps-post', label='Limits')
        ots = OrderbookTradingSimulator(orderbooks=episode_windows, volume=volume, tradingperiods=len(limits),
                                             period_length=period_length)

        traded_volume = []
        volume_left = [100]
        costs = []
        avgs = []
        for t, lim in enumerate(limits_y):
            if t == len(limits_y) -1:
                # forced trade at very end of tradingperiod
                lim = None
            ots.adjust_masterbook()
            summary = ots.perform_trade_wrapper(ots.masterbook, limit=lim)
            ots.t +=1
            ots.volume -= summary['volume']

            traded_volume.append(summary['volume'])
            volume_left.append(ots.volume)
            
            cost = 0
            avg = np.nan
            if abs(summary['volume']) > 0:
                avg = abs(summary['cash']) / summary['volume']
                cost = summary['volume'] * (avg - ots.initial_center) / ots.market_slippage
            costs.append(cost)
            avgs.append(avg)
            if abs(ots.volume) < 1e-10:
                break

        axs[1].bar(range(len(traded_volume)), traded_volume, align='edge', alpha=0.7)
        axs[1].set_ylabel("Traded shares")
        ax2 = axs[1].twinx()
        ax2.step(range(len(volume_left)), volume_left, where='mid', color='red', label='volume')
        ax2.set_ylabel("Volume left")
        ax2.set_xlim((-1,len(timestamps)))
        ax2.legend(loc='best')
        axs[1].set_xlabel("t")
        ax2.axhline(0, color='black', alpha=0.5)
        
        axs[2].set_ylim((np.nanmin(avgs), np.nanmax(avgs)))
        axs[2].bar(range(len(avgs)), avgs, align='edge', alpha=0.7, label='avg price')
        axs[2].set_ylabel("avg price")

        ax4 = axs[2].twinx()
        ax4.step(range(len(costs)), np.cumsum(np.array(costs)), where='mid', color='red', alpha=0.6, label='acc costs')
        ax4.step(range(len(costs)), np.cumsum(np.array(costs))/np.cumsum(np.array(traded_volume))*volume, where='mid', color='green', alpha=0.6, label='expected costs')
        ax4.set_ylabel("Costs")
        ax4.legend(loc='best')
        ax4.set_xlim((-1,len(timestamps)))
        if vlines is None:
            vlines = [period_length*t for t in range(int(len(episode_windows)/period_length))]
        axs[2].set_xlabel("actions: {}, costs: {:1.2f}".format(actions, np.array(costs).sum()))
    
    ax.plot(timestamps, center, color='black', label='Center')
    ax.fill_between(timestamps, price_ask[intervals-1], ask, color='red', alpha=0.1)    
    ax.plot(timestamps, ask, color='red', linestyle="--", label='Ask')

    ax.plot(timestamps, bid, color='green', linestyle="--", label='Bid')
    ax.fill_between(timestamps, price_bid[intervals-1], bid, color='green', alpha=0.1)

    if isinstance(vlines, list):
        for vl in vlines:
            ax.axvline(timestamps[vl], color='black', alpha=0.2)
            if limits is not None:
                axs[1].axvline(vl, color='black', alpha=0.2)
                axs[2].axvline(vl, color='black', alpha=0.2)
    if isinstance(hlines, list):
        for hl in hlines:
            ax.axhline(hl)
    
    for i in range (intervals):
        volume_fraction = 1.*(i+1)/intervals
        ax.plot(timestamps, price_bid[i], color='green', alpha=volume_fraction)
        ax.plot(timestamps, price_ask[i], color='red', alpha=volume_fraction)
        
    title = "'{}' Price comparison for a trade volume of {} shares".format(kind, volume)

    if intervals > 1:
        title = "{} (interval size: {})".format(title, 1.*volume/intervals)
        
    ax.set_title(title)
    if vlines is not None:
        ax.set_xticks([timestamps[vl] for vl in vlines])
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlim((timestamps[0] - pd.Timedelta(1, unit='m'), timestamps[-1] + pd.Timedelta(1, unit='m')))
    print(timestamps[-1] + pd.Timedelta(1, unit='m'), timestamps[-1], pd.Timedelta(1, unit='m'))
    
    ax.set_ylabel('Price')
    ax.set_xlabel(episode_windows[0].timestamp)
    if legend:
        ax.legend(loc='best', prop={'size': 6})
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if outfile is None:
        plt.show()
    else:
        if outfile[-len(outformat):] != outformat:
            outfile = "{}.{}".format(outfile, outformat)
        plt.savefig(outfile, format=outformat)
        print("successfully saved '{}'".format(outfile))
    plt.close()

def convert_line_to_OrderbookContainer(line):
    dictionary = json.loads(line)

    bids = pd.DataFrame.from_dict(dictionary['bids'])[::-1]
    asks = pd.DataFrame.from_dict(dictionary['asks'])

    # Must undo float to string conversion, which was necessary to prevent rounding
    # issues, occuring when DataFrame.to_dict() is called
    bids.index = bids.index.astype(float)
    bids['Amount'] = bids.Amount.values.astype(float)
    asks.index = asks.index.astype(float)
    asks['Amount'] = asks.Amount.values.astype(float)

    return OrderbookContainer(dictionary['timestamp'], bids=bids, asks=asks)


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
        
        container = OrderbookContainer(dictionary['timestamp'], bids=bids, asks=asks)
        
        data.append(container)

    if verbose:
        print("Loaded {} orderbooks from file '{}'.".format(len(data), infile))
    return data


def extract_orderbooks_history(datafiles, currency_pair, outfile, overwrite=True):
    history = pd.DataFrame()
    
    for fullpath in tqdm(datafiles):
        with gzip.open(fullpath, 'r') as f_in:
            df = json.load(f_in)
            if df['orderbook_' + currency_pair].keys()[0] == 'error':
                continue
            
            timestamp = df['timestamp'][:16] # cut off milliseconds

            df_history = pd.DataFrame(pd.DataFrame(df['ticker'])[currency_pair]).T
            df_history.index = [timestamp]

            history = pd.concat([history, df_history])
    
    history.index = pd.to_datetime(history.index)
    if outfile is not None:
        history.to_csv(outfile)
        print("Exported history '{}'".format(outfile))

    return history


def extract_orderbooks_for_one_currencypair(datafiles, currency_pair, outfile, overwrite=True,
    range_factor=None, range_volume=None, pricelevel_precision=2, verbose=True):
    assert len(datafiles)>0
    assert isinstance(currency_pair, str)
    assert isinstance(outfile, str)
    assert isinstance(overwrite, bool)
    assert isinstance(range_factor, (float, int)) or not range_factor
    if range_factor:
        assert range_factor > 1, "'range_factor' must be larger than 1, not '{}'".format(range_factor)
    assert isinstance(range_volume, (float, int)) or not range_volume
    if range_volume:
        assert range_volume > 0, "'range_volume' must be a positive 'int' or 'float', given: '{}', {}".format(range_volume, type(range_volume))
    assert isinstance(pricelevel_precision, int) and pricelevel_precision>=0
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
                
            # extract all bid orders
            bids = pd.DataFrame(df['bids'], columns=['Price', 'Amount'])
            bids['Price'] = pd.to_numeric(bids['Price']).round(decimals=pricelevel_precision)
            bids = bids.groupby('Price', as_index=False).sum()[::-1]
            bids = bids.set_index(bids.Price.values).drop("Price", axis=1)
            bids.sort_index(inplace=True, ascending=False)
            
            # extract all ask orders
            asks = pd.DataFrame(df['asks'], columns=['Price', 'Amount'])
            asks['Price'] = pd.to_numeric(asks['Price']).round(decimals=pricelevel_precision)
            asks = asks.groupby('Price', as_index=False).sum()
            asks = asks.set_index(asks.Price.values).drop("Price", axis=1)
            asks.sort_index(inplace=True)
            
            if bids.index.values[0] == asks.index.values[0]:
                # Due to rounding issues (parameter pricelevel_precision) it can happen that ask and bid
                # are equal (=zero spread). We must take care of this problem by 'matching' (=fulfilling)
                # corresponding orders.
                bids_vol = bids.Amount.values[0]
                asks_vol = asks.Amount.values[0]
                
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
                bids = bids[bids.index >= center / range_factor].dropna()
                asks = asks[asks.index <= center * range_factor].dropna()

            if range_volume:
                # limit orderbook depth, such that both asks and bids contain
                #  a volume greater than given range_volume.
                asks['accAmount'] = asks.cumsum()
                asks_level = asks[asks['accAmount'] > range_volume].index[0]
                asks = asks[asks.index<=asks_level]
                asks.drop('accAmount', axis=1, inplace=True)

                bids['accAmount'] = bids.cumsum()
                bids_level = bids[bids['accAmount'] > range_volume].index[0]
                bids = bids[bids.index>=bids_level]
                bids.drop('accAmount', axis=1, inplace=True)
                
            # must convert floats to unicode to prevent precision loss when
            # executing DataFrame.to_dict() :
            # e.g. index: 706.17 -> 706.16999999999996
            # e.g. Amount: 0.5283784 -> 0.052837839999999997
            bids.index = bids.index.map(unicode)
            bids['Amount'] = bids.Amount.map(unicode)            
            asks.index = asks.index.map(unicode)
            asks['Amount'] = asks.Amount.map(unicode)

            obj = {'bids': bids.to_dict(),
                   'asks': asks.to_dict(),
                   'timestamp': timestamp}

            f_out.write(json.dumps(obj) + "\n")
    
    if verbose:
        if overwrite:
            print("Successfully created file '{}'".format(outfile))
        else:
            print("Successfully appended {} lines to file '{}'".format(len(datafiles), outfile))
            
            
def log_mean(x, y):
    assert isinstance(x, (int, float)), 'Bad value: {}'.format(x)
    assert isinstance(y, (int, float)), 'Bad value: {}'.format(y)
    
    if x == y:
        return x
    return (x - y) / (math.log(x) - math.log(y))
