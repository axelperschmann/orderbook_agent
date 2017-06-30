import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
from IPython.display import display

from functools import wraps
from time import time

class OrderbookContainer(object):

    def timing(f):
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            print('func:%r  took: %2.4f sec' % \
              (f.__name__, te-ts))
            return result
        return wrap

    def __init__(self, timestamp, bids, asks, features=None, kind='orderbook'):  #enriched=False, 
        assert isinstance(timestamp, (str, pd.tslib.Timestamp)), "Parameter 'timestamp' is '{}', type={}".format(timestamp, type(timestamp))
        assert isinstance(bids, pd.DataFrame)  # and len(bids)>0
        assert isinstance(asks, pd.DataFrame)  # and len(asks)>0
        # assert isinstance(enriched, bool)
        assert isinstance(kind, str)
        self.timestamp = timestamp
        self.bids = bids
        self.asks = asks  
        self.kind = kind
        self.norm_factor = 1.0
        self.features = features or {}
        if kind == 'orderbook':
            # sorting is computationally expensive. Avoid it when working on 'diff'-book
            self.asks.sort_index(inplace=True)
            self.bids.sort_index(inplace=True, ascending=False)

            assert len(self.asks) > 0, "shape: {}, ts: {}".format(asks.shape, timestamp)
            assert len(self.bids) > 0, "shape: {}, ts: {}".format(bids.shape, timestamp)
            self.features['spread'] = asks.index[0]-bids.index[0]
    
    
    def copy(self):
        return OrderbookContainer(
            timestamp=self.timestamp,
            bids=self.bids,
            asks=self.asks,
            # enriched=self.enriched,
            kind=self.kind,
            features=self.features)
    
    def compare_with(self, other):        
        bids_diff = self.bids.subtract(other.bids, axis=1, fill_value=0)
        asks_diff = self.asks.subtract(other.asks, axis=1, fill_value=0)

        return OrderbookContainer(timestamp=self.timestamp,
                                  bids=bids_diff[bids_diff != 0].dropna(),
                                  asks=asks_diff[asks_diff != 0].dropna(),
                                  features=None,
                                  kind='diff')
        
    def __str__(self):

        representation = "OrderbookContainer from {} (factor: {})\n  {} bids (best: {})\n  {} asks (best: {})\nprice: {}\n  kind: '{}'".format(
            self.timestamp,
            self.norm_factor,
            len(self.bids),
            self.get_bid(),
            len(self.asks),
            self.get_ask(),
            self.get_center(),
            self.kind)

        if hasattr(self, 'features') and len(self.features) > 0:
            for key in self.features.keys():
                representation = "{}\n  -{}: {}".format(representation, key, self.features[key])
        
        return representation
    
    def __repr__(self):
          return self.__str__()


    def get_current_sharecount(self, cash):
        assert isinstance(cash, (int,float)) and cash != 0, "'cash' must not be 0"
        shares = 0
        if cash > 0:
            orders = self.asks
            orderdirection = 1
        else:
            orders = self.bids
            orderdirection = -1
        orders['Volume'] = orders.Amount * orders.index
        assert abs(cash) < orders.Volume.sum(), "Can't handle trade. Orderbookvolume exceeded, must fulfill abs({}) < {}".format(cash, orders.Volume.sum())
        
        accVol = orders.Volume.cumsum()
        fast_items = len(accVol[accVol < abs(cash)].index)
        
        if fast_items > 0:
            orders_sub = orders.iloc[:fast_items,:]
            shares = orders_sub.Amount.sum()
            cash -= orders_sub.Volume.sum() * orderdirection
        
        if round(cash, 2) != 0:
            order = orders.iloc[fast_items]
            order_price = order.name

            if abs(cash) - order.Volume >= 0:
                current_order_volume = order.Volume/order.name
            else:
                current_order_volume = cash/order.name

            cash -= current_order_volume * order_price
            shares += current_order_volume * orderdirection
            limit = order_price
        else:
            limit = orders_sub.index[-1]
                   
        assert round(cash, 2) == 0, "given: {}".format(cash)
        
        
        return shares, limit
        
    def get_current_price(self, volume):
        assert isinstance(volume, (int, float)) and volume != 0, "'volume' must not be 0"
        price = 0
        
        if volume > 0:
            orders = self.asks
            orderdirection = 1
        else:
            orders = self.bids
            orderdirection = -1
            # volume = abs(volume)
        
        assert volume < orders.Amount.sum(), "Can't handle trade. Orderbookvolume exceeded: NOT({} < {})".format(volume, orders.Amount.sum())
            
        for row in orders.itertuples():
            order_price = row[0]
            order_volume = row[1]

            if abs(volume) >= order_volume:
                current_order_volume = order_volume
            else:
                current_order_volume = abs(volume)
                
            volume -= current_order_volume * orderdirection
            price += current_order_volume * order_price
            
            if abs(volume) == 0:
                break

        limit = order_price

        return price, limit
        
        
    def get_ask(self):
        if len(self.asks) > 0:
            ask = self.asks.index.values[0]
        else:
            ask = np.nan
        return ask
    
    def get_bid(self):
        if len(self.bids) > 0:
            bid = self.bids.index.values[0]
        else:
            bid = np.nan
        return bid
    
    def get_center(self):
        return log_mean(self.get_ask(), self.get_bid())
    
    
    def tail(self, depth=3, range_factor=None):
        assert (isinstance(depth, int) and depth > 0), "depth={}, {}".format(depth, type(depth))
        assert (isinstance(range_factor, (float, int)) and range_factor > 1) or range_factor is None, "range_factor={}, {}".format(range_factor, type(range_factor))
        return self.to_DataFrame(depth=-depth, range_factor=range_factor)
    
    def head(self, depth=3, range_factor=None):
        assert (isinstance(depth, int) and depth > 0)
        assert (isinstance(range_factor, (float, int)) and range_factor > 1) or range_factor is None, "range_factor={}, {}".format(range_factor, type(range_factor))
        return self.to_DataFrame(depth=depth, range_factor=range_factor)
        
    def to_DataFrame(self, depth=None, range_factor=None):
        assert (isinstance(depth, int) and depth != 0) or depth is None, "depth={}, {}".format(depth, type(depth))
        assert (isinstance(range_factor, (float, int)) and range_factor > 1) or range_factor is None, "range_factor={}, {}".format(range_factor, type(range_factor))
        bids = self.bids.copy()
        asks = self.asks.copy()
        
        if range_factor:
            bids = bids[bids.index > self.get_center()/range_factor]
            asks = asks[asks.index < self.get_center()*range_factor]
        
        if depth:
            if depth > 0:
                bids = bids.head(depth)
                asks = asks.head(depth)
            elif depth < 0:
                bids = bids.tail(abs(depth))
                asks = asks.tail(abs(depth))
            
        return self.__to_DataFrame(bids, asks)
    
    
    def __to_DataFrame(self, bids, asks):
        assert isinstance(bids, pd.DataFrame)
        assert isinstance(asks, pd.DataFrame)
        
        bids['Type'] = 'bid'
        asks['Type'] = 'ask'
        
        if self.kind == 'orderbook':
            center = pd.DataFrame([[np.nan, 'center']],
                                  columns=['Amount', 'Type'], index=[self.get_center()])
            df = pd.concat([bids[::-1], center, asks])
        elif self.kind == 'diff':
            df = pd.concat([bids[::-1], asks])
        else:
            raise("Error! Unknown orderbooktype: {}".format(self.kind))
        return df
        
    # def enrich(self):
    #     self.enriched = True
    #     self.bids['norm_Price'] = self.bids.index / self.get_center()
    #     self.bids['Volume'] = self.bids.index * self.bids.Amount
    #     self.bids['VolumeAcc'] = 0
    #     self.bids['VolumeAcc'] = (self.bids.Volume).cumsum().values
    # 
    #     self.asks['norm_Price'] = self.asks.index / self.get_center()
    #     self.asks['Volume'] = self.asks.index * self.asks.Amount
    #     self.asks['VolumeAcc'] = 0
    #     self.asks['VolumeAcc'] = (self.asks.Volume).cumsum().values
    # 
    # def enrich_undo(self):
    #     self.bids = pd.DataFrame(self.bids.Amount)
    #     self.asks = pd.DataFrame(self.asks.Amount)
    # 
    #     self.enriched = False

    def discretize(self, ticksize=0.1, depth=None, center='ask', range_factor=None):
        data = self.to_DataFrame(range_factor=range_factor)

        # center = data[data.Type=='center'].index[0]
        # data['norm_Price'] = data.index / center
        # data['Volume'] = data.index * data.Amount
        # data['VolumeAcc'] = 0

        bids = self.bids.copy()
        bids['Volume'] = bids.index * bids.Amount
        asks = self.asks.copy()
        asks['Volume'] = asks.index * asks.Amount


        bids['VolumeAcc'] = bids.Volume.cumsum()
        asks['VolumeAcc'] = asks.Volume.cumsum()

        display("asks", asks.head(5))
        display("bids", bids.head(3)  )

        center = float(self.get_center())
        print("center", center)

        query = center * (1.001)
        idx = asks.index.get_loc(query, method='bfill')
        print("idx", idx, query)
        print(asks.iloc[idx,:])

        if isinstance(depth, int):
            center_idx = data.index.get_loc(data[data.Type=='center'].index[0])
            data = data.iloc[center_idx-depth:center_idx+depth+1,:].copy()

        return asks

    
    def plot(self, normalized=False, range_factor=None, outfile=None, figsize=(8,6), outformat='pdf'):
        assert isinstance(normalized, bool)
        assert (isinstance(range_factor, (float, int)) and range_factor > 1) or range_factor is None
        assert isinstance(outfile, str) or outfile is None
        
        assert self.kind == 'orderbook',  "Can only plot OrderbookContainers of kind 'orderbook'. This OrderbookContainer is of type '{}'.".format(self.kind)
            
        data = self.to_DataFrame(range_factor=range_factor)
        
        center = data[data.Type=='center'].index[0]
        data['norm_Price'] = data.index / center
        data['Volume'] = data.index * data.Amount
        data['VolumeAcc'] = 0

        bids = data[data.Type=='bid'][::-1].copy()
        asks = data[data.Type=='ask'].copy()

        bids['VolumeAcc'] = bids.Volume.cumsum()
        asks['VolumeAcc'] = asks.Volume.cumsum()

        plt.figure(figsize=figsize)
        if normalized:
            if range_factor:
                xlim = (1./range_factor, range_factor)
            else:
                xlim = (bids.norm_Price.values[-1], asks.norm_Price.values[-1])

            bids_lim = bids.VolumeAcc.values[-1]
            asks_lim = asks.VolumeAcc.values[-1]
            y_factor = asks_lim + bids_lim

            asks_x = asks.norm_Price.values
            asks_y = asks.VolumeAcc / y_factor

            bids_x = bids.norm_Price.values
            bids_y = bids.VolumeAcc / y_factor    # added .copy()

            # lowest bid and highhest ask should sum up to 100% of y-axis
            plt.ylim((0,1))
        else:
            center = data[data.Type == 'center'].index[0]        
            if range_factor:
                xlim = (center/range_factor, center*range_factor)
            else:
                xlim = (bids.index.values[-1], asks.index.values[-1])

            bids_lim = bids[bids.index>xlim[0]].VolumeAcc.values[-1]
            asks_lim = asks[asks.index<xlim[1]].VolumeAcc.values[-1]
            y_factor = asks_lim + bids_lim

            asks_x = asks.index
            asks_y = asks.VolumeAcc
            bids_x = bids.index
            bids_y = bids.VolumeAcc

            # lowest bid and highhest ask should sum up to 100% of y-axis
            plt.ylim((0,y_factor))
            
        plt.plot(bids_x, bids_y, color='g', label='VolumeAcc Bid')
        plt.plot(asks_x, asks_y, color='r', label='VolumeAcc Ask')
        plt.fill_between(bids_x, bids_y, 0, color='g', alpha=0.1)
        plt.fill_between(asks_x, asks_y, 0, color='r', alpha=0.1)
        plt.xlim(xlim)
        center = data[data.Type=='center'].index[0]
        plt.suptitle("{} - center: {:1.4f}, ".format(self.timestamp, center))
        plt.legend()
        plt.ylabel("Accumulated Volume")
        plt.xlabel("Price Level")
        plt.ylim((0,80000))
        if outfile:
            # if outfile[-4:] != ".{}".format(outformat):
            #     outfile = "{}.{}".format(outfile, outformat)
            plt.savefig(outfile)  #, format=outformat)
            print("Successfully saved '{}'".format(outfile))
            plt.close()
        else:
            plt.show()
            plt.close()


def log_mean(x, y):
    assert isinstance(x, (int, float)), "Bad value: {}, '{}'".format(x, type(x))
    assert isinstance(y, (int, float)), "Bad value: {}, '{}'".format(y, type(y))
    
    if x == y:
        return x
    return (x - y) / (math.log(x) - math.log(y))
 