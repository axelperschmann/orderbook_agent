import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display

class OrderbookContainer:
    def __init__(self, timestamp, bids, asks):
        assert isinstance(timestamp, unicode) or isinstance(timestamp, str)
        assert isinstance(bids, pd.DataFrame) and len(bids)>0
        assert isinstance(asks, pd.DataFrame) and len(asks)>0
        
        self.timestamp = timestamp
        self.bids = bids
        self.asks = asks  
        self.enriched = False

    def __str__(self):
        return "Orderbook snapshot: {}".format(self.timestamp)
        
    def get_ask(self):
        return self.asks.index.values[0]
    
    def get_bid(self):
        return self.bids.index.values[0]
    
    def get_center(self):
        return log_mean(self.get_ask(), self.get_bid())
    
    def to_DataFrame(self, depth=None, range_factor=None):
        assert (isinstance(depth, int) and depth > 0) or depth is None, "depth={}, {}".format(depth, type(depth))
        assert ((isinstance(range_factor, float) or isinstance(range_factor, int)) and range_factor > 1) or range_factor is None, "range_factor={}, {}".format(range_factor, type(range_factor))
        
        if depth:
            bids = self.bids.head(depth).copy()
            asks = self.asks.head(depth).copy()
        else:
            bids = self.bids.copy()
            asks = self.asks.copy()
            
        if range_factor:
            bids = bids[bids.index > self.get_center()/range_factor]
            asks = asks[asks.index < self.get_center()*range_factor]
            
        bids['Type'] = 'bid'
        center = pd.DataFrame([[np.nan, 'center']], columns=['Amount', 'Type'], index=[self.get_center()])
        asks['Type'] = 'ask'
        
        return pd.concat([bids[::-1], center, asks])
        
    def enrich(self):
        self.enriched = True
        self.bids['norm_Price'] = self.bids.index / self.get_center()
        self.bids['Volume'] = self.bids.index * self.bids.Amount
        self.bids['VolumeAcc'] = 0
        self.bids['VolumeAcc'] = (self.bids.Volume).cumsum().values

        self.asks['norm_Price'] = self.asks.index / self.get_center()
        self.asks['Volume'] = self.asks.index * self.asks.Amount
        self.asks['VolumeAcc'] = 0
        self.asks['VolumeAcc'] = (self.asks.Volume).cumsum().values

    
    def plot(self, normalized=False, range_factor=None, outfile=None):
        assert isinstance(normalized, bool)
        assert ((isinstance(range_factor, float) or isinstance(range_factor, int)) and range_factor > 1) or range_factor is None
        assert isinstance(outfile, str) or isinstance(outfile, unicode) or outfile is None
        
        if not self.enriched:
            print("enrich")
            self.enrich()
            
        data = self.to_DataFrame(range_factor=range_factor)
        
        plt.figure(figsize=(16,8))
        if normalized:
            if range_factor:
                xlim = (1./range_factor, range_factor)
            else:
                xlim = (data.norm_Price.values[0], data.norm_Price.values[-1])

            bids_lim = data.VolumeAcc.values[0]
            asks_lim = data.VolumeAcc.values[-1]
            y_factor = asks_lim + bids_lim

            asks_x = data[data.Type == 'ask'].norm_Price.values
            asks_y = data[data.Type == 'ask'].VolumeAcc / y_factor

            bids_x = data[data.Type == 'bid'].norm_Price.values
            bids_y = data[data.Type == 'bid'].VolumeAcc / y_factor    # added .copy()

            # lowest bid and highhest ask should sum up to 100% of y-axis
            plt.ylim((0,1))
        else:
            center = data[data.Type == 'center'].index[0]        
            if range_factor:
                xlim = (center/range_factor, center*range_factor)
            else:
                xlim = (data.index.values[0], data.index.values[-1])

            bids_lim = data[data.index>xlim[0]].VolumeAcc.values[0]
            asks_lim = data[data.index<xlim[1]].VolumeAcc.values[-1]
            y_factor = asks_lim + bids_lim

            asks_x = data[data.Type == 'ask'].index
            asks_y = data[data.Type == 'ask'].VolumeAcc
            bids_x = data[data.Type == 'bid'].index
            bids_y = data[data.Type == 'bid'].VolumeAcc

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

        if outfile:
            if outfile[-4:] != '.svg':
                outfile = "{}.svg".format(outfile)
            plt.savefig(outfile, format='svg')
            print("Successfully saved'{}'".format(outfile))
            plt.close()
        else:
            plt.show()
            plt.close()


def log_mean(x, y):
    assert isinstance(x, int) or isinstance(x, float), 'Bad value: {}'.format(x)
    assert isinstance(y, int) or isinstance(y, float), 'Bad value: {}'.format(y)
    
    if x == y:
        return x
    return (x - y) / (math.log(x) - math.log(y))
 