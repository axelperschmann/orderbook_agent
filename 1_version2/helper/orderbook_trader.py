import pandas as pd
import numpy as np
from IPython.display import display
from tqdm import tqdm

PRECISION = 10
EPSILON = 0.00001

from helper.orderbook_container import OrderbookContainer

class OrderbookTradingSimulator(object):
    
    def __init__(self, orderbooks, volume, tradingperiods, decisionfrequency=1):
        assert isinstance(orderbooks, list) and type(orderbooks[0]).__name__ == OrderbookContainer.__name__, "{}".format(type(orderbooks[0]))
        print(len(orderbooks), tradingperiods, decisionfrequency)
        assert len(orderbooks) == tradingperiods*decisionfrequency
        assert isinstance(volume, (float, int)) and volume != 0, "Parameter 'volume' must be 'float' or 'int' and not 0, given: {}".format(type(initial_volume))
        assert isinstance(tradingperiods, int) and tradingperiods>0, "Parameter 'tradingperiods' must be 'int' and larger than 0, given: {}".format(type(timespan))
        assert isinstance(decisionfrequency, int) and decisionfrequency > 0, "Parameter 'decisionfrequency' must be 'int', given: {}".format(type(decisionfrequency))
        # tradingperiods * decisionfrequency: trading period

        self.orderbooks = orderbooks
        self.t = 0
        self.decisionfrequency = decisionfrequency
        self.timespan = tradingperiods * decisionfrequency
        self.volume = volume

        if volume > 0:
            # buy from market
            self.order_type = 'buy'
            self.order_direction = -1
        else:
            # buy from market
            self.order_type = 'sell'
            self.order_direction = 1
        
        self.sell_history = pd.DataFrame({'Amount' : []}) 
        self.buy_history = pd.DataFrame({'Amount' : []})
        self.last_trade = pd.DataFrame({'Amount' : []})
        
        self.history = pd.DataFrame([])

        self.summary = pd.DataFrame([])

    def adjusted_orderbook(self):
        ob = self.orderbooks[self.t].copy()

        if self.t == 0:
            return ob

        # diff = ob.compare_with(self.orderbooks[self.t-1])
        
        # merge orderbook with own trade history
        # adjust asks
        ob.asks = ob.asks.subtract(self.buy_history, fill_value=0)
        drop_idx = ob.asks[ob.asks<=0].dropna().index
        ob.asks.drop(drop_idx, inplace=True)
        ob.asks.sort_index(inplace=True)
        
        # adjust bids
        ob.bids = ob.bids.subtract(self.sell_history, fill_value=0)  #[::-1]
        drop_idx = ob.bids[ob.bids<=0].dropna().index
        ob.bids.drop(drop_idx, inplace=True)

        return ob
        
    def adjust_orderbook(self, orderbook):
        ob = orderbook.copy()
        
        # merge orderbook with own trade history
        # adjust asks
        ob.asks = ob.asks.subtract(self.buy_history, fill_value=0)
        drop_idx = ob.asks[ob.asks<=0].dropna().index
        ob.asks.drop(drop_idx, inplace=True)
        ob.asks.sort_index(inplace=True)
        # adjust bids
        ob.bids = ob.bids.subtract(self.sell_history, fill_value=0)  #[::-1]
        drop_idx = ob.bids[ob.bids<=0].dropna().index
        ob.bids.drop(drop_idx, inplace=True)
        
        return ob
    
    def trade(self, limit=None, agression_factor=None, verbose=False, extrainfo={}): # orderbooks, 
        # assert isinstance(orderbooks, list)
        # assert type(orderbooks[0]).__name__ == OrderbookContainer.__name__, "{}".format(type(orderbooks[0]))
        # assert len(orderbooks)>=self.decisionfrequency
        assert (isinstance(limit, (float, int)) and limit > 0) or not limit
        assert (isinstance(agression_factor, (float, int)) and not limit) or not agression_factor
        assert isinstance(verbose, bool)
        assert isinstance(extrainfo, dict)

        timestamp = self.orderbooks[self.t].timestamp
        info = pd.DataFrame(data={'BID': None,
                                  'ASK': None,
                                  'SPREAD': None,
                                  'CENTER': None,
                                  'T': self.decisionfrequency,
                                  'VOLUME': self.volume,
                                  'volume_traded': 0,
                                  'LIMIT': limit,
                                  'cashflow': 0,
                                  'high': 0,
                                  'low': np.inf,
                                  'avg': 0,
                                  'cost_avg': 0,
                                  'cost':0,
                                  'forced':False},
                            index=[timestamp])

        if self.volume == 0:
            return -1

        if len(extrainfo) > 0:
            extrainfo = pd.DataFrame(extrainfo, columns=extrainfo.keys(), index=[timestamp])
            info = pd.concat([extrainfo, info], axis=1)

        if len(self.history) > 0:
            assert(abs(self.history.volume_traded.sum() + self.volume - self.history.VOLUME.values[0]) < EPSILON)
        
        for t in range(self.decisionfrequency):
            if self.volume==0:
                # Do nothing!
                return info  #ob
            
            assert type(self.orderbooks[t]).__name__ == OrderbookContainer.__name__, "{}".format(type(self.orderbooks[t]))
            
            ob = self.adjusted_orderbook()

            if t==0 and agression_factor is not None:

                if self.order_type == 'buy':
                    best_price = ob.get_ask()

                elif self.order_type == 'sell':
                    best_price = ob.get_bid()
                current_price, max_limit = ob.get_current_price(self.volume)
                limit_gap = max_limit - best_price
                limit = best_price + limit_gap * agression_factor
                info['LIMIT'] = limit          
                info['LIMIT_MAX'] = max_limit

            if t == 0:
                # record basic informations from beginning of trading period
                info.ASK = ob.get_ask()
                info.BID = ob.get_bid()
                info.SPREAD = (info.ASK - info.BID)
                info.CENTER = ob.get_center()
                
            if self.t == self.timespan-1:
                # must sell all remaining shares. Place Market Order!
                info['forced'] = True
                if verbose:
                    print("Run out of time (t={}).\nTrade remaining {:.4f}/{:.4f} shares for current market order price".format(self.t,
                                                                                                    self.volume,
                                                                                                    self.history.VOLUME.values[0]))
                limit = None
                
            # perform trade
            trade_result = self.__perform_trade(ob, limit)

            info.cashflow += trade_result['cashflow']
            info.volume_traded += trade_result['volume']
            # info.volume_left = round((info.VOLUME - info.volume_traded).values[0], PRECISION)
            self.volume = round((info.VOLUME - info.volume_traded).values[0], PRECISION)

            if len(self.last_trade) > 0:
                current_high = self.last_trade.index.max()
                if current_high > info.high.values[0]:
                    info.high = current_high
                    
                current_low = self.last_trade.index.min()
                if current_low < info.low.values[0]:
                    info.low = current_low
            
            
            if abs(self.volume) == 0:
                if verbose:
                    print("No shares left at t={} (self.t={}), Done!".format(t, self.t))
                break
                
        if info.volume_traded.values[0] != 0:
            info.avg = round(abs((info.cashflow / info.volume_traded)), 5)

        self.history = self.history.append(info, ignore_index=False)
        
        if info.volume_traded.values[0] != 0:
            if info.VOLUME.values[0] > 0:
                initial_bestprice = self.history.ASK.values[0]
            elif info.VOLUME.values[0] < 0:
                initial_bestprice = self.history.BID.values[0]
            
            # compute costs
            # self.history.loc[timestamp, 'cost'] = - 1. * (self.history.cashflow[-1] +
            #                                               self.history.volume_traded.values[-1] * initial_bestprice)
            self.history.loc[timestamp, 'cost'] = self.history.volume_traded.values[-1] * (self.history.avg[-1] - self.history.CENTER.values[0]) / self.history.CENTER.values[0]


            # self.history.loc[timestamp, 'cost_avg'] = np.sign(info.volume_traded.values[0]) * (self.history.avg[-1] - 
            #                                                                                    initial_bestprice)
            
        if verbose:
            # self.summarize(ob)
            if info.volume_traded.values[0] != 0:
                traded = "{:.4f}/{:.4f} shares for {}".format(info.volume_traded.values[0],
                                                info.volume_traded.values[0]+self.volume,
                                                info.cashflow.values[0])


                print("t={}: Traded {}, {:.4f} shares left".format(self.t, traded, self.volume))

        return info  # self.adjust_orderbook(ob)
        
    def __perform_trade(self, orderbook, limit=None, simulation=False):
        assert type(orderbook).__name__ == OrderbookContainer.__name__, "{}".format(type(orderbook))
        assert orderbook.timestamp == self.orderbooks[self.t].timestamp, "received wrong orderbook. Timestamp mismatch: {} vs. {}".format(orderbook.timestamp, self.orderbooks[self.t].timestamp)
        assert isinstance(limit, (float, int)) or limit is None

        volume = self.volume
        assert (self.order_type == 'buy' and volume > 0) or (self.order_type == 'sell' and volume < 0), "given: {} {}".format(self.order_type, volume)
        
        info = {'cashflow': 0,
                'volume':   0,
                'trade_summary': pd.DataFrame({'Amount' : []})}
        
        if volume > 0:
            # buy from market
            orders = orderbook.asks.copy()
            if limit:
                orders = orders[orders.index <= limit]
        elif volume <0:
            # sell to market
            orders = orderbook.bids.copy()
            if limit:
                orders = orders[orders.index >= limit]
        
        if not simulation:
            self.t += 1
        
        for pos in range(len(orders)):
            order = orders.iloc[pos]
            price = order.name

            if abs(volume) - order.Amount >= 0:
                current_order_volume = order.Amount
            else:
                current_order_volume = abs(volume)

            # remember trade activity
            info['trade_summary'] = info['trade_summary'].append(pd.DataFrame({'Amount': current_order_volume}, index=[price]))
            
            info['cashflow'] += current_order_volume * price * self.order_direction
            info['volume'] -= current_order_volume * self.order_direction
            
            volume += current_order_volume * self.order_direction

            if volume == 0:
                break
        

        if not simulation:
            self.volume_nottraded = volume
            self.last_trade = info['trade_summary']
            if self.order_type == 'buy':
                self.buy_history = self.buy_history.add(info['trade_summary'], fill_value=0)
            elif self.order_type == 'sell':
                self.sell_history = self.sell_history.add(info['trade_summary'], fill_value=0)
            else:
                print("unknown order_type")
        
        # Round trading volume to prevent subsequent rounding issues like volume_left = -1.421085e-14 (vs. intended 0.0)
        info['volume'] = round(info['volume'], PRECISION)
        
        return info
