import pandas as pd
import numpy as np
from IPython.display import display
from tqdm import tqdm

PRECISION = 10

from orderbook_container import OrderbookContainer

class OrderbookTradingSimulator(object):
    
    def __init__(self, decisionfrequency=1, initial_volume=4):
        assert isinstance(initial_volume, (float, int)), "Parameter 'initial_volume' must be 'float' or 'int', given: {}".format(type(initial_volume))
        
        self.t = 0
        self.volume = initial_volume
        self.timespan = decisionfrequency
        
        self.sell_history = pd.DataFrame({'Amount' : []}) 
        self.buy_history = pd.DataFrame({'Amount' : []}) # {'705.45': 3.172181}
        self.last_trade = pd.DataFrame({'Amount' : []})
        
        self.history = pd.DataFrame([])
        
    def adjust_orderbook(self, orderbook):
        ob = orderbook.copy()

        # merge orderbook with own trade history
        # adjust asks
        ob.asks = ob.asks.subtract(self.buy_history, fill_value=0)
        drop_idx = ob.asks[ob.asks<=0].dropna().index
        ob.asks.drop(drop_idx, inplace=True)
        # adjust bids
        ob.bids = ob.bids.subtract(self.sell_history, fill_value=0)
        drop_idx = ob.bids[ob.bids<=0].dropna().index
        ob.bids.drop(drop_idx, inplace=True)
        
        return ob
    
    def trade_timespan(self, orderbooks, volume, limit, verbose=False, must_trade=False):
        assert isinstance(orderbooks, list)
        assert type(orderbooks[0]).__name__ == OrderbookContainer.__name__, "{}".format(type(orderbooks[0]))
        assert len(orderbooks)>=self.timespan
        assert isinstance(volume, (float, int))
        assert (isinstance(limit, (float, int)) and limit > 0) or not limit
        assert isinstance(verbose, bool)
        assert isinstance(must_trade, bool)
        timestamp = orderbooks[0].timestamp
        info = pd.DataFrame(data={'BID': None,
                                  'ASK': None,
                                  'SPREAD': None,
                                  'CENTER': None,
                                  'T': self.timespan,
                                  'VOLUME': volume,
                                  'volume_traded': 0,
                                  'volume_left': volume,
                                  'LIMIT': limit,
                                  'cashflow': 0,
                                  'high': 0,
                                  'low': np.inf,
                                  'avg': 0,
                                  'cost_avg': 0,
                                  'cost':0},
                            index=[timestamp])
        
        for t in range(self.timespan):
            assert type(orderbooks[t]).__name__ == OrderbookContainer.__name__, "{}".format(type(orderbooks[t]))
            ob = orderbooks[t].copy()
            ob = self.adjust_orderbook(ob)

            if t == 0:
                # record basic informations from beginning of trading period
                info.ASK = ob.get_ask()
                info.BID = ob.get_bid()
                info.SPREAD = (info.ASK - info.BID)
                info.CENTER = ob.get_center()
               
            if volume==0:
                # Do nothing!
                return ob
                
            if must_trade and t == self.timespan-1:
                # must sell all remaining shares. Place Market Order!
                print("Run out of time (t={}).\nTrade remaining {:.4f}/{:.4f} shares for current market order price".format(t,
                                                                                                  info.volume_left.values[0],
                                                                                                  info.VOLUME.values[0]))
                limit = None
                
            # perform trade
            trade_result = self.__perform_trade(ob, info.volume_left.values[0], limit)

            info.cashflow += trade_result['cashflow']
            info.volume_traded += trade_result['volume']
            info.volume_left = round((info.VOLUME - info.volume_traded).values[0], PRECISION)
            
            if len(self.last_trade) > 0:
                current_high = self.last_trade.index.max()
                if current_high > info.high.values[0]:
                    info.high = current_high
                    
                current_low = self.last_trade.index.min()
                if current_low < info.low.values[0]:
                    info.low = current_low
            
            
            if abs(info.volume_left.values[0]) == 0:
                print("No shares left at t={} (self.t={}), Done!".format(t, self.t))
                break
                
        if info.volume_traded.values[0] != 0:
            info.avg = round(abs((info.cashflow / info.volume_traded)), 5)

        self.history = self.history.append(info, ignore_index=False)
        
        if info.volume_traded.values[0] != 0:
            if volume > 0:
                initial_bestprice = self.history.ASK.values[0]
            elif volume < 0:
                initial_bestprice = self.history.BID.values[0]
            
            # compute costs
            self.history.loc[timestamp, 'cost'] = - 1. * (self.history.cashflow[-1] +
                                                          self.history.volume_traded.values[-1] * initial_bestprice)
            self.history.loc[timestamp, 'cost_avg'] = np.sign(info.volume_traded.values[0]) * (self.history.avg[-1] - 
                                                                                               initial_bestprice)
        
        if verbose:
            # self.summarize(ob)
            
            print("Traded {:.4f}/{:.4f} shares for {}, {:.4f} shares left".format(info.volume_traded.values[0],
                                                              (info.volume_traded + info.volume_left).values[0],
                                                              info.cashflow.values[0],
                                                              info.volume_left.values[0]))
        return self.adjust_orderbook(ob)
        
    def __perform_trade(self, orderbook, volume=None, limit=None, simulation=False):
        assert type(orderbook).__name__ == OrderbookContainer.__name__, "{}".format(type(orderbook))
        assert isinstance(volume, (float, int)) or volume is None
        if volume is None:
            volume = self.volume
        assert volume != 0
        assert isinstance(limit, (float, int)) or limit is None
        
        info = {'cashflow': 0,
                'volume':   0,
                'trade_summary': pd.DataFrame({'Amount' : []})}
        
        if not simulation:
            self.t += 1

        if volume > 0:
            # buy from market
            order_type = 'ask'
            order_direction = -1
            orders = orderbook.asks.copy()
            if limit:
                orders = orders[orders.index <= limit]
        elif volume <0:
            # sell to market
            print("Not implemented yet!")
            assert True==False, "ToDo!"
        
        for pos in range(len(orders)):
            order = orders.iloc[pos]
            price = order.name
            
            if abs(volume) - order.Amount >= 0:
                current_order_volume = order.Amount
            else:
                current_order_volume = abs(volume)

            # remember trade activity
            info['trade_summary'] = info['trade_summary'].append(pd.DataFrame({'Amount': current_order_volume}, index=[price]))
            
            info['cashflow'] += current_order_volume * price * order_direction
            info['volume'] -= current_order_volume * order_direction
            
            volume += current_order_volume * order_direction

            if volume == 0:
                break

        # display(orders.subtract(trade_summary, fill_value=0).dropna())

        if not simulation:
            self.volume_nottraded = volume
            self.last_trade = info['trade_summary']
            if order_type == 'ask':
                self.buy_history = self.buy_history.add(info['trade_summary'], fill_value=0)
            elif order_type == 'bid':
                self.sell_history = self.sell_history.add(info['trade_summary'], fill_value=0)
            else:
                print("unknown order_type")
        
        # Round trading volume to prevent subsequent rounding issues like volume_left = -1.421085e-14 (vs. intended 0.0)
        info['volume'] = round(info['volume'], PRECISION)
        
        return info
