import pandas as pd
import numpy as np
from IPython.display import display
from tqdm import tqdm

PRECISION = 10
EPSILON = 0.00001

from helper.orderbook_container import OrderbookContainer

class OrderbookTradingSimulator(object):
    
    def __init__(self, orderbooks, volume, tradingperiods, *, decisionfrequency=1):
        assert isinstance(orderbooks, list) and type(orderbooks[0]).__name__ == OrderbookContainer.__name__, "{}".format(type(orderbooks[0]))
        assert len(orderbooks) == tradingperiods*decisionfrequency, "Expected len(orderbooks) to equal tradingperiods*decisionfrequency, but: {} != {}*{}".format(len(orderbooks), tradingperiods, decisionfrequency)
        assert isinstance(volume, (float, int)) and volume != 0,  "Parameter 'volume' must be 'float' or 'int' and not 0, given: {}".format(type(initial_volume))
        assert isinstance(tradingperiods, int) and tradingperiods>0, "Parameter 'tradingperiods' must be 'int' and larger than 0, given: {}".format(type(timespan))
        assert isinstance(decisionfrequency, int) and decisionfrequency > 0, "Parameter 'decisionfrequency' must be 'int', given: {}".format(type(decisionfrequency))
        # tradingperiods * decisionfrequency: trading period

        self.orderbooks = orderbooks
        self.masterbook = orderbooks[0].copy()

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
        
        self.history = pd.DataFrame([])

        self.summary = {'amount': 0, 'cashflow':0, 'remaining': volume, 'cost':0}

    #def __merge_negative_shares(self, book, max_iterations=1000):
    #    """
    #    Orderbook adjustment can lead to negative share numbers.
    #    Countermeasurement:
    #    Iteratively merge price_levels showing a negative share
    #    number with it's closest neighbour at that time.
    #    """
    #
    #    drop_idx = book[book==0].dropna().index
    #    book.drop(drop_idx, inplace=True)
    #
    #    loop_counter = 0
    #    while loop_counter < max_iterations:
    #        pricelevels_with_negative_shares = book[book<=0].dropna()
    #        if len(pricelevels_with_negative_shares) == 0:
    #            # Done, nothing to fix here
    #            break
    #
    #        # get first candidate with negative shares and it's closest neighbour
    #        price_level = pricelevels_with_negative_shares.index[0]
    #        closest_neighbour = book.iloc[abs(book.index-price_level).argsort()[1]]
    #
    #        # merge candidate and neighbour, then drop candidate
    #        book.loc[closest_neighbour.name] += pricelevels_with_negative_shares.loc[price_level]
    #        book.drop(price_level, inplace=True)
    #        
    #        loop_counter  += 1
    #    
    #    return book

    def __subtract_lastTrade_fromMaster(self, last_trade):
        if len(last_trade) > 0:
            if self.order_type == 'buy':
                self.masterbook.asks = self.masterbook.asks.subtract(last_trade, fill_value=0)
                drop_idx = self.masterbook.asks[self.masterbook.asks.Amount<=0].Amount.dropna().index
                self.masterbook.asks.drop(drop_idx, inplace=True)
            elif self.order_type == 'sell':
                # ToDo: Implement
                raise NotImplementedError
            else:
                print(self.order_type)
                raise NotImplementedError

    def get_next_masterbook(self):
        if self.t < len(self.orderbooks)-1:

            self.t += 1
            masterbook = self.adjust_masterbook(inplace=False)
            self.t -= 1

            return masterbook
        else:
            return self.masterbook


    def adjust_masterbook(self, inplace=True):
        if self.t == 0:
            return

        masterbook = self.masterbook.copy()

        # check difference between previous and current orderbook
        ob_current = self.orderbooks[self.t]
        ob_previous = self.orderbooks[self.t-1]
        diff = ob_current.compare_with(ob_previous)

        # adjust asks of masterbook
        masterbook.asks = masterbook.asks.add(diff.asks, fill_value=0)
        drop_idx = masterbook.asks[masterbook.asks.Amount<=0].Amount.dropna().index
        masterbook.asks.drop(drop_idx, inplace=True)
        masterbook.asks.sort_index(inplace=True)

        # adjust bids of masterbook
        masterbook.bids = masterbook.bids.add(diff.bids, fill_value=0)
        drop_idx = masterbook.bids[masterbook.bids.Amount<=0].Amount.dropna().index
        masterbook.bids.drop(drop_idx, inplace=True)
        masterbook.bids.sort_index(inplace=True, ascending=False)

        masterbook.timestamp = ob_current.timestamp

        if inplace:
            self.masterbook = masterbook
        else:
            return masterbook

    
    def trade(self, limit=None, agression_factor=None, *, verbose=False, extrainfo={}): # orderbooks, 
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
            
            self.adjust_masterbook()
            ob = self.masterbook

            if t == 0:
                # record basic informations from beginning of trading period
                info.ASK = ob.get_ask()
                info.BID = ob.get_bid()
                info.SPREAD = (info.ASK - info.BID)
                info.CENTER = ob.get_center()

                if agression_factor is not None:

                    if self.order_type == 'buy':
                        best_price = ob.get_ask()

                    elif self.order_type == 'sell':
                        best_price = ob.get_bid()
                    current_price, max_limit = ob.get_current_price(self.volume)
                    limit_gap = max_limit - best_price
                    limit = best_price + limit_gap * agression_factor
                    info['LIMIT'] = limit          
                    info['LIMIT_MAX'] = max_limit
                
            if self.t == self.timespan-1:
                # must sell all remaining shares. Place Market Order!
                info['forced'] = True
                if verbose:
                    print("Run out of time (t={}).\nTrade remaining {:.4f}/{:.4f} shares for current market order price".format(self.t,
                                                                                                    self.volume,
                                                                                                    self.history.VOLUME.values[0]))
                limit = None
                
            # perform trade
            trade_result = self.__perform_trade(ob, limit=limit)

            info.cashflow += trade_result['cashflow']
            info.volume_traded += trade_result['volume']
            # info.volume_left = round((info.VOLUME - info.volume_traded).values[0], PRECISION)
            self.volume = round((info.VOLUME - info.volume_traded).values[0], PRECISION)

            if len(trade_result.get('trade_summary')) > 0:
                current_high = trade_result.get('trade_summary').index.max()
                if current_high > info.high.values[0]:
                    info.high = current_high
                    
                current_low = trade_result.get('trade_summary').index.min()
                if current_low < info.low.values[0]:
                    info.low = current_low
            
            self.t += 1
            if abs(self.volume) == 0:
                # if verbose:
                #     print("No shares left after t={}. Done!".format(self.t-1))
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

            self.summary['amount'] += info.volume_traded.values[0]
            self.summary['cashflow'] += info.cashflow.values[0]
            self.summary['remaining'] = self.volume
            self.summary['cost'] += self.history.loc[timestamp, 'cost']

            # self.history.loc[timestamp, 'cost_avg'] = np.sign(info.volume_traded.values[0]) * (self.history.avg[-1] - 
            #                                                                                    initial_bestprice) 
            
        if verbose:
            if info.volume_traded.values[0] != 0:
                traded = "{:.4f}/{:.4f} shares for {}".format(info.volume_traded.values[0],
                                                info.volume_traded.values[0]+self.volume,
                                                info.cashflow.values[0])


                print("t={}: Traded {}, {:.4f} shares left".format(self.t-1, traded, self.volume))

        return self.summary  # self.adjust_orderbook(ob)
        
    def __perform_trade(self, orderbook, *, limit=None, simulation=False):
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
            
            if self.order_type == 'buy':
                self.buy_history = self.buy_history.add(info['trade_summary'], fill_value=0)
            elif self.order_type == 'sell':
                self.sell_history = self.sell_history.add(info['trade_summary'], fill_value=0)
            else:
                print("unknown order_type")
        
        # Round trading volume to prevent subsequent rounding issues like volume_left = -1.421085e-14 (vs. intended 0.0)
        info['volume'] = round(info['volume'], PRECISION)

        self.__subtract_lastTrade_fromMaster(last_trade=info['trade_summary'])
        
        return info
