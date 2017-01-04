import pandas as pd
import numpy as np
from manage_orderbooks import log_mean, orderbook_preview
from IPython.display import display
from tqdm import tqdm

PRECISION = 10

class OrderbookTradingSimulator:
    
    def summarize(self, df):
        assert isinstance(df, pd.DataFrame)
        
        df_adj = self.adjust_orderbook(df)
        ask = df_adj[df_adj.Type == 'ask'].iloc[0].Price
        bid = df_adj[df_adj.Type == 'bid'].iloc[-1].Price
        
        
        print(" #####   TRADE Number {}   #####".format(self.t))
        print("Purchase history")
        display(self.purchase_history)
        print("Sell history")
        display(self.sell_history)
        print("Last trade")
        display(self.last_trade)
        
        print("Last trade  : {:1.4f}/{:1.4f} shares".format(self.history.volume_traded.values[-1], self.history.VOLUME.values[-1]))
        print("Total trades: {:1.4f}/{:1.4f} shares".format(sum(self.history.volume_traded), self.history.VOLUME.values[0]))
        
        print("Spread: {} (bid: {}, ask: {})".format(ask-bid, bid, ask))
        
        print("Adjusted Orderbook:")
        display(orderbook_preview(df_adj, 3))
        
    
    def __init__(self, T=5):
        self.t = 0
        self.T = T
        
        self.purchase_history = {}  # {'705.45': 3.1721811199999999}
        self.sell_history = {}  # {'703.70': 0.001}
        self.last_trade = {}
        
        self.history = pd.DataFrame([])

    
    def adjust_orderbook(self, df, verbose=False):
        assert isinstance(df, pd.DataFrame)
        assert isinstance(verbose, bool)
        
        df = df.copy()
        if not 'Volume' in df.columns and not 'VolumeAcc' in df.columns:
            # compact databook provided, contains only Price, Amount and Type
            if verbose:
                print("Compact orderbook provided. Enriched it with column 'Volume' and 'VolumeAcc' to DataFrame")
            df['Volume'] = df.Price * df.Amount
            df['VolumeAcc'] = 0
        
        # adjust dataFrame to our own trade_history. Simulate influence caused by our previous trades.
        for pricelevel in self.purchase_history.keys():
            pd_row = df[(df.Type=='ask') & (df.Price == float(pricelevel))]
            if len(pd_row) == 1:
                new_amount = (pd_row.Amount - self.purchase_history[pricelevel])
                df.loc[pd_row.index, 'Amount']  = new_amount
                df.loc[pd_row.index, 'Volume']  = new_amount * pd_row.Price
        for pricelevel in self.sell_history.keys():
            pd_row = df[(df.Type=='bid') & (df.Price == float(pricelevel))]
            if len(pd_row) == 1:
                new_amount = (pd_row.Amount - self.sell_history[pricelevel])
                df.loc[pd_row.index, 'Amount']  = new_amount
                df.loc[pd_row.index, 'Volume']  = new_amount * pd_row.Price
        df = df[(df.Amount > 0) | (df.Type=='center')]
        
        assert len(df[df.Type=='ask']) > 0, "Bad Orderbook, no ask-orders left."
        assert len(df[df.Type=='center']) == 1
        assert len(df[df.Type=='bid']) > 0, "Bad Orderbook, no bid-orders left."
        
        self.ask = df[df.Type == 'ask'].iloc[0].Price
        self.bid = df[df.Type == 'bid'].iloc[-1].Price
        self.center_log = log_mean(self.ask, self.bid)
        df.loc[df[df['Type'] == 'center'].index, 'Price'] = self.center_log
        
        # compute accumulated order volume
        df.loc[df.Type=='bid', 'VolumeAcc'] = (df[df.Type=='bid'].Volume)[::-1].cumsum().values[::-1]
        df.loc[df.Type=='ask', 'VolumeAcc'] = (df[df.Type=='ask'].Volume).cumsum().values

        # normalization
        df['norm_Price'] = df.Price / self.center_log

        return df
    
    def __perform_trade(self, df, volume, limit=None):
        assert isinstance(df, pd.DataFrame)
        assert isinstance(volume, float) or isinstance(volume, int)
        assert volume != 0
        assert isinstance(limit, float) or isinstance(limit, int) or not limit
        
        info = {'cashflow': 0,
                'volume':   0}
        
        self.t += 1
        trade_summary = {}
        
        if volume > 0:
            # buy from market
            order_type = 'ask'
            order_direction = -1
            df = df[df.Type == order_type].copy()
            history = self.purchase_history
        elif volume <0:
            # sell to market
            order_type = 'bid'
            order_direction = 1
            df = df[df.Type == order_type].iloc[::-1].copy()
            history = self.sell_history

        for pos in range(len(df)):
            order = df.iloc[pos]
            price = str(order.Price)
           
            if limit:
                if (order_type == 'ask' and order.Price > limit) or (order_type == 'bid' and order.Price < limit):
                    # Price limit exceeded, stop trading now!
                    break
                    
            if abs(volume) - order.Amount >= 0:
                current_order_volume = order.Amount
            else:
                current_order_volume = abs(volume)

            # remember trade activity
            if price in trade_summary.keys():
                trade_summary[price] += current_order_volume
            else:
                trade_summary[price] = current_order_volume

            if price in history.keys():
                history[price] += current_order_volume
            else:
                history[price] = current_order_volume

            info['cashflow'] += current_order_volume * order.Price * order_direction
            info['volume'] -= current_order_volume * order_direction
            
            volume += current_order_volume * order_direction

            if volume == 0:
                break
        self.volume_nottraded = volume
        
        self.last_trade = trade_summary
        
        # Round trading volume to prevent subsequent rounding issues like volume_left = -1.421085e-14 (vs. intended 0.0)
        info['volume'] = round(info['volume'], PRECISION)
        
        return info
        
    def trade_timespan(self, orderbooks, timestamps, volume, limit, verbose=True, timespan=1, must_trade=False):
        assert isinstance(orderbooks, list)
        assert len(orderbooks)==timespan
        assert isinstance(volume, float) or isinstance(volume, int)
        assert isinstance(limit, float) or isinstance(limit, int)
        assert limit > 0
        assert isinstance(verbose, bool)
        assert isinstance(timespan, int) and timespan > 0
        assert isinstance(must_trade, bool)
        
        volume_traded = 0
        self.volume_of_last_trade_period = 0

        info = pd.DataFrame(data={'BID': None,
                                  'ASK': None,
                                  'SPREAD': None,
                                  'CENTER': None,
                                  'TIMESPAN': timespan,
                                  'VOLUME': volume,
                                  'volume_traded': 0,
                                  'volume_left': volume,
                                  'LIMIT': limit,
                                  'cashflow': 0,
                                  'high': 0,
                                  'low': np.inf,
                                  'avg': 0,
                                  'slippage': 0},
                            index=[timestamps[0][:-10]])
        
        for t in tqdm(range(timespan), leave=True):
            assert isinstance(orderbooks[t], pd.DataFrame)
            df = orderbooks[t].copy()
            df = self.adjust_orderbook(df)
            
            if t == 0:
                info.ASK = df[df.Type == 'ask'].iloc[0].Price
                info.BID = df[df.Type == 'bid'].iloc[-1].Price
                info.SPREAD = (info.ASK - info.BID)
                info.CENTER = log_mean(info.ASK.values[0], info.BID.values[0])           

               
            if volume==0:
                # Do nothing!
                return df
                
            if must_trade and t == timespan-1:
                # must sell all remaining shares. Place Market Order!
                print("Run out of time (t={}).\nSell remaining {:.4f}/{:.4f} shares for current market order price".format(t,
                                                                                                  info.volume_left.values[0],
                                                                                                  info.VOLUME.values[0]))
                limit = None
                
            # perform trade
            trade_result = self.__perform_trade(df, info.volume_left.values[0], limit)
           
            info.cashflow += trade_result['cashflow']
            info.volume_traded += trade_result['volume']
            info.volume_left = round((info.VOLUME - info.volume_traded).values[0], PRECISION)
            
            if len(self.last_trade) > 0:
                current_high = max([float(x) for x in self.last_trade.keys()])
                if current_high > info.high.values[0]:
                    info.high = current_high
                    
                current_low = min([float(x) for x in self.last_trade.keys()])
                if current_low < info.low.values[0]:
                    info.low = current_low
            
            
            if abs(info.volume_left.values[0]) == 0:
                print("No shares left at t={}, Done!".format(t))
                break
                
        info.avg = abs(info.cashflow / info.volume_traded)
        
        if volume > 0:
            # buy from market
            info.slippage = info.cashflow + info.volume_traded * info.ASK
        elif volume < 0:
            # sell to market
            info.slippage = info.cashflow + info.volume_traded * info.BID
        
        self.history = self.history.append(info, ignore_index=False)
        display(self.history)

        if verbose:
            self.summarize(df)
            
            print("Traded {:.4f}/{:.4f} shares for {}, {:.4f} shares left".format(info.volume_traded.values[0],
                                                              (info.volume_traded + info.volume_left).values[0],
                                                              info.cashflow.values[0],
                                                              info.volume_left.values[0]))
            
        # self.summarize(df)
        return self.adjust_orderbook(df)
        
    
    
    
    ## deprecated ... delete?
    def trade(self, orderbook, volume, limit=None, verbose=True):
        assert isinstance(orderbook, pd.DataFrame)
        assert isinstance(volume, float) or isinstance(volume, int)
        assert isinstance(limit, float) or isinstance(limit, int) or not limit
        if limit:
            assert limit > 0
        assert isinstance(verbose, bool)
        
        df = orderbook.copy()
        df = self.adjust_orderbook(df)
        
        if not self.initial_center_log:
            # very first trade
            self.initial_ask = df[df.Type == 'ask'].iloc[0].Price
            self.initial_bid = df[df.Type == 'bid'].iloc[-1].Price
            self.initial_center_log = log_mean(self.initial_ask, self.initial_bid)
        
        self.__perform_trade(df, volume, limit)
        
        if verbose:
            self.summarize(df)
            
            # print("Traded {:.4f}/{:.4f} shares for {}".format(self.volume_traded, self.volume_traded + self.volume_nottraded,
            #                                                       self.cashflow))
            
        # self.summarize(df)
        return self.adjust_orderbook(df)