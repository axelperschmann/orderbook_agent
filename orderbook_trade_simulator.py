import pandas as pd
import numpy as np
from manage_orderbooks import log_mean, orderbook_preview
from IPython.display import display
from tqdm import tqdm

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
        if self.volume_traded != 0:
            print("Cashflow: {} (Avg: {}, Low: {}, High: {})".format(self.cashflow,
                                                                     self.cashflow / self.volume_traded,
                                                                     self.lowest_price, self.highest_price))
        else:
            print("Cashflow: 0")
        print("Traded: {}/{} shares".format(self.volume_traded, (self.volume_traded + self.volume_nottraded)))
        
        print("Spread: {} (bid: {}, ask: {})".format(ask-bid, bid, ask))
        
        print("Adjusted Orderbook:")
        display(orderbook_preview(df_adj, 3))
        
    
    def __init__(self, T=5):
        self.t = 0
        self.T = T
        
        self.purchase_history = {}  # {'705.45': 3.1721811199999999}
        self.sell_history = {}  # {'703.70': 0.001}

        self.last_trade = {}
        self.volume_of_last_trade = 0
        self.volume_of_last_trade_period = 0
        self.volume_traded = 0
        self.volume_nottraded = 0
        self.cashflow = 0
        self.highest_price = 0
        self.lowest_price = np.inf
    
        self.initial_bid = None
        self.initial_ask = None
        self.initial_center = None
    
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
        # df['VolumeAcc'].loc[df.Type=='bid'] = (df[df.Type=='bid'].Volume)[::-1].cumsum().values[::-1]
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
        
        self.t += 1
        trade_summary = {}
        self.volume_of_last_trade = 0
        
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
            
            self.cashflow += current_order_volume * order.Price * order_direction
            self.volume_traded -= current_order_volume * order_direction
            self.volume_of_last_trade -= current_order_volume * order_direction

            volume += current_order_volume * order_direction
            if order.Price > self.highest_price:
                self.highest_price = order.Price
            if order.Price < self.lowest_price:
                self.lowest_price = order.Price
            
            if volume == 0:
                break
        self.volume_nottraded = volume
        
        self.last_trade = trade_summary
        
    def trade_timespan(self, orderbooks, volume, limit, verbose=True, timespan=1, must_trade=False):
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
        
        for t in tqdm(range(timespan)):
            assert isinstance(orderbooks[t], pd.DataFrame)
            df = orderbooks[t].copy()
            df = self.adjust_orderbook(df)
            # display(orderbook_preview(df,3))
            # print("Trade:", volume, volume-self.volume_traded)
               
            if volume==0:
                # Do nothing!
                return df
            elif volume > 0:
                # buy from market
                order_direction = -1
            elif volume < 0:
                # sell to market
                order_direction = 1
                
            if must_trade and t == timespan-1:
                # must sell all remaining shares. Place Market Order!
                
                print("Run out of time (t={}). Sell remaining {:.4f}/{:.4f} shares for current market order price".format(t, volume + order_direction*volume_traded, volume))
                limit = None
                
            self.__perform_trade(df, volume + volume_traded*order_direction, limit)
            self.volume_of_last_trade_period -= self.volume_of_last_trade * order_direction
            
            volume_traded -= self.volume_of_last_trade * order_direction

            if abs(volume)-abs(volume_traded) == 0:
                print("No shares left at t={}, Done!".format(t))
                break

        if verbose:
            self.summarize(df)
            
            print("Traded {:.4f}/{:.4f} shares for {}".format(self.volume_traded, self.volume_traded+ self.volume_nottraded,
                                                                  self.cashflow))
            
        # self.summarize(df)
        return self.adjust_orderbook(df)
        
    
    def trade(self, orderbook, volume, limit=None, verbose=True):
        assert isinstance(orderbook, pd.DataFrame)
        assert isinstance(volume, float) or isinstance(volume, int)
        assert isinstance(limit, float) or isinstance(limit, int) or not limit
        if limit:
            assert limit > 0
        assert isinstance(verbose, bool)
        
        df = orderbook.copy()
        df = self.adjust_orderbook(df)
        
        if not self.initial_center:
            # very first trade
            self.initial_ask = df[df.Type == 'ask'].iloc[0].Price
            self.initial_bid = df[df.Type == 'bid'].iloc[-1].Price
            self.initial_center_log = log_mean(self.initial_ask, self.initial_bid)
        
        self.volume_of_last_trade = 0
        self.__perform_trade(df, volume, limit)
        
        if verbose:
            self.summarize(df)
            
            print("Traded {:.4f}/{:.4f} shares for {}".format(self.volume_traded, self.volume_traded + self.volume_nottraded,
                                                                  self.cashflow))
            
        # self.summarize(df)
        return self.adjust_orderbook(df)