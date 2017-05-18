from tqdm import tqdm
import sys
sys.path.append('orderbook_agent/')

from helper.manage_orderbooks import *
import os
from datetime import datetime

currency_pairs = ['USDT_BTC'] # , 'BTC_ETH', 'BTC_XMR', 'BTC_XRP', 'BTC_FCT', 'BTC_NAV', 'BTC_DASH', 'BTC_MAID', 'BTC_ZEC']

months = ['2017-01', '2017-02']  # '2016-11', '2016-12', 

for currency_pair in tqdm(currency_pairs):
    for month in tqdm(months):

        # if currency_pair == 'USDT_BTC':
        #     precision_level = 3
        # else:
        #     precision_level = precision

        folder = '../data/'
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
        datafiles = sorted([f for f in files if (f.endswith('.log.gz') and (month in f))])
        print("Number of datafiles to extract from: {} (month: {})".format(len(datafiles), month))

        filename = "../data/history/history_{}_{}.csv".format(month, currency_pair)

        history = extract_orderbooks_history(data, currency_pair, outfile=filename, overwrite=True)