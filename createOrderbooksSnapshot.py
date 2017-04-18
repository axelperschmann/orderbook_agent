from tqdm import tqdm
import sys
sys.path.append('1_version2/')

from helper.manage_orderbooks_v2 import *
import os
from datetime import datetime

currency_pairs = ['USDT_BTC', 'BTC_ETH', 'BTC_XMR', 'BTC_XRP', 'BTC_FCT', 'BTC_NAV', 'BTC_DASH', 'BTC_MAID', 'BTC_ZEC']
# currency_pair = 'USDT_BTC'
range_factor = None  # 1.2
range_volume = 100
precision = 7

months = ['2016-11', '2016-12', '2017-01', '2017-02']

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

        filename = "../data/snapshots/monthly_maxVol100/obs_{}_{}_maxVol{}.dict".format(month, currency_pair, range_volume)

        extract_orderbooks_for_one_currencypair(datafiles, currency_pair=currency_pair,
                                                outfile=filename, range_factor=range_factor,
                                                range_volume=range_volume,
                                                pricelevel_precision=precision)
