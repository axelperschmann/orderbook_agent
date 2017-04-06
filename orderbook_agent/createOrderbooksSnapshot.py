from helper.manage_orderbooks import *
import os
from datetime import datetime

currency_pair = 'USDT_BTC'
range_factor = 1.15

folder = '../../data/'
files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(folder)) for f in fn]
datafiles = sorted([f for f in files if f.endswith('.log.gz')])
print("Number of datafiles to extract from: {}".format(len(datafiles)))

filename = "../../data/snapshots/orderbooks_{}_range{}_snapshot{}.dict".format(
    currency_pair, range_factor, str(datetime.now().isoformat())[:-10])
print(filename)

extract_orderbooks_for_one_currencypair(datafiles, currency_pair='USDT_BTC', outfile=filename,
                                        range_factor=range_factor, pricelevel_precision=2)
