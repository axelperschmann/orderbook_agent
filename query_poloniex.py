import pandas as pd
import calendar
from datetime import datetime
import urllib2
import json

def timestamp_to_unix(timestamp, timeformat='%Y-%m-%d %H:%M:%S'):
    assert isinstance(timestamp, str) or isinstance(timestamp, unicode)
    d = datetime.strptime(timestamp, timeformat)
    return (int(calendar.timegm((d.timetuple()))))


# e.g. hist = query_historic_trade_activity('USDT_BTC', querytime=u'2016-12-15 12:20:00')
def query_historic_trade_activity(currencypair, querytime, intervals=[60, 600, 3600], timeformat='%Y-%m-%d %H:%M:%S'):
    assert isinstance(currencypair, str) or isinstance(currencypair, unicode)
    assert isinstance(querytime, str) or isinstance(querytime, unicode)
    assert isinstance(intervals, list)
    assert len(intervals) > 0 and isinstance(intervals[0], int)
    coll = pd.DataFrame([], columns=['sum_sell', 'sum_buy'])
    
    max_interval = max(intervals)
    
    # query trade history from poloniex
    end = timestamp_to_unix(querytime, timeformat)
    start = end - max_interval

    query = 'https://poloniex.com/public?command=returnTradeHistory&currencyPair=' + currencypair + '&start=' + str(start) + '&end=' + str(end)
    ret = urllib2.urlopen(urllib2.Request(query))
    results = json.loads(ret.read())    
    df = pd.DataFrame.from_dict(results)

    # convert date from string to unix
    df['date'] = [timestamp_to_unix(x) for x in df.date.values]

    for length in intervals:
        sub = df[df.date > end - length]

        stat = pd.DataFrame([[0,0]], columns=['sum_sell', 'sum_buy'], index=[length])
        
        if len(sub[sub.type=='sell']) > 0:
            stat['sum_sell'] = sub[sub.type=='sell'].amount.astype(float).sum()
        if len(sub[sub.type=='buy']) > 0:
            stat['sum_buy'] = sub[sub.type=='buy'].amount.astype(float).sum()
        coll = pd.concat([coll, stat])
    return coll
