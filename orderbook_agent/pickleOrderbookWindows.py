import sys
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

sys.path.append('../Runs')
from helper.manage_orderbooks import OrderbookEpisodesGenerator, plot_episode
from helper.orderbook_trader import OrderbookTradingSimulator
from agents.QTable_Agent import QTable_Agent

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

decision_points = 4
period_length = 15


path = "/home/axel/data/small"
data_files = [f for f in os.listdir(path) if f.endswith('.dict')]
print(data_files)

for inputfile in tqdm(data_files):
    episodes_train = OrderbookEpisodesGenerator(filename=os.path.join(path, inputfile),
                                            episode_length=decision_points*period_length)
    
    print(inputfile, len(episodes_train))
    outputfile = "/home/axel/notebooks/orderbook_agent/orderbook_agent/cached_windows/{}.p".format(inputfile.split(".")[0])
    
    print("Outputfile: {}".format(outputfile))

    data = list(episodes_train)
    pickle.dump( data, open( os.path.join(path, outputfile), "wb" ) )