import pandas as pd
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import random
import matplotlib.pyplot as plt

# import numpy as np
from IPython.display import display

from helper.orderbook_trader import *
from helper.general_helpers import gauss_window

def collect_samples_backward(brain, window):
    ots = OrderbookTradingSimulator(orderbooks=window, volume=brain.V, consume=brain.consume,
                                    tradingperiods=brain.T, period_length=brain.period_length)
    lim_increments = ots.initial_center * (brain.lim_stepsize / 100)

    if brain.consume=='volume':
        initial_marketprice, initial_limitWorst = window[0].get_current_price(volume=brain.V)
        initial_limitAvg = initial_marketprice / brain.V
        market_slippage = initial_limitAvg - ots.initial_center
    elif brain.consume=='cash':
        initial_marketShares, initial_limitWorst = window[0].get_current_sharecount(cash=brain.V)
        initial_limitAvg = brain.V / initial_marketShares

    # empty DataFrame to store samples
    samples = pd.DataFrame([], columns=brain.columns)
    samples['action_idx'] = samples.action_idx.astype(int)

    for tt in tqdm(range(brain.T)[::-1], desc='timepoints'):
        timepoint = brain.period_length*tt
        timepoint_next = min((tt+1)*brain.period_length, (brain.period_length*brain.T)-1)
        
        time_left = brain.T-tt

        ob_now = window[timepoint]
        ob_next = window[timepoint_next]
        timestamp = ob_now.timestamp

        price_incScale = int(ob_now.get_center()/lim_increments)

        ask = ob_now.get_ask()

        for vol in brain.volumes:
            state = brain.generate_state(time_left=time_left, 
                                         volume_left=vol,
                                         orderbook=ob_now)

            for action_idx, action in enumerate(brain.actions):
                ots.reset(custom_starttime=tt, custom_startvolume=vol)

                if brain.limit_base == 'currAsk':
                    limit = ob_now.get_ask() * (1. + (action/100.))
                    summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
                elif brain.limit_base == 'incStepUnits':
                    price_incScale = int(ob_now.get_center()/lim_increments)
                    limit = lim_increments * (price_incScale + action)
                    summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
                elif brain.limit_base =='agression':
                    summary = ots.trade(agression_factor=action, verbose=False, extrainfo={'ACTION':action})
                else:
                    raise NotImplementedError

                if brain.consume=='volume':
                    volume_left = ots.volume
                    volume_traded = ots.history.volume_traded.values[-1]
                elif brain.consume=='cash':
                    volume_left = ots.cash
                    volume_traded = ots.history.cash_traded.values[-1]

                cost = ots.history.cost.values[-1]
                avg = ots.history.avg[-1]

                new_state = brain.generate_state(time_left=time_left-1,
                                                 volume_left=volume_left,
                                                 orderbook=ob_next)
                new_sample = brain.generate_sample(
                        state=state,
                        action=action,
                        action_idx=action_idx,
                        cost=cost,
                        timestamp=ob_now.timestamp,
                        avg=ots.history.avg[-1],
                        initial_center=ots.initial_center,
                        new_state=new_state
                    )
                samples = pd.concat([samples, new_sample], axis=0, ignore_index=True)

    return samples



def collect_samples_forward(brain, window, epochs, guiding_agent=None, random_start=True, exploration=0):

    ots = OrderbookTradingSimulator(orderbooks=window, volume=brain.V, consume=brain.consume,
                                    tradingperiods=brain.T, period_length=brain.period_length)
    lim_increments = ots.initial_center * (brain.lim_stepsize / 100)

    if brain.consume=='volume':
        initial_marketprice, initial_limitWorst = window[0].get_current_price(volume=brain.V)
    elif brain.consume=='cash':
        initial_marketShares, initial_limitWorst = window[0].get_current_sharecount(cash=brain.V)
    
    # empty DataFrame to store samples
    samples = pd.DataFrame([], columns=brain.columns)
    samples['action_idx'] = samples.action_idx.astype(int)

    for e in range(epochs):
        epsilon = 1.0/20**(e/epochs)  # vanishing exploration-rate: 1.0 to 0.05
        # print("{}: epsilon = {}".format(e, epsilon))

        volume = brain.V
        startpoint = 0
        if random_start and random.random() < 1.:
            # randomly start at other states in environment
            # volume = random.randint(1, brain.V)
            startpoint = random.randint(0, brain.T-1)
        ots.reset(custom_starttime=startpoint, custom_startvolume=volume)
        
        for t in range(startpoint, brain.T):
            time_left = brain.T - t

            ob_now = window[ots.t]
            ob_next = window[min(ots.t+brain.period_length, len(window)-1)]
            ob_next_next = window[min(ots.t+(brain.period_length*2), len(window)-1)]

            state = brain.generate_state(time_left=time_left, 
                                         volume_left=ots.get_units_left(),
                                         orderbook=ob_now, orderbook_cheat=ob_next)

            ### epsilon-greedily ask guiding_agent for optimal action
            if guiding_agent is not None and (random.random() < epsilon or brain.model is None):
                # compute appropriate state that fits to guiding agent.
                # This state might be different due to volume rounding and more/fewer state_variables
                state_guide = guiding_agent.generate_state(time_left=time_left, 
                                         volume_left=ots.get_units_left(),
                                         orderbook=ob_now, orderbook_cheat=ob_next)
                
                action, action_idx = guiding_agent.get_action(state_guide)
                if exploration > 0:
                    # explore neighbourhood of proposed action
                    gauss = gauss_window(guiding_agent.actions, a_idx=action_idx, std=exploration)
                    action_idx = np.random.choice(range(len(guiding_agent.actions)), p=gauss)
                    action = guiding_agent.actions[action_idx]
            else:
                action, action_idx = brain.get_action(state)    
                if exploration > 0:
                    # explore neighbourhood of proposed action
                    gauss = gauss_window(brain.actions, a_idx=action_idx, std=exploration)
                    action_idx = np.random.choice(range(len(brain.actions)), p=gauss)
                    action = brain.actions[action_idx]

            if brain.limit_base == 'currAsk':
                limit = ob_now.get_ask() * (1. + (action/100.))
                summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
            elif brain.limit_base == 'incStepUnits':
                price_incScale = int(ob_now.get_center()/lim_increments)
                limit = lim_increments * (price_incScale + action)
                summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
            elif brain.limit_base =='agression':
                summary = ots.trade(agression_factor=action, verbose=False, extrainfo={'ACTION':action})
            else:
                raise NotImplementedError

            new_state = brain.generate_state(time_left=time_left-1,
                                             volume_left=ots.get_units_left(),
                                             orderbook=ob_next, orderbook_cheat=ob_next_next)

            cost = ots.history.cost.values[-1]
            
            new_sample = brain.generate_sample(
                    state=state,
                    action=action,
                    action_idx=action_idx,
                    cost=cost,
                    timestamp=ob_now.timestamp,
                    avg=ots.history.avg[-1],
                    initial_center=ots.initial_center,
                    new_state=new_state
                )
            samples = pd.concat([samples, new_sample], axis=0, ignore_index=True)
            
            if ots.check_is_done():
                break
    return samples
