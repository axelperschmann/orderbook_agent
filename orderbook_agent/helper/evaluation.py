import multiprocessing
from joblib import Parallel, delayed
import dill, pickle
from functools import partial
import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm import tqdm

import sys
sys.path.append('..')
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import OrderbookTradingSimulator
from agents.NN_Agent import RLAgent_NN


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def plot_evaluation_costs(experiments, vlines=None, name=None, ylim=None, showfliers=False, hline=True, outfile=None, verbose=False):
        assert isinstance(hline, (str, bool)) or hline is None
        
        fig, ax = plt.subplots(figsize=(8, 4))
        experiments.plot.box(showmeans=True, color={'medians':'green'}, ax=ax, showfliers=showfliers)
        
        if verbose:
            perf = pd.DataFrame(experiments.mean(), columns=['slippage'])
            perf['med'] = experiments.median()
            perf['std'] = experiments.std()
            perf['perf_2'] = (experiments.mean() / experiments["2"].mean() * 100-100).map('{:,.2f}%'.format)

            perf['perf_4'] = (experiments.mean() / experiments["4"].mean() * 100-100).map('{:,.2f}%'.format)
            perf['perf_M'] = (experiments.mean() / experiments["MarketOrder"].mean() * 100-100).map('{:,.2f}%'.format)
            # perf['perf_VolTime'] = (experiments.mean() / experiments["VolTime"].mean() * 100-100).map('{:,.2f}%'.format)
        
            perf = round(perf, 2)
            print(experiments.mean().argmin())
            display(perf)
            print(perf.to_latex())
        
        if vlines is None:
            simpleStrategies_count = np.sum([is_number(elem) for elem in experiments.columns])
            
            if "MarketOrder" in experiments.columns:
                simpleStrategies_count += 1
            vlines = []
            if hline in experiments.columns:
                vlines = [1]
            vlines = vlines + [experiments.shape[1]-simpleStrategies_count]

        for line in vlines:
            plt.axvline(line + 0.5, color='black')
        
        if hline in experiments.columns:
            plt.axhline(experiments[hline].mean(), color='red', alpha=0.5, linewidth=1, linestyle='--', label="mean '{}':  {:1.4f}".format(hline, experiments[hline].mean()))
            plt.axhline(experiments[hline].median(), color='green', alpha=0.5, linewidth=1, linestyle='--', label="median '{}':  {:1.4f}".format(hline, experiments[hline].median()))
        elif hline =='min':
            hline = experiments.mean().argmin()
            plt.axhline(experiments[hline].mean(), color='red', alpha=0.5, linewidth=1, linestyle='--', label="mean '{}':  {:1.4f}".format(hline, experiments[hline].mean()))
            plt.axhline(experiments[hline].median(), color='green', alpha=0.5, linewidth=1, linestyle='--', label="median '{}':  {:1.4f}".format(hline, experiments[hline].median()))


        title = "{} trading periods  \n{} - {}".format(len(experiments), experiments.index[0], experiments.index[-1])
        if name is not None:
            title = "{}: {}".format(name, title)
        
        plt.suptitle(title)
        plt.xlabel("Strategies")
        plt.ylabel("Occured costs")
        # plt.savefig("boxplot_train.pdf")
        plt.xticks(rotation=90)
        if ylim is not None:
            plt.ylim(ylim)    
        plt.legend(loc='upper left')
        
        if outfile:
            plt.gcf().subplots_adjust(bottom=0.23)
            plt.savefig(outfile)
            print("Saved figure to '{}'".format(outfile))
            plt.close()
        else:
            plt.show()

def evaluate(testdata, agents=None, evaluate_actions=[], custom_strategies=None,
                 name=None, limit_num_cores=0, verbose=False, baseline=None):
        if limit_num_cores==0:
            limit_num_cores = multiprocessing.cpu_count()

        agents = agents.copy()

        # reorder agents, such that baseline-agent is the first one.
        evaluate_agents = {}
        if baseline is not None:
            evaluate_agents[baseline] = agents.pop(baseline)
            baseline_agent = baseline
        else:
            baseline_agent = list(agents.keys())[0]    
        if isinstance(agents, dict):
            evaluate_agents.update(agents)

        print(evaluate_agents.keys())
        strategy_count = len(evaluate_agents) + len(evaluate_actions)
        if custom_strategies:
            strategy_count += len(custom_strategies)
        
        if limit_num_cores > 1:
            print("Start parallel evalutions of {} strategies over {} tradingperiods. (num_cores={})".format(strategy_count, len(testdata), limit_num_cores))

            results = Parallel(n_jobs=limit_num_cores, verbose=15)(delayed(evaluate_window)(
                evaluate_agents=evaluate_agents, baseline=baseline_agent,
                custom_strategies=custom_strategies,
                evaluate_actions=evaluate_actions,
                window=window, verbose=verbose) for window in testdata)
        else:
            print("Start non-parallel evalutions of {} strategies over {} tradingperiods.".format(strategy_count, len(testdata)))
            results = []
            for window in tqdm(testdata, leave=False, desc='testdata'):
                results.append(
                    evaluate_window(
                        evaluate_agents=evaluate_agents, baseline=baseline_agent,
                        custom_strategies=custom_strategies,
                        evaluate_actions=evaluate_actions,
                        window=window, verbose=verbose)
                )
        print("done")
        costs = pd.concat([elem[0] for elem in results], axis=0, ignore_index=False)
        slippages = pd.concat([elem[1] for elem in results], axis=0, ignore_index=False)
            
        return costs, slippages


def evaluate_window(evaluate_agents, baseline, custom_strategies, evaluate_actions, window,
            verbose=False):

        baseline_agent = evaluate_agents[baseline]

        costs = pd.DataFrame([])
        slippage = pd.DataFrame([])

        index = window[0].timestamp
        init_center = window[0].get_center()
        
        ## Learned strategy
        try:
            ots = OrderbookTradingSimulator(orderbooks=window, volume=baseline_agent.V, consume=baseline_agent.consume, tradingperiods=baseline_agent.T,
                                             period_length=baseline_agent.period_length)
            lim_increments = init_center * (baseline_agent.lim_stepsize / 100)

            for agentname in evaluate_agents.keys():
                
                agent = evaluate_agents[agentname]
                if isinstance(agent, dict):
                    # can't pickle.dumps or dill.dumps tensorflow graphs and sessions ... this is a hacky workaround:
                    agent = RLAgent_NN.load(agent_name=agent.get('agent_name'), path=agent.get('path'), ignore_samples=True)
                    

                ots.reset()
                ots.period_length = agent.period_length
                for t in range(0, agent.T):

                    time_left = agent.T - t
                    timepoint = t*agent.period_length
                    
                    ob_now = window[timepoint]
                    if 'BT' in agentname:
                        ob_now = ots.masterbook
                    ob_next = window[min(timepoint+baseline_agent.period_length, len(window)-1)]
                    price_incScale = int(round(ob_now.get_center()/lim_increments, 3))
                    
                    if baseline_agent.consume=='volume':
                        volume = float(ots.volume)
                    elif baseline_agent.consume=='cash':
                        volume = float(ots.cash)
                    if hasattr(agent, 'volumes_base') and not agent.interpolate_vol:
                        # discretize volume through rounding (needed for QLookupTable)
                        volume = agent.round_custombase(volume, non_zero=True)
        
                    state = agent.generate_state(time_left=time_left, 
                                                volume_left=volume,
                                                orderbook=ob_now,
                                                orderbook_cheat=ob_next)

                    if agent.agent_type=='NN_Agent':
                        state = pd.DataFrame(state).T

                    action, action_idx = agent.get_action(state)
                    
                    if agent.limit_base == 'currAsk':
                        limit = ob_now.get_ask() * (1. + (action/100.) * agent.lim_stepsize)
                        summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
                    elif agent.limit_base == 'currBid':
                        limit = ob_now.get_bid() * (1. + (action/100.) * agent.lim_stepsize)
                        summary = ots.trade(limit=limit, verbose=False, extrainfo={'ACTION':action})
                    elif agent.limit_base == 'agression':
                        summary = ots.trade(agression_factor=action, extrainfo={'ACTION':action})
                    elif agent.limit_base == 'incStepUnits':
                        limit = lim_increments * (price_incScale + action)
                        summary = ots.trade(limit=limit, extrainfo={'ACTION':action})
                    else:
                        limit = agent.action_to_limit(action, init_center=init_center, current_ask=ob_now.get_ask())
                        summary = ots.trade(limit=limit, extrainfo={'ACTION':action})
                    if ots.summary['done']:
                        break

                if verbose:
                    print(agentname, agent.limit_base, lim_increments)
                    print(" cost", ots.history.cost.sum())
                    print(" slippage", ots.history.slippage.sum())
                    print(agent.actions)
                    display(ots.history)


                costs.loc[index, agentname] = ots.history.cost.sum()
                slippage.loc[index, agentname] = ots.history.slippage.sum()


            if custom_strategies is not None:
                for strategyname in custom_strategies.keys():

                    strategy = custom_strategies[strategyname]
                    ots.reset()
                    ots.period_length = int(len(window) / len(strategy))

                    for t in range(agent.T):
                        timepoint = t*baseline_agent.period_length
                        ob_now = window[timepoint]
                        price_incScale = int(round(ob_now.get_center()/lim_increments, 3))

                        action = strategy[t]            
                        limit = lim_increments * (price_incScale + action)

                        summary = ots.trade(limit=limit, extrainfo={'ACTION':action})

                        if ots.summary['done']:
                            break

                    if verbose:
                        print(strategyname, agent.limit_base, ots.history.cost.sum())
                        print(" cost", ots.history.cost.sum())
                        print(" slippage", ots.history.slippage.sum())
                        display(ots.history)
                    costs.loc[index, strategyname] = ots.history.cost.sum()
                    slippage.loc[index, strategyname] = ots.history.slippage.sum()

                
            for action in evaluate_actions:
                ots.reset()
                ots.period_length = baseline_agent.period_length
                # limit = window[0].get_center() * (1. + (action/100.))
                if action=='MarketOrder':
                    limit = None
                else:
                    price_incScale = int(round(window[0].get_center()/lim_increments, 3))
                    limit = lim_increments * (price_incScale + action)

                for t in range(0, baseline_agent.T):
                    ots.trade(limit = limit, extrainfo={'ACTION':action})

                    if ots.summary['done']:
                        break
                costs.loc[index, str(action)] = ots.history.cost.sum()
                slippage.loc[index, str(action)] = ots.history.slippage.sum()

                if verbose:
                    print("action", action, limit)
                    print(" cost", ots.history.cost.sum())
                    print(" slippage", ots.history.slippage.sum())
                    display(ots.history)
                # if verbose:
                #     display(ots.history)
        except AssertionError as err:
            print("Couldn't trade {}: '{}'".format(window[0].timestamp, err))
        except ValueError as err:
            # print("Skipping orderbook window:", window[0].timestamp, err)
            pass

        return costs, slippage

if __name__ == '__main__':
    pass