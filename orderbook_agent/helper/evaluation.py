import multiprocessing
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

import sys
sys.path.append('..')
from helper.orderbook_container import OrderbookContainer
from helper.orderbook_trader import OrderbookTradingSimulator

def plot_evaluation_costs(experiments, vlines=None, name=None, ylim=None, showfliers=False, hline=True, outfile=None):
        assert isinstance(hline, (str, bool)) or hline is None
                
        experiments.plot.box(showmeans=True, color={'medians':'green'}, figsize=(12, 8), showfliers=showfliers)
        
        if vlines is None:
            simpleStrategies_count = np.sum([elem.isnumeric() for elem in experiments.columns])
            vlines = []
            if hline in experiments.columns:
                vlines = [1]
            vlines = vlines + [experiments.shape[1]-simpleStrategies_count]

        for line in vlines:
            plt.axvline(line + 0.5, color='black')
        
        if hline in experiments.columns:
            plt.axhline(experiments[hline].mean(), color='red', alpha=0.5, linewidth=1, linestyle='--', label='min mean')
            plt.axhline(experiments[hline].median(), color='green', alpha=0.5, linewidth=1, linestyle='--', label='min median')


        title = "{} trading periods  \n{} - {}".format(len(experiments), experiments.index[0], experiments.index[-1])
        if name is not None:
            title = "{}: {}".format(name, title)
        
        plt.suptitle(title)
        plt.xlabel("Strategies")
        plt.ylabel("Occured costs")
        # plt.savefig("boxplot_train.pdf")
        plt.xticks(rotation=70)
        if ylim is not None:
            plt.ylim(ylim)    
        plt.legend(loc='best')
        
        if outfile:
            plt.savefig(outfile)
            print("Saved figure to '{}'".format(outfile))
            plt.close()
        else:
            plt.show()

def evaluate(testdata, agents=None, evaluate_actions=[], custom_strategies=None,
                 costs=None, name=None, verbose=False, baseline=None):
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

        if costs is not None:
            costs = costs
        else:
            costs = pd.DataFrame([])

        num_cores = multiprocessing.cpu_count()
        strategy_count = len(evaluate_agents) + len(evaluate_actions)
        if custom_strategies:
            strategy_count += len(custom_strategies)
        print("Start parallel evalutions of {} strategies over {} tradingperiods. (num_cores={})".format(strategy_count, len(testdata), num_cores))

        results = Parallel(n_jobs=num_cores, verbose=5)(delayed(evaluate_window)(
            evaluate_agents=evaluate_agents, baseline=baseline_agent,
            custom_strategies=custom_strategies,
            evaluate_actions=evaluate_actions,
            window=window, verbose=verbose) for window in testdata)
        
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
        ots = OrderbookTradingSimulator(orderbooks=window, volume=baseline_agent.V, consume=baseline_agent.consume, tradingperiods=baseline_agent.T,
                                         period_length=baseline_agent.period_length)
        lim_increments = init_center * (baseline_agent.lim_stepsize / 100)

        for agentname in evaluate_agents.keys():
            agent = evaluate_agents[agentname]

            ots.reset()
            ots.period_length = agent.period_length
            for t in range(0, agent.T):

                time_left = agent.T - t
                timepoint = t*agent.period_length
                
                ob_now = window[timepoint]
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

                action, action_idx = agent.get_action(state)
                
                if agent.limit_base == 'agression':
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
            limit = window[0].get_center() * (1. + (action/100.))

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

        return costs, slippage