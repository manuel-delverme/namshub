import csv
import datetime
import queue
import random
import sys
import time
from collections import OrderedDict

import gym
import numpy as np
import tqdm
from gym import spaces
from gym.utils import seeding

from commons.disk_utils import disk_cache
from commons.tf_utils import get_summary

VERY_SMALL_NUMBER = 0.01


class MarketActions(object):
    SELL = 0
    BUY = 1
    KEEP = 2


class Observations(object):
    BUY = 0
    SELL = 1
    VOLUME = 2
    INVESTED = 3
    LIQUID = 4


class CachedMarketData(object):
    def __init__(self, original_dataset):
        self.S = original_dataset

    def __len__(self):
        return len(self.S)

    def __getitem__(self, time_idx):
        # try:
        price, volume = self.S[time_idx]
        # except IndexError:
        # price, volume = 0, 0
        return price * 1.0001, price, volume


class MarketData(object):
    def __init__(self, exchange="coinbase"):
        if exchange == "coinbase":
            import gdax
            public_client = gdax.PublicClient()

            def get_data_fn(epoch):
                tickers = None
                for sleep_time in [0, 1, 10, 20, 40]:
                    time.sleep(sleep_time)
                    tickers = public_client.get_product_ticker(product_id='BTC-USD')
                    self.minimum_trade = tickers['size']
                    buy_price = tickers['ask']
                    sell_price = tickers['bid']
                    volume = tickers['volume']
                    return buy_price, sell_price, volume
                else:
                    raise Exception(tickers['error'])

            initial_time = datetime.datetime.utcfromtimestamp(int(public_client.get_time()['epoch']))
        else:
            import krakenex
            pair = 'XXBTZUSD'
            # NOTE: for the (default) 1-minute granularity, the API seems to provide data up to 12 hours old only!
            endpoint = krakenex.API()
            resp = endpoint.query_public("Time")["result"]["unixtime"]
            initial_time = datetime.datetime.utcfromtimestamp(int(resp))

            def get_data_fn(epoc):
                since = epoc - datetime.timedelta(hours=11)
                response = endpoint.query_public('OHLC', req={
                    'pair': pair, 'since': since
                })
                return response

        self.get_data = get_data_fn
        self.initial_time = initial_time
        self.last_query_time = datetime.datetime.utcnow() - datetime.timedelta(hours=12)
        self.delay = datetime.timedelta(seconds=60)
        self.S = {}

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, time_idx):
        time_idx = self.initial_time + datetime.timedelta(minutes=time_idx)
        try:
            item = self.S[time_idx]
        except KeyError:
            since = datetime.datetime.utcnow() - datetime.timedelta(hours=12)

            time_since_last_query = datetime.datetime.utcnow() - self.last_query_time
            time_to_wait = self.delay - time_since_last_query
            if time_to_wait > datetime.timedelta(seconds=0):
                print("sleeping", time_to_wait.seconds, "+1 sec")
                time.sleep(time_to_wait + 1)

            resp = self.get_data(time_idx)
            self.S[time_idx] = np.array(tuple(map(float, resp)))
            # for result in response:
            #    tick_time, open, high, low, close, vwap, volume, count = result
            #    tick_time = datetime.datetime.utcfromtimestamp(tick_time)
            #    if tick_time not in self.S:
            #         self.S[tick_time] = np.array((open, high, low, close, vwap, volume, count))
            #    else:
            #        # TODO: average?, vote?
            #        self.S[tick_time] += np.array((open, high, low, close, vwap, volume, count))
            #         # datetime.datetime.fromtimestamp(time).strftime('%d/%m/%Y %H:%M:%S %Z'),
            item = self.S[time_idx]
        return item


class BitEnv(gym.Env):
    def __init__(self, max_loss, max_gain, bitcoin_fraction, transaction_fee, max_timesteps, initial_budget,
                 history_length, verbose, use_historic_data, logger_path='logs/logs.csv'):
        # TODO a counter should be a property of the action market action class
        self.taken_actions = {
            MarketActions.SELL: 0,
            MarketActions.BUY: 0,
            MarketActions.KEEP: 0,
        }
        self._logger_path = logger_path
        self._write_head = True
        self.max_loss = max_loss
        self.max_gain = max_gain
        self.bitcoin_fraction = bitcoin_fraction
        self.transaction_fee = transaction_fee
        self.sim_ep_max_length = self.sim_ep_timestep_bound = max_timesteps
        self.initial_budget = self.liquid_budget = initial_budget
        self.ep_summary = get_summary()
        # max_loss = .20
        # max_gain = .50
        # self.is_greedy = (max_loss, max_gain)
        # self.range = 1000  # +/- value the randomly select number can be between
        # self.bounds = 5000  # Action space bounds

        # self.number = 0
        # step counter
        self.ledger = []
        self.sim_time, self.invested_budget = 0, 0
        # sim_ep_length: hyper parameter, sim_time_max is a bound on the end of the episode
        # self.liquid_budget = self.initial_budget

        if use_historic_data:
            self.S = CachedMarketData(self.init_coinbase())
        else:
            self.S = MarketData()

        self.verbose = verbose
        if self.verbose:
            self.progress_bar = tqdm.tqdm(total=len(self.S), unit="samples")

        self.action_space = spaces.Discrete(len([0, 1, 2]))

        self._seed()
        self.history = queue.deque(maxlen=history_length)
        # TODO don't think this should be here
        first_observation = self._preprocess_state()
        for _ in range(history_length):
            self.history.append(first_observation.copy())
        self.observation = self._reset()
        obs_dim = self.observation.shape[0]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim),
            high=np.array([np.inf] * obs_dim),
        )

    @disk_cache
    def init_coinbase(self):
        S = []
        with open("data/coinbaseUSD.csv") as fin:
            for row in fin:
                price, something = map(float, row[:-1].split(",")[1:])
                S.append((price, something))
        return np.array(S)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        self.observation = self._preprocess_state()
        self.taken_actions[action] += 1

        penalty = 0
        new_price = self.S[self.sim_time][Observations.SELL]
        transaction_fee = self.bitcoin_fraction * (0.16 / 100)

        can_buy_btc = self.liquid_budget > (new_price * self.bitcoin_fraction + transaction_fee)
        if action == MarketActions.SELL:
            if self.invested_budget > self.bitcoin_fraction and self.liquid_budget > transaction_fee:
                self.invested_budget -= self.bitcoin_fraction
                self.liquid_budget += new_price * self.bitcoin_fraction
                self.liquid_budget -= transaction_fee
                self.ledger.append(new_price)  # doesn't make sense
            else:
                penalty += 1

        elif action == MarketActions.BUY:
            if can_buy_btc:
                self.invested_budget += self.bitcoin_fraction
                self.liquid_budget -= new_price * self.bitcoin_fraction
                self.liquid_budget -= transaction_fee
                self.ledger.append(new_price)
            else:
                penalty += 1

        elif action == MarketActions.KEEP:
            pass

        # (new_price - self.initial_price)/self.initial_price
        self.sim_time += 1
        done = self.sim_time == self.sim_ep_timestep_bound or self.sim_time == len(self.S) or (not can_buy_btc)
        # normalized_initial_investment = (self.initial_budget + 0) * (new_price / self.initial_price)
        # agent_value = (self.liquid_budget - self.initial_budget) / self.initial_budget
        # current_budget = (self.invested_budget * new_price + self.liquid_budget)
        # portfolio_value = (current_budget - self.initial_budget) / self.initial_budget
        #
        # bonus = 0
        if not done:
            agent_value = (self.liquid_budget - self.initial_budget) / self.initial_budget
        else:
            current_value = self.liquid_budget - self.invested_budget * new_price
            agent_value = ((current_value) - self.initial_budget) / self.initial_budget
            # I want to be able to manage risk profile using bonuses or..goals :)
            # if portfolio_value > self.max_gain:
            #     bonus = 10
            # elif portfolio_value < self.max_loss:
            #     bonus = -10

            # current_value = self.liquid_budget - self.invested_budget * new_price
            # agent_value = (current_value - self.initial_budget) / self.initial_budget
            # current_value = self.invested_budget * new_price
            # portfolio_value = (self.liquid_budget + current_value - self.initial_budget) / self.initial_budget

        reward = agent_value - penalty  # + bonus
        if self.verbose:
            self.progress_bar.update(1)

        assert self.sim_time <= self.sim_ep_timestep_bound or self.sim_time < len(self.S) or self.invested_budget > 0
        # assert self.sim_time < len(self.S)
        # assert self.invested_budget >= 0
        # print(self.liquid_budget)
        if self.liquid_budget < 0:
            print(self.liquid_budget, "Liquid budget < 0 ")
            raise AssertionError()

        return self.observation, reward, done, {"time_step": self.sim_time}

    def _close(self):
        print("LOL BYE")

    def _reset(self):
        # self.sim_time -= int(self.sim_time_max / 10)  # TODO try re-experience a 1th of the last run
        self.sim_ep_timestep_bound = self.sim_time + self.sim_ep_max_length

        self.initial_price = self.S[self.sim_time][Observations.SELL]
        # sell everything from last run and start a new
        self.initial_budget = self.liquid_budget + self.invested_budget * self.initial_price
        return self._preprocess_state()

    def _preprocess_state(self):
        # PERIOD = 10
        buy, sell, volume = tuple(self.S[self.sim_time])

        state = [buy, sell, volume, self.invested_budget, self.liquid_budget]
        self.history.append(np.array(state))
        return np.array(self.history).flatten()

    def print_stats(self, epsilon=None, extra=""):
        buy, sell, volume = tuple(self.S[self.sim_time])

        # normalized_initial_investment = (self.initial_budget + 0) * (sell / self.initial_price)
        agent_value = (self.invested_budget * sell + self.liquid_budget)
        delta_cash = agent_value - self.initial_budget
        market_return = (sell - self.initial_price) / self.initial_price
        ep_return = delta_cash / self.initial_budget
        step = datetime.datetime.utcnow().strftime("%d-%m-%H-%M-%S")
        ep_stats = OrderedDict(
            epsilon=epsilon,
            act_dist=self.taken_actions,
            n_btc=self.invested_budget,
            liquid_budget=self.liquid_budget,
            agent_value=agent_value,
            delta_cash=delta_cash,
            ep_return=ep_return,
            market_return=market_return,
            return_over_market=ep_return / market_return,
            trade_volume=self.taken_actions[MarketActions.BUY] - self.taken_actions[MarketActions.SELL],
            adv_over_keep_policy=delta_cash + self.initial_budget - (1000. / self.S[0][0]) * sell,
        )
        progress = round(self.sim_time / float(len(self.S)), 6)

        print("===== Progress {} =====".format(progress))
        for k, v in ep_stats.items():
            # it's easy to ask forgiveness than permission
            try:
                self.ep_summary.value.add(tag=v, simple_value=k)
                print("{}: {:.3f}".format(k, v))
            except TypeError:
                print("{}: {}".format(k, v))
        with open(self._logger_path, 'a') as fout:
            writer = csv.DictWriter(fout, fieldnames=list(ep_stats.keys()) + ['step'])
            if self._write_head == True:
                writer.writeheader()
                self._write_head = False
            ep_stats['step'] = step
            writer.writerow(ep_stats)

        self.taken_actions = {
            MarketActions.SELL: 0,
            MarketActions.BUY: 0,
            MarketActions.KEEP: 0,
        }

        # import tensorflow as tf
        # ep_summary = tf.summary.Summary()
        # writer = tf.summary.FileWriter(logdir='logs')
        # Add max gain and max loss
        # Add prize for goal directed behavior...something like - beta * (max_gain -ep_return)
        # Market return per episode, final_price - opening price / opening price.
        # Average market return 1/n return per episode
        # Episode standard deviation, ep mean. Running variance, Running average price
        # Invested budget, liquid budget, cash_out_value, cumulative balance
        # Advantage_over_conservative_actor
        # Value_loss_per_episode (maybe adjusted like value_loss_per_episode /
        # Varies of value of the latest discounted state - (final_price -init_price) i.e true incremeent
        # Action ditsribtuion plot
        # Action distribution entropy
        # Return_per_episode / number of seen datapoint
        # P entropy
        # Information gain: entropy_t-1 - entropy_t
        # Dkl of p
        # tensorboard:
        # - conv weight
        # - value loss
        # - distribution over actions
        # - distributions over state
        # create logger file
        # create trader dashboard
        # print(
        #     # "progress:", round(self.sim_time / float(len(self.S)), 6),
        #     "epsilon", round(epsilon, 2),
        #     "btc", round(self.invested_budget, 2),
        #     "$$", round(self.liquid_budget, 2),
        #     "price", round(sell, 2),
        #     "ep_prft", round(delta_cash, 2),
        #     # "gain_over_market", round(agent_value - normalized_initial_investment, 2),
        #     "sordi", round(delta_cash + self.initial_budget, 2),
        #     "hold",
        #     self.taken_actions,
        #     extra,
        #     sep="\t",
        #     file=sys.stderr,
        # )


def test_BitEnv():
    env = BitEnv(None)
    print(env.reset())
    done = False
    step = 0
    while not done:
        step += 1
        act = random.randint(0, 2)
        s, r, done, info = env.step(act)
        print("step", step, s[:5], r)
        print("step", step, s[5:], r)


if __name__ == "__main__":
    test_BitEnv()
