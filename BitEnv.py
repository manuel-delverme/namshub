import gym
from disk_utils import disk_cache
import random
from gym import spaces
from gym.utils import seeding
import numpy as np

import queue
import sys
import tqdm
import datetime
import time

VERY_SMALL_NUMBER = 0.01


class MarketActions(object):
    SELL = 0
    BUY = 1
    NOOP = 2


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
        price, volume = self.S[time_idx]
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
    def __init__(self, use_historic_data=False):
        self.taken_actions = {
            MarketActions.SELL: 0,
            MarketActions.BUY: 0,
            MarketActions.NOOP: 0,
        }
        self.ledger = []
        self.range = 1000  # +/- value the randomly select number can be between
        self.bounds = 5000  # Action space bounds

        self.bitcoin_fraction = 1 / 10
        self.number = 0
        self.sim_time = 0
        self.sim_episode_length = 200
        self.sim_time_max = self.sim_episode_length

        self.initial_budget = 1000
        self.liquid_budget = self.initial_budget
        self.invested_budget = 0

        print("init")
        if use_historic_data:
            self.S = CachedMarketData(self.init_coinbase())
        else:
            self.S = MarketData()

        self.progress_bar = tqdm.tqdm(total=len(self.S), unit="samples")

        self.action_space = spaces.Discrete(len([0, 1, 2]))
        # self.action_space = spaces.Box(

        self._seed()
        self.history = queue.deque(maxlen=100)
        first_observation = self._preprocess_state()
        for _ in range(100):
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

        new_price = self.S[self.sim_time][Observations.SELL]
        transaction_fee = self.bitcoin_fraction * (0.16 / 100)

        penalty = 0
        self.taken_actions[action] += 1
        can_buy_btc = self.liquid_budget > (new_price * self.bitcoin_fraction + transaction_fee)
        if action == MarketActions.SELL:
            if self.invested_budget > self.bitcoin_fraction and self.liquid_budget > transaction_fee:
                self.invested_budget -= self.bitcoin_fraction
                self.liquid_budget += new_price * self.bitcoin_fraction
                self.liquid_budget -= transaction_fee
                self.ledger.append(new_price)
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

        elif action == MarketActions.NOOP:
            pass

        # normalized_initial_investment = (self.initial_budget + 0) * (new_price / self.initial_price)
        # agent_value = (self.invested_budget * new_price + self.liquid_budget)


        # simone style

        # whitehead style
        agent_value = (self.liquid_budget - self.initial_budget) / self.initial_budget
        reward = agent_value - penalty

        self.sim_time += 1
        self.progress_bar.update(1)

        done = self.sim_time == self.sim_time_max or self.sim_time == len(self.S) or (not can_buy_btc)

        assert self.sim_time <= self.sim_time_max
        assert self.sim_time < len(self.S)
        assert self.invested_budget >= 0
        # print(self.liquid_budget)
        if self.liquid_budget < 0:
            print(self.liquid_budget, "< 0")
            raise AssertionError()

        return self.observation, reward, done, {"time_step": self.sim_time}

    def _close(self):
        print("LOL BYE")

    def _reset(self):
        # self.sim_time -= int(self.sim_time_max / 10)  # TODO try re-experience a 1th of the last run
        self.sim_time_max = self.sim_time + self.sim_episode_length

        self.initial_price = self.S[self.sim_time][Observations.SELL]
        # sell everything from last run and start anew
        self.initial_budget = self.liquid_budget + self.invested_budget * self.initial_price
        return self._preprocess_state()

    def _preprocess_state(self):
        PERIOD = 10

        buy, sell, volume = tuple(self.S[self.sim_time])
        state = [buy, sell, volume, self.invested_budget, self.liquid_budget]
        """

        period = min(PERIOD, self.sim_time + 1 + 1)
        start = max(self.sim_time - period, 0)
        price_history = self.S[start:self.sim_time + 1][:, 0]

        indicators = [
            technical_indicators.exponential_moving_average.exponential_moving_average,
            technical_indicators.simple_moving_average.simple_moving_average,
            technical_indicators.bollinger_bands.bandwidth,
        ]
        # TODO: handle values before -2, either avoid calculation or add to state
        for indicator in indicators:
            if period > len(price_history):
                value = [0]
            else:
                value = indicator(price_history, period=period)
            state.append(value[-1])
        """

        # technical_indicators.moving_average_convergence_divergence
        # technical_indicators.exponential_moving_average
        # technical_indicators.simple_moving_average
        # technical_indicators.smoothed_moving_average
        # technical_indicators.stochrsi
        # technical_indicators.relative_strength_index
        # technical_indicators.on_balance_volume
        # technical_indicators.directional_indicators
        # technical_indicators.ichimoku_cloud
        # technical_indicators.momentum
        # technical_indicators.money_flow
        # technical_indicators.money_flow_index
        self.history.append(np.array(state))
        return np.array(self.history).flatten()

    def print_stats(self, epsilon=None, extra=""):
        buy, sell, volume = tuple(self.S[self.sim_time])

        normalized_initial_investment = (self.initial_budget + 0) * (sell / self.initial_price)
        agent_value = (self.invested_budget * sell + self.liquid_budget)

        delta_cash = self.invested_budget * sell + self.liquid_budget - self.initial_budget
        print(
            # "progress:", round(self.sim_time / float(len(self.S)), 6),
            # "epsilon", round(epsilon, 2),
            "btc", round(self.invested_budget, 2),
            "$$", round(self.liquid_budget, 2),
            "price", round(sell, 2),
            "ep_prft", round(delta_cash, 2),
            # "gain_over_market", round(agent_value - normalized_initial_investment, 2),
            "sordi", round(delta_cash + self.initial_budget, 2),
            "hold", round((1000. / self.S[0][0]) * sell, 2) - round(delta_cash + self.initial_budget, 2),
            self.taken_actions,
            extra,
            sep=" ",
            file=sys.stderr,
        )
        self.taken_actions = {
            MarketActions.SELL: 0,
            MarketActions.BUY: 0,
            MarketActions.NOOP: 0,
        }


def test_BitEnv():
    env = BitEnv()
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
