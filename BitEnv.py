import gym
from utils import disk_cache
import random
from gym import spaces
from gym.utils import seeding
import numpy as np

import technical_indicators.bollinger_bands
import technical_indicators.exponential_moving_average
import technical_indicators.simple_moving_average
import technical_indicators.smoothed_moving_average
import technical_indicators.stochrsi
import technical_indicators.relative_strength_index
import technical_indicators.on_balance_volume
import technical_indicators.directional_indicators
import technical_indicators.ichimoku_cloud
import technical_indicators.momentum
import technical_indicators.money_flow
import technical_indicators.money_flow_index

import technical_indicators.moving_average_convergence_divergence
# import volume_oscillator

# DEPRECATED

# import keltner_bands
# import technical_indicators.accumulation_distribution
# import technical_indicators.aroon
# import technical_indicators.average_true_range
# import technical_indicators.average_true_range_percent

# import double_exponential_moving_average
# import hull_moving_average
# import volume_adjusted_moving_average
# import weighted_moving_average
# import linear_weighted_moving_average
# import moving_average_envelope
# import triangular_moving_average
# import triple_exponential_moving_average

# import price_channels
# import price_oscillator
# import rate_of_change
# import standard_deviation
# import standard_variance
# import stochastic
# import true_range
# import typical_price
# import ultimate_oscillator
# import vertical_horizontal_filter
# import volatility
# import volume_index
# import volume_oscillator
# import williams_percent_r

# import chaikin_money_flow
# import chande_momentum_oscillator
# import commodity_channel_index
# import detrended_price_oscillator
# import double_smoothed_stochastic


VERY_SMALL_NUMBER = 0.01


class MarketActions(object):
    SELL = 0
    BUY = 1
    NOOP = 2


class Observations(object):
    PRICE = 0
    INVESTED = 1
    LIQUID = 2


class BitEnv(gym.Env):
    def __init__(self):
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
        # self.init_brownian_motion()
        self.S = self.init_coinbase()

        self.action_space = spaces.Discrete(len([0, 1, 2]))
        # self.action_space = spaces.Box(

        self._seed()
        self.observation = self._reset()
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * len(self.observation)),
            high=np.array([np.inf] * len(self.observation)),
        )

    @disk_cache
    def init_coinbase(self):
        S = []
        with open("data/coinbaseUSD.csv") as fin:
            for row in fin:
                price, something = map(float, row[:-1].split(",")[1:])
                S.append((price, something))
        return np.array(S)

    def init_brownian_motion(self):
        T = 2
        mu = 0.1
        sigma = 0.01
        S0 = 20
        dt = 0.01
        N = round(T / dt)
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size=N)
        W = np.cumsum(W) * np.sqrt(dt)  ### standard brownian motion ###
        # X = (mu-0.5*sigma**2)*t + sigma*W
        # self.S = S0*np.exp(X) ### geometric brownian motion ###
        self.S = S0 * np.exp(W)  ### geometric brownian motion ###

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        self.observation = self.preprocess_state()

        new_price = self.observation[Observations.PRICE]
        transaction_fee = self.bitcoin_fraction * (0.16 / 100)

        penalty = 0
        if action == MarketActions.SELL:
            if self.invested_budget > VERY_SMALL_NUMBER and self.liquid_budget > transaction_fee:
                self.invested_budget -= self.bitcoin_fraction
                self.liquid_budget += new_price * self.bitcoin_fraction
                self.liquid_budget -= transaction_fee
            else:
                penalty += 1

        elif action == MarketActions.BUY:
            if self.liquid_budget > (new_price * self.bitcoin_fraction + transaction_fee):
                self.invested_budget += self.bitcoin_fraction
                self.liquid_budget -= new_price * self.bitcoin_fraction
                self.liquid_budget -= transaction_fee
            else:
                penalty += 1

        elif action == MarketActions.NOOP:
            pass

        normalized_initial_investment = (self.initial_budget + 0) * (new_price / self.initial_price)
        agent_value = (self.invested_budget * new_price + self.liquid_budget)
        reward = agent_value - normalized_initial_investment - penalty

        self.sim_time += 1
        done = self.sim_time == self.sim_time_max or self.sim_time == len(self.S)

        assert self.sim_time <= self.sim_time_max
        assert self.sim_time < len(self.S)
        assert self.invested_budget > -VERY_SMALL_NUMBER
        # print(self.liquid_budget)
        if self.liquid_budget < -VERY_SMALL_NUMBER:
            print(self.liquid_budget, "< -VERY_SMALL_NUMBER")
            raise AssertionError()

        return self.observation, reward, done, {"time_step": self.sim_time}

    def _reset(self):
        # self.sim_time -= int(self.sim_time_max / 10)  # TODO try re-experience a 1th of the last run
        self.sim_time_max = self.sim_time + self.sim_episode_length

        self.initial_price = self.S[self.sim_time][0]
        # sell everything from last run and start anew
        self.initial_budget = self.liquid_budget + self.invested_budget * self.initial_price
        return self.preprocess_state()

    def preprocess_state(self):
        PERIOD = 10

        new_price, some_number = tuple(self.S[self.sim_time])
        state = [new_price, some_number, self.invested_budget, self.liquid_budget]

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
        return np.array(state)


def test_BitEnv():
    env = BitEnv()
    print(env.reset())
    done = False
    step = 0
    while not done:
        step += 1
        act = random.randint(0, 2)
        s, r, done, info = env.step(act)
        print("step", step, s, r)


if __name__ == "__main__":
    test_BitEnv()
