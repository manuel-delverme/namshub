import gym
from utils import disk_cache
import random
from gym import spaces
from gym.utils import seeding
import numpy as np

from enum import Enum


class MarketActions(Enum):
    SELL = 0
    BUY = 1
    NOOP = 2


class Observations(Enum):
    PRICE = 0
    INVESTED = 1
    LIQUID = 2


class BitEnv(gym.Env):
    def __init__(self):
        self.range = 1000  # +/- value the randomly select number can be between
        self.bounds = 5000  # Action space bounds

        self.number = 0
        self.sim_time = 0
        self.sim_time_max = 200

        self.liquid_budget = 1000
        self.invested_budget = 0

        print("init")
        # self.init_brownian_motion()
        self.S = self.init_coinbase()
        # print(self.S)
        self.sim_time_max = len(self.S)

        self.action_space = spaces.Discrete(len([0, 1, 2]))
        # self.action_space = spaces.Box(

        lower_bound = []
        upper_bound = []
        for idx in range(self.S.shape[1]):
            lower_bound.append(self.S[:, idx].min())
            upper_bound.append(self.S[:, idx].max())
        lower_bound.extend([0, 0])
        upper_bound.extend([self.bounds, self.bounds])
        self.observation_space = spaces.Box(
            low=np.array(lower_bound),
            high=np.array(upper_bound)
        )

        self._seed()
        self.observation = self._reset()

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
        # import ipdb; ipdb.set_trace()
        assert self.action_space.contains(action)
        # self.np_random.uniform(-self.range, self.rangea)
        new_price, some_number = tuple(self.S[self.sim_time])

        market_return = new_price / self.observation[Observations.PRICE.value]

        if action == MarketActions.SELL.value and self.invested_budget > 0:
            self.invested_budget -= 1
            self.liquid_budget += 1
        elif action == MarketActions.BUY.value and self.liquid_budget > 0:
            self.invested_budget += 1
            self.liquid_budget -= 1
        elif action == MarketActions.NOOP.value:
            pass

        reward = self.invested_budget * market_return

        self.observation = np.array((new_price, some_number, self.invested_budget, self.liquid_budget))

        self.sim_time += 1
        done = self.sim_time >= self.sim_time_max
        assert self.liquid_budget + self.invested_budget == 1000

        return self.observation, reward, done, {"time_step": self.sim_time}

    def _reset(self):
        self.sim_time = 0
        self.observation = np.array((self.S[self.sim_time][0], self.S[self.sim_time][1], self.invested_budget, self.liquid_budget))
        return self.observation


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
