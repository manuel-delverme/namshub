import gym
import numpy as np
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
    PRICE = 1
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
        T = 2
        mu = 0.1
        sigma = 0.01
        S0 = 20
        dt = 0.01
        N = round(T/dt)
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size = N)
        W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
        # X = (mu-0.5*sigma**2)*t + sigma*W
        # self.S = S0*np.exp(X) ### geometric brownian motion ###
        self.S = S0*np.exp(W) ### geometric brownian motion ###
        print(self.S)
        self.sim_time_max = len(self.S)

        self.action_space = spaces.MultiDiscrete([[0, 2]])
        # self.action_space = spaces.Box(

        self.observation_space = spaces.Box(
            low=np.array([0]),
            high=np.array([self.bounds])
        )

        self._seed()
        self.observation = self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        import ipdb; ipdb.set_trace()
        assert self.action_space.contains(action)
        # self.np_random.uniform(-self.range, self.rangea)
        new_price = self.S[self.tim_time]

        market_return = new_price / self.observation[Observations.PRICE]
        money_pre_step = self.budget

        if action == MarketActions.SELL and self.invested_budget > 0:
            self.invested_budget -= 1
            self.liquid_budget += 1
        elif action == MarketActions.BUY and self.liquid_budget > 0:
            self.invested_budget += 1
            self.liquid_budget -= 1
        elif action == MarketActions.NOOP:
            pass

        reward = self.invested_budget * market_return

        self.observation = np.array((new_price, self.invested_budget, self.liquid_budget))

        self.sim_time += 1
        done = self.sim_time >= self.sim_time_max
        assert self.liquid_budget + self.invested_budget == 1000

        return self.observation, reward, done, {"time_step": self.sim_time}

    def _reset(self):
        self.observation = np.array((self.S[0], self.invested_budget, self.liquid_budget))
        self.sim_time = 0
        return self.observation


def test_BitEnv():
    env = BitEnv()
    print(env.reset())
    done = False
    while not done:
        act = random.randint(0, 2)
        s, r, d, info = env.step(act)

if __name__ == "__main__":
    test_BitEnv()
