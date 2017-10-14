import tensorflow as tf
import numpy as np
import scipy.misc
import os
import random
from functools import partial

from history_buffer import *


def _preprocess_o(o, some_param=None):
    return o


class A3CGroupAgent():
    def __init__(self, envs, master_actor_critic, unroll_step, discount_factor, seed=None, image_size=(84, 84),
                 frames_for_state=4):
        self.envs = envs
        self.nA = envs[0].action_space.n
        self.observe_shape = list(image_size) + [1]  # [84,84,1]
        self.frames_for_state = frames_for_state
        self.discount_factor = discount_factor
        self.preprocess = partial(_preprocess_o, some_param=None)

        self.master_actor_critic = master_actor_critic
        self.policy_func = master_actor_critic.get_policy
        self.value_func = master_actor_critic.get_value
        self.unroll_step = unroll_step

        # For enqueu_op only. Please do not use in other methods.
        self.random = np.random.RandomState(seed)  # TODO: set tf seed
        self.hist_bufs = [HistoryBuffer(self.preprocess, self.observe_shape, frames_for_state) for _ in envs]
        self.states = [None for _ in envs]
        # Hmm... I don't like this pattern. Anyway, list itself is thread-safe, so it is correct implementation.
        # TODO: verify
        self.episode_rewards = [[]] * len(envs)
        self.episode_reward = [0.] * len(envs)

    def pick_actions(self, s, greedy=False, epsilon=0.01):
        assert not greedy

        pi_given_s = self.policy_func(s)
        actions = [self.random.choice(self.nA, 1, p=action_probs)[0] for action_probs in pi_given_s]
        return actions

    def enqueue_op(self, queue):
        def _func():
            # Initialize states, if the game is done at the last iteration
            for i, _ in enumerate(self.envs):
                if self.states[i] is None:
                    self.episode_rewards[i].append(self.episode_reward[i])
                    self.episode_reward[i] = 0.

                    self.hist_bufs[i].clear()
                    o = self.envs[i].reset()
                    self.states[i] = self.hist_bufs[i].add(o)

            # Proceed the multiple games.
            done_envs = set()
            sras = [[] for _ in self.envs]  # state reward action pairs for each envs.
            for step in range(self.unroll_step):
                actions = self.pick_actions(np.stack(self.states, axis=0))

                for i, (env, action) in enumerate(zip(self.envs, actions)):
                    if i in done_envs:
                        continue

                    o, reward, done, _ = env.step(action)
                    self.episode_reward[i] += reward

                    reward = max(-1, min(1, reward))  # reward clipping -1 to 1
                    sras[i].append((self.states[i], reward, action))
                    self.states[i] = self.hist_bufs[i].add(o)

                    if done:
                        done_envs.add(i)

            # Calculate the estimated values for each envs.
            vs = self.value_func(np.stack(self.states, axis=0))
            states = []
            actions = []
            values = []
            for i, sra in enumerate(sras):
                if i in done_envs:
                    vs[i] = 0.
                    self.states[i] = None

                for s, r, a in sra[::-1]:  # TODO: add advantage extimation schulman GAE
                    vs[i] = r + self.discount_factor * vs[i]
                    states.append(s)
                    actions.append(a)
                    values.append(vs[i])

            states = np.stack(states, axis=0)
            actions = np.stack(actions, axis=0).astype(np.int32)
            target_values = np.stack(values, axis=0).astype(np.float32)

            policy_loss, entropy_loss, value_loss = self.master_actor_critic.update(states, actions, target_values)
            self.master_actor_critic.sync()

            return policy_loss, entropy_loss, value_loss

        data = tf.py_func(_func, [], [tf.float32, tf.float32, tf.float32], stateful=True, name="enqueue_op")
        return queue.enqueue(data)

    def num_episodes(self):
        return sum([len(l) - 1 for l in self.episode_rewards if len(l) > 1])

    def reward_info(self):
        recent_rewards = [l[-1] for l in self.episode_rewards if len(l) > 1]
        avg_r = sum(recent_rewards) / len(recent_rewards) if len(recent_rewards) > 0 else float("nan")
        max_r = max(recent_rewards) if len(recent_rewards) > 0 else float("nan")
        return avg_r, max_r

    def test_run(self, env, greedy, render=False):
        episode_reward = 0.0
        hist_buf = HistoryBuffer(self.preprocess, self.observe_shape, self.frames_for_state)

        o = env.reset()
        state = hist_buf.add(o)
        while True:
            if render:
                env.render()
            action = self.pick_actions(np.expand_dims(state, axis=0), greedy)[0]
            o, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break

            state = hist_buf.add(o)
        return episode_reward
