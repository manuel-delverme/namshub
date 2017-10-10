import tensorflow as tf
import sys
from collections import deque
from tf_utils import fc
from BitEnv import BitEnv
import random
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, name):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.sess = tf.Session()
        with tf.variable_scope(name_or_scope=name):
            self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        self._observation = tf.placeholder(tf.float32, [None, self.state_size])

        self._action_target = tf.placeholder(tf.int32, [None], name='action_target')
        self._q_target = tf.placeholder(tf.float32, [None], name='q_value_target')

        with tf.variable_scope('deepq_model'):
            _hidden = fc(self._observation, h_size=24, name='fc_input', act=tf.nn.relu)
            for idx in range(2):
                _hidden = fc(_hidden, h_size=24, name='fc' + str(idx), act=tf.nn.relu)
            self._q_hat = fc(_hidden, h_size=self.action_size, name='fc', act=None)

        # turn (0..2) into 1-hot encoding
        _action_one_hot = tf.one_hot(self._action_target, self.action_size, 1.0, 0.0, name='action_target_one_hot')

        # values collected following action_target
        _q_acted = tf.reduce_sum(self._q_hat * _action_one_hot, reduction_indices=1, name='q_hat')
        _delta = self._q_target - _q_acted

        self._loss = tf.reduce_mean(tf.square(_delta))
        self._train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

    def get_q(self, state):

        if np.ndim(state) < 2:
            state = [state]

        return self.sess.run(self._q_hat, feed_dict={self._observation: state})

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.get_q(state)
        return np.argmax(act_values[0])  # returns action

    def train(self, state, q_target, act_target):
        loss, _ = self.sess.run([self._loss, self._train_op],
                                feed_dict={self._observation: state, self._q_target: q_target,
                                           self._action_target: act_target})
        return loss

    def replay(self, batch_size):
        loss = 0
        if len(self.memory) < batch_size:
            return 0
        else:
            # TODO refactor this
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.amax(self.get_q(next_state))
                q_target, = self.get_q(state)
                q_target[action] = target
                loss += self.train(state=[state], q_target=q_target, act_target=[action])
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return loss / batch_size


MAX_EP = 500


def main():
    env = BitEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, "agent")
    agent_target = DQNAgent(env.observation_space.shape[0], env.action_space.n, "agent_target")
    stats = {
        'reward': deque(maxlen=100),
    }

    for e in range(MAX_EP):
        observation = env.reset()
        done = False
        total_loss = 0
        steps = 0
        while not done:
            steps += 1
            q = agent_target.get_q(state=observation)
            if agent.epsilon >= np.random.random():
                act = np.random.randint(low=0, high=env.action_space.n)
            else:
                # TODO: use q_target
                act = np.argmax(q)
            s1, r, done, _ = env.step(act)
            agent.remember(state=observation, action=act, reward=r, done=done, next_state=s1)
            observation = s1.copy()
            total_loss += agent.replay(batch_size=64)
            stats['reward'].append(r)
            print('episode {}, step {}, state {}, avg_reward {}'.format(
                e,
                steps,
                s1,
                sum(stats['reward']) / len(stats['reward'])
            ))
        local_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="agent")
        target_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_agent")

        ops = [target_var.assign(local_var) for local_var, target_var in zip(local_list, target_list)]
        agent.sess.run(ops)


def random_run():
    env = BitEnv()
    stats = {
        'reward': deque(maxlen=100),
    }

    for e in range(MAX_EP):
        observation = env.reset()
        done = False
        total_loss = 0
        steps = 0
        while not done:
            steps += 1
            act = np.random.randint(low=0, high=env.action_space.n)
            s1, r, done, _ = env.step(act)
            observation = s1.copy()
            stats['reward'].append(r)
            print('episode {}, step {}, state {}, avg_reward {}'.format(
                e,
                steps,
                s1,
                sum(stats['reward']) / len(stats['reward'])
            ))

if __name__ == "__main__":
    random_agent = len(sys.argv) > 1 and sys.argv[1] == "random"
    if random_agent:
        random_run()
    else:
        main()
