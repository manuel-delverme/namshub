import numpy as np

from commons.memory import ReplayBuffer
from commons.tf_utils import transfer_learning, get_trainable_variables, make_session, init_graph
from commons.utils import LinearSchedule
from models.dqn import DQN


class Agent(object):
    def __init__(self, obs_dim, acts_dim, agent_config, train_config):
        self.acts_dim = acts_dim
        self.eps = 1.
        self.target = DQN(name='target', obs_dim=obs_dim, acts_dim=acts_dim, network_config=agent_config['network'])
        self.local = DQN(name='agent', obs_dim=obs_dim, acts_dim=acts_dim, network_config=agent_config['network'])
        self.memory = ReplayBuffer(size=train_config['batch_size'])
        self.scheduler = LinearSchedule(init_value=1., final_value=train_config['final_eps'],
                                        max_steps=(train_config['max_steps'] * train_config['expl_fraction']))
        self.__sync_op = transfer_learning(to_tensors=get_trainable_variables(scope='target'),
                                           from_tensors=get_trainable_variables('agent'))
        self.sess = make_session(
            num_cpu=train_config['num_cpu'])  # tf.Session(config=make_config(num_cpu=train_config['num_cpu']))
        self.reset_graph()
        self.update_target()

    def reset_graph(self):
        init_graph(sess=self.sess)

    def update_target(self):
        self.sess.run(self.__sync_op)

    def step(self, obs, schedule):
        self.eps = self.scheduler.value(t=schedule)
        if self.eps < np.random.random():
            act = self.sess.run(self.local.next_action, feed_dict={self.local.obs: obs})[0]
        else:
            act = np.random.randint(low=0, high=self.acts_dim)
        return act

    def get_p(self, obs):
        return self.sess.run(self.local.p, feed_dict={self.local.obs: obs})[0]

    def train(self, obs, acts, rws, obs1, dones):
        # =============
        # TODO this should be done inside the TF graph....
        # =============
        thtz = self.sess.run([self.target.ThTz],
                             feed_dict={self.target.obs: obs1, self.target.rws: rws, self.target.dones: dones})[0]
        loss, _ = self.sess.run([self.local.cross_entropy, self.local.train_op],
                                feed_dict={self.local.obs: obs, self.local.acts: acts, self.local.thtz: thtz})
        return loss

    def close(self):
        self.sess.close()
