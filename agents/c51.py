import numpy as np

from commons.memory import ReplayBuffer, PrioritizedReplayBuffer
from commons.tf_utils import transfer_learning, get_trainable_variables, make_session, init_graph
from commons.utils import LinearSchedule
from models.dqn import DQN


class Agent(object):
    def __init__(self, obs_dim, acts_dim, agent_config, train_config):
        self.acts_dim = acts_dim
        self.eps = 1.

        # TODO this should be cleaned up
        self.target = DQN(name='target', obs_dim=obs_dim, acts_dim=acts_dim, network_config=agent_config['network'],
                          gamma=agent_config['gamma'],
                          v_min=agent_config['v_min'],
                          v_max=agent_config['v_max'],
                          nb_atoms=agent_config['nb_atoms'])
        self.local = DQN(name='agent', obs_dim=obs_dim, acts_dim=acts_dim, network_config=agent_config['network'],
                         gamma=agent_config['gamma'],
                         v_min=agent_config['v_min'],
                         v_max=agent_config['v_max'],
                         nb_atoms=agent_config['nb_atoms'])

        # TODO fix me
        if train_config['use_replay'] == True:
            self.memory = PrioritizedReplayBuffer(size=train_config['buffer_size'],alpha = train_config['alpha_replay'])
            self.beta_scheduler = LinearSchedule(init_value=train_config['beta_replay'],final_value=1.0, max_steps=train_config['max_steps'])
        else:
            self.memory = ReplayBuffer(size=train_config['buffer_size'])
        self.alpha_scheduler = LinearSchedule(init_value=1., final_value=train_config['final_eps'],
                                              max_steps=(train_config['max_steps'] * train_config['expl_fraction']))
        self.__sync_op = transfer_learning(to_tensors=get_trainable_variables(scope='target'),
                                           from_tensors=get_trainable_variables('agent'))
        self.sess = make_session(
            num_cpu=train_config['num_cpu'])  # tf.Session(config=make_config(num_cpu=train_config['num_cpu']))
        self.reset_graph()
        self.update_target()

    def get_train_summary(self, feed_dict):

        return self.sess.run([self.local._summary_op, self.local._global_step], feed_dict = feed_dict)

    def reset_graph(self):
        init_graph(sess=self.sess)

    def update_target(self):
        self.sess.run(self.__sync_op)

    def step(self, obs, schedule):
        self.eps = self.alpha_scheduler.value(t=schedule)
        if self.eps < np.random.random():
            act = self.sess.run(self.local.next_action, feed_dict={self.local.obs: obs})[0]
        else:
            act = np.random.randint(low=0, high=self.acts_dim)
        return act

    def get_p(self, obs):
        return self.sess.run(self.local.p, feed_dict={self.local.obs: obs})[0]

    def sample(self, batch_size, t):
        return self.memory.sample(batch_size, self.beta_scheduler.value(t))

    def train(self, obs, acts, rws, obs1, dones, idxs_and_ws):
        # TODO this should be done inside the TF graph....
        thtz = self.sess.run([self.target.ThTz],
                             feed_dict={self.target.obs: obs1, self.target.rws: rws, self.target.dones: dones})[0]
        loss, _ = self.sess.run([self.local.cross_entropy, self.local.train_op],
                                feed_dict={self.local.obs: obs, self.local.acts: acts, self.local.thtz: thtz})

        if self.memory._kind != 'simple':
            # make sure this is the average td error
            ws = thtz.mean(axis = 1)
            idxs, _ = idxs_and_ws
            self.memory.update_memory(idxs, ws)

        feed_dict = {
            self.local.obs:obs,
            self.local.acts:acts,
            self.local.thtz:thtz,
            self.local.rws:rws
        }
        # TODO compute td error

        return loss, feed_dict

    def close(self):
        self.sess.close()
