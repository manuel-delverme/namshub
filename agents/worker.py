from agents.a3c import A3C
from env import BitEnv


class Worker(object):
    def __init__(self, name, target, network_config, training_config):
        self.env = BitEnv.BitEnv(use_historic_data=True, verbose=  (name=='worker_0'))  # gym.make(training_config['env_name'])  # .unwrapped
        self.env = BitEnv.BitEnv(use_historic_data=True, verbose=(name == 'worker_0'))  # gym.make(training_config['env_name'])  # .unwrapped
        self.name = name
        self.agent = A3C(name, self.env.observation_space.shape[0], self.env.action_space.n, target, network_config)
        self.ep_stats = {'ep_rw': 0, 'ep_len': 0, 'total_ep': 0}
        self.config = training_config
        self.gamma = training_config['gamma']
        self._lambda = training_config['_lambda']
        self.ob = self.env.reset()
        self.ep_r = 0
        self.local_t = 0

    def run_once(self):
        batch = []
        for t in range(self.config['update_freq']):
            act, v = self.agent.step(self.ob)
            ob1, r, done, info = self.env.step(act)
            self.ep_r += r
            self.local_t += 1
            batch.append((self.ob, act, r, v))
            self.ob = ob1
            if done:
                self.ob = self.env.reset()
                self.ep_stats['total_ep'] += 1
                self.ep_stats['ep_r'] = self.ep_r
                self.ep_r = 0
                break

        feed_dict = self.agent.get_batch(batch=batch, ob1=ob1, done=done, gamma=self.gamma, _lambda=self._lambda)
        return feed_dict, done

    def run(self, sess, coord):
        with sess.as_default():
            while not coord.should_stop() and self.ep_stats['total_ep'] < self.config['max_ep']:
                self.agent.sync_from_target()
                feed_dict, done = self.run_once()
                losses, global_step = self.agent.train(feed_dict=feed_dict)
                if self.name == 'worker_0' and done and self.ep_stats['total_ep'] % 10 == 0:
                    self.env.print_stats(
                        extra=
                        " ".join((self.name, "Global_step:", str(global_step),
                                  '| Total_ep: {}'.format(self.ep_stats['total_ep']),
                                  "| Ep_r: {}".format(self.ep_stats['ep_r']),
                                  '| Ep_a_loss: {}'.format(losses[0]),
                                  '| Ep_c_Loss: {}'.format(losses[1])
                                  )))
