import numpy as np
from tensorflow import get_default_session

from models.actor_critic import ActorCritic


class A3C(object):
    def __init__(self, scope, obs_dim, acts_dim, target, network_config):
        print("obs", obs_dim, "acts", acts_dim)
        self.model = ActorCritic(scope=scope, obs_dim=obs_dim, acts_dim=acts_dim, target=target,
                                 network_config=network_config)

    def train(self, feed_dict):  # run by a local
        sess = get_default_session()  # local grads applies to global net
        pi_loss, vf_loss, global_step, _ = sess.run(
            [self.model.pi_loss, self.model.vf_loss, self.model.global_step, self.model.train_op], feed_dict)
        return (pi_loss, vf_loss), global_step

    def sync_from_target(self):  # run by a local
        sess = get_default_session()
        sess.run([
            self.model.sync_op
        ])

    def get_v(self, ob):
        sess = get_default_session()
        return sess.run(self.model.v, {self.model.obs: ob})

    def step(self, ob):  # run by a local
        sess = get_default_session()
        pi, v = sess.run([self.model.pi, self.model.v], feed_dict={self.model.obs: [ob]})  # ob[np.newaxis, :]})
        action = np.random.choice(pi.shape[1], p=np.clip(pi.ravel(), a_min=0, a_max=1-1e-4))  # select action w.r.t the actions prob
        return action, v[0]

    def get_batch(self, batch, ob1, done, gamma=.95, _lambda=1.):

        batch = np.array(batch)
        obs = batch[:, 0]
        acts = batch[:, 1]
        rws = batch[:, 2]
        vs = batch[:, 3]

        r_hat = 0
        if not done:
            r_hat = self.get_v(ob=[ob1])[0]

        # d_rws, adv = compute_gae(rws =rws, vs= vs, r_hat=r_hat, gamma=gamma, _lambda=_lambda)
        d_rws = []
        for r in rws[::-1]:  # reverse buffer r
            r_hat = r + gamma * r_hat
            d_rws.append(r_hat)

        d_rws = np.array(d_rws)[::-1]  # .copy()
        adv = d_rws - vs

        feed_dict = {
            self.model.obs: np.vstack(obs),
            self.model.acts: acts,
            self.model.rws: d_rws,
            self.model.adv: adv  # d_rws - np.array(vs)
        }
        return feed_dict
