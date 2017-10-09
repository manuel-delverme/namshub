import os

import numpy as np
import tensorflow as tf


def _clip_grad(loss, vars, clip=50):
    grads = tf.gradients(loss, vars)
    # assert all(grads)  # make sure that there are no none in gradients
    g, _ = tf.clip_by_global_norm(grads, clip)
    return zip(g, vars)


def fc(x, h_size, name, act=None, std=0.1):
    with tf.variable_scope(name):
        input_size = x.get_shape()[1]
        w = tf.get_variable('w', (input_size, h_size), initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable('b', (h_size), initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        if act is not None:
            z = act(z)
        return z


class GaussianPD(object):
    def __init__(self, mu, logstd):
        self.mu = mu
        self.std = tf.exp(logstd)
        self.logstd = logstd
        self.act_dim = tf.to_float(tf.shape(mu)[-1])

    def neglogp(self, action):
        return 0.5 * tf.reduce_sum(tf.square((action - self.mu) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(action)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def kl(self, pi_other):
        return tf.reduce_sum(
            pi_other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mu - pi_other.mu)) / (
                2.0 * tf.square(pi_other.std)) - 0.5, axis=-1)

    def logp(self, action):
        return -self.neglogp(action)

    def sample(self):
        return self.mu + tf.random_normal(tf.shape(self.mu)) * self.std


def get_params(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def _save(saver, sess, log_dir):
    try:
        saver.save(sess=sess, save_path=os.path.join(log_dir, 'model.ckpt'))
    except Exception as e:
        tf.logging.error(e)
        raise e


def _load(saver, sess, log_dir):
    try:
        ckpt = tf.train.latest_checkpoint(log_dir)
        saver.restore(sess=sess, save_path=ckpt)
    except Exception as e:
        tf.logging.error(e)
        raise e

def set_global_seed(seed = 10):

    tf.set_random_seed(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)



