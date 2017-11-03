import tensorflow as tf
from tensorflow.contrib.layers import flatten
import commons.ops
from tensorflow.contrib.framework import get_or_create_global_step

from tf_utils import fc, get_trainable_variables, transfer_learning


class ActorCritic(object):
    def __init__(self, scope, obs_dim, acts_dim, network_config, target=None):
        self.scope = scope
        self.global_step = get_or_create_global_step()
        self.__init_ph(obs_dim=obs_dim)
        self.__build_graph(acts_dim=acts_dim, act=network_config['act'], units=network_config['units'])
        self.__loss_op(acts_dim=acts_dim, beta=network_config['beta'])
        self.__grads_op()
        if target is not None:
            self.__train_op(target, optim=network_config['optim'], clip=network_config['clip'],
                            lr=network_config['lr'])
            # else:
            #     self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def __init_ph(self, obs_dim):
        self.obs = tf.placeholder(tf.float32, [None, obs_dim], name='obs')
        self.acts = tf.placeholder(tf.int32, [None], name='acts')
        self.rws = tf.placeholder(tf.float32, [None], name='rws')
        self.adv = tf.placeholder(tf.float32, [None], name='adv')

    def __build_graph(self, acts_dim, units=64, act=tf.nn.relu):
        with tf.variable_scope(self.scope):
            h = tf.reshape(self.obs, (-1, 100, 5))
            # shared block
            input_channels = h.shape.dims[2].value
            for filter_width, output_channels in zip(*([5, 3, 1], [3, 2, 1])):
                filter_shape = [filter_width, input_channels, output_channels]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[output_channels]), name="b")
                conv = tf.nn.conv1d(
                    h, W, stride=1,
                    padding="SAME",
                    name="conv",
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                input_channels = output_channels
            h = flatten(h)

            # for idx, size in enumerate(units):
            #     h = fc(x=h, h_size=size, act=act, name='h_{}'.format(idx))

            with tf.variable_scope('actor'):
                self._pi = fc(h, h_size=acts_dim, act=None, name='actor')
                self.pi = tf.nn.softmax(self._pi)
                self.log_pi = tf.nn.log_softmax(self._pi)
                # self.sample = tf.multinomial(logits=self._pi, num_samples=1)
                # self.sample = categorical_sample(logits=self._pi)
            with tf.variable_scope('critic'):
                v = fc(h, h_size=1, act=None, name='critic')
                self.v = tf.reshape(v, [-1])

        self.params = get_trainable_variables(scope=self.scope)

    def __grads_op(self):
        self.grads = tf.gradients(self.loss, self.params)

    def __loss_op(self, acts_dim, beta):
        self.vf_loss = tf.reduce_sum(tf.square(self.rws - self.v), name='vf_loss')
        self.entropy = -tf.reduce_sum(self.pi * self.log_pi, axis=1)  # encourage exploration
        log_pi = tf.reduce_sum(self.log_pi * tf.one_hot(self.acts, acts_dim, dtype=tf.float32), axis=1)
        self.pi_loss = - tf.reduce_sum(log_pi * self.adv, name='pi_loss')

        self.loss = self.pi_loss + .5 * self.vf_loss - beta * self.entropy

    def __train_op(self, target, optim, clip, lr):
        self.sync_op = transfer_learning(to_tensors=self.params, from_tensors=target.params)
        self.grads, _ = tf.clip_by_global_norm(t_list=self.grads, clip_norm=clip)
        self.train_op = optim(lr).apply_gradients(
            zip(self.grads, target.params), global_step=self.global_step)
