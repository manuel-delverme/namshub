import os

import numpy as np
import tensorflow as tf
from scipy.signal import lfilter
from tensorflow.contrib.layers import flatten


def build_z(v_min, v_max, n_atoms):
    dz = (v_max - v_min) / (n_atoms - 1)
    z = tf.range(v_min, v_max + dz / 2, dz, dtype=tf.float32, name='z')
    return z, dz


def categorical_sample(logits):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return value  # tf.one_hot(value, d)


def p_to_q(p_dist, v_min, v_max, n_atoms):
    z, _ = build_z(v_min, v_max, n_atoms)
    return tf.tensordot(p_dist, z, axes=[[-1], [-1]])


def set_global_seed(seed=1234):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    pass


def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)


def transfer_learning(to_tensors, from_tensors, tau=1.):
    update_op = []
    for from_t, to_t in zip(sorted(from_tensors, key=lambda v: v.name),
                            sorted(to_tensors, key=lambda v: v.name)):
        update_op.append(
            # C <- C * tau + C_old * (1-tau)
            tf.assign(to_t, tf.multiply(from_t, tau) + tf.multiply(to_t, 1. - tau))
        )

    return update_op


def get_pa(p, acts, batch_size):
    cat_idx = tf.transpose(
        tf.reshape(tf.concat([tf.range(batch_size), acts], axis=0), shape=[2, batch_size]))
    p_target = tf.gather_nd(params=p, indices=cat_idx)
    return p_target


def load_model(sess, load_path, var_list=None):
    ckpt = tf.train.load_checkpoint(ckpt_dir_or_file=load_path)
    saver = tf.train.Saver(var_list=var_list)
    try:
        saver.restore(sess=sess, save_path=ckpt)
    except Exception as e:
        tf.logging.error(e)


def save(sess, save_path, var_list=None):
    os.makedirs(save_path, exist_ok=True)
    saver = tf.train.Saver(var_list=var_list)
    try:
        saver.save(sess=sess, save_path=os.path.join(save_path, 'model.ckpt'),
                   write_meta_graph=False)
    except Exception as e:
        tf.logging.error(e)


def fc(x, h_size, name, reuse=False, act=None, std=0.1):
    with tf.variable_scope(name, reuse=reuse):
        input_size = x.get_shape()[1]
        w = tf.get_variable('w', (input_size, h_size), initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable('b', (h_size), initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        if act is not None:
            z = act(z)
        return z


def make_config(num_cpu, memory_fraction=.25):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        log_device_placement=False
    )
    tf_config.gpu_options.allow_growth = True
    return tf_config


def make_session(num_cpu=1):
    return tf.Session(config=make_config(num_cpu=num_cpu))


def init_graph(sess):
    sess.run(tf.global_variables_initializer())
    tf.logging.info('Graph initialized')


def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def train_op(grads, vars, optim=tf.train.RMSPropOptimizer, global_step=None, use_lock=True, max_clip=40., lr=1e-4):
    grads, _ = tf.clip_by_global_norm(t_list=grads, clip_norm=max_clip)
    return tf.group(optim(learning_rate=lr, use_locking=use_lock).apply_gradients(zip(grads, vars),
                                                                                  global_step=global_step))


def get_env_dims(env_name):
    from bit_env import BitEnv
    env = BitEnv.BitEnv()
    obs_dim = env.observation_space.shape[0]
    acts_dim = env.action_space.n
    env.close()
    return obs_dim, acts_dim


def get_default_session():
    return tf.get_default_session()


def compute_gae(rws, r_hat, vs, gamma=0.95, _lambda=1.0):
    d_rws = np.append(rws, r_hat)
    d_rws = discount(d_rws, gamma)[:-1]
    vs = np.append(vs, r_hat)
    td_error = rws + gamma * vs[1:] - vs[:-1]
    adv = discount(td_error, gamma=gamma * _lambda)
    return d_rws, adv

def conv1d(obs, seq_len = 100, obs_dim = 5):
    h = tf.reshape(obs, (-1, seq_len, obs_dim))
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
    return h