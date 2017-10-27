import tensorflow as tf
import os
import numpy as np


def build_z(v_min, v_max, n_atoms):
    dz = (v_max - v_min) / (n_atoms - 1)
    z = tf.range(v_min, v_max + dz / 2, dz, dtype=tf.float32, name='z')
    return z, dz


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
    return update_op #tf.group(*update_op)


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


def fc(x, h_size, name, act=None, std=0.1):
    with tf.variable_scope(name):
        input_size = x.get_shape()[1]
        w = tf.get_variable('w', (input_size, h_size), initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable('b', (h_size), initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        if act is not None:
            z = act(z)
        return z


