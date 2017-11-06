import threading

import numpy as np
import tensorflow as tf

from agents.worker import Worker
from commons.tf_utils import get_env_dims, make_config
from models.actor_critic import ActorCritic

network_config = {
    'units': [128, ],
    'act': tf.nn.elu,
    'lr': 1e-3,
    'optim': tf.train.RMSPropOptimizer,
    'clip': 40.0,
    'beta': 1e-1

}
training_config = {
    'env_name': 'CartPole-v0',
    'n_threads': 1,  # multiprocessing.cpu_count()
    'max_ep': np.inf,
    'update_freq': 10,
    'gamma': 0.9,
    '_lambda': 1.0,
}


def main():
    obs_dim, acts_dim = get_env_dims(training_config['env_name'])
    print("obs", obs_dim, "acts", acts_dim)
    sess =  tf.Session(config=make_config(num_cpu=training_config['n_threads']))
    with tf.device("/cpu:0"):
        target = ActorCritic(scope='target', obs_dim=obs_dim, acts_dim=acts_dim,
                             network_config=network_config)  # we only need its params
        workers = []
        # Create worker
        for idx in range(training_config['n_threads']):
            workers.append(
                Worker('worker_{}'.format(idx), target, network_config=network_config, training_config=training_config))
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=worker.run, args=(sess, coord,))
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)


if __name__ == "__main__":
    main()
