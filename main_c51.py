import numpy as np
import tensorflow as tf

import bit_env.BitEnv
from agents.c51 import Agent
# from commons.running_stats import ZFilter
from commons.tf_utils import create_writer, create_saver
from commons.utils import PrintMachine

tf.logging.set_verbosity(tf.logging.INFO)
network_config = {
    'units': (32, 16),
    'topology': 'conv',  # 'fc',  # conv
    'act': tf.nn.relu,
    'lr': 1e-3,
    'max_clip': 40.

}

agent_config = {
    'gamma': 0.90,
    'v_min': 0.,
    'v_max': 25.,
    'nb_atoms': 11,
    '_lambda': 1.0,
    'network': network_config
}
train_config = {
    # 'env_name': 'CartPole-v0',
    'max_ep': np.inf,  # int(1e5),
    'max_steps': 1e5 * 10,
    'summary_freq': 100,
    'save_freq': 1000,
    'update_target_freq': 500,
    'train_freq': 1,
    'num_cpu': 1,
    'batch_size': 64,
    'buffer_size': int(1e5),
    'expl_fraction': .5,
    'final_eps': .01,
}

env_config = {
    'max_loss': -.2,
    'max_gain': .5,
    'bitcoin_fraction': .1,
    'transaction_fee': .1 * .0016,
    'max_timesteps': 200,
    'initial_budget': 1000,
    'history_length': 100,
    'use_historic_data': True,
    'verbose': False,
}


# TODO: add tensorbaord and add replay memory


def warmup(env, ob_filter, steps=1000):
    from numpy.random import randint
    ob = env.reset()
    for _ in range(steps):
        ob = ob_filter(ob)
        ob, _, done, _ = env.step(randint(0, env.action_space.n))
        if done:
            ob = env.reset()


def main():
    env = bit_env.BitEnv.BitEnv(**env_config)
    agent = Agent(obs_dim=env.observation_space.shape[0], acts_dim=env.action_space.n, agent_config=agent_config,
                  train_config=train_config)
    # plotter = PlotMachine(agent=agent, v_min=agent_config['v_min'], v_max=agent_config['v_max'],
    #                       nb_atoms=agent_config['nb_atoms'],
    #                       n_actions=env.action_space.n, action_set=None)

    ob_filter = lambda x: x  # ZFilter(shape=env.observation_space.shape)
    warmup(env=env, ob_filter=ob_filter)

    # load_model(sess = agent.sess, load_path='logs')
    ep = 0
    total_steps = 0
    printer = PrintMachine(train_config['summary_freq'])
    writer = create_writer(logdir='logs/stats/')
    saver = create_saver(var_list=agent.target._params)
    printer.restart()
    while ep < train_config['max_ep']:
        ep += 1
        done = False
        ob = env.reset()
        ep_rw = 0
        while not done:
            ob = ob_filter(ob)

            act = agent.step(obs=[ob], schedule=total_steps)
            ob1, r, done, _ = env.step(action=act)
            # TODO : shoud the filter be applied here?
            agent.memory.add(step=(ob, act, r, ob_filter(ob1), float(done)))
            ob = ob1.copy()
            ep_rw += r
            total_steps += 1
        if total_steps % train_config['train_freq'] == 0:
            batch = agent.memory.sample(batch_size=train_config['batch_size'])
            loss, feed_dict = agent.train(*batch)
        if printer.is_up_reset():
            ep_stats = env.print_stats(epsilon=agent.eps,
                                       extra='EP {}, total steps {}, loss {}, rw {}'.format(ep, total_steps,
                                                                                            loss, ep_rw))
            writer.add_summary(ep_stats, global_step=global_step)
            writer.flush()
            # plotter.plot_dist(obs=[ob])
        if total_steps % train_config['summary_freq'] == 0:
            train_stats, global_step = agent.get_train_summary(feed_dict)
            writer.add_summary(train_stats, global_step)
            writer.flush()
        if total_steps % train_config['update_target_freq'] == 0:
            agent.update_target()

        if total_steps % train_config['save_freq'] == 0:
            saver.save(sess=agent.sess, save_path='logs/ckpt/model.ckpt', write_meta_graph=False)


if __name__ == '__main__':
    main()
