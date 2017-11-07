import gym
import numpy as np
import tensorflow as tf
import bit_env.BitEnv

from agents.c51 import Agent
from commons.running_stats import ZFilter
from commons.tf_utils import save, get_trainable_variables
from commons.utils import PlotMachine, PrintMachine

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
    'summary_freq': 30,
    'save_freq': 50000,
    'update_target_freq': 500,
    'train_freq': 1,
    'num_cpu': 1,
    'batch_size': 64,
    'buffer_size': int(1e5),
    'expl_fraction': .5,
    'final_eps': .01,
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
    env = bit_env.BitEnv.BitEnv(use_historic_data=True)
    agent = Agent(obs_dim=env.observation_space.shape[0], acts_dim=env.action_space.n, agent_config=agent_config,
                  train_config=train_config)
    # plotter = PlotMachine(agent=agent, v_min=agent_config['v_min'], v_max=agent_config['v_max'],
    #                       nb_atoms=agent_config['nb_atoms'],
    #                       n_actions=env.action_space.n, action_set=None)
    ep_rw = 0
    total_steps = 0

    ob_filter = lambda x: x  # ZFilter(shape=env.observation_space.shape)
    warmup(env=env, ob_filter=ob_filter)

    # load_model(sess = agent.sess, load_path='logs')

    ep = 0
    printer = PrintMachine(train_config['summary_freq'])
    printer.restart()
    loss = "lolwut"
    while ep < train_config['max_ep']:
        ep += 1
        done = False
        ob = env.reset()
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
            loss = agent.train(*batch)

        if printer.is_up_reset():
            env.print_stats(epsilon=agent.eps, extra='EP {}, total steps {}, loss {}, rw {}'.format(ep, total_steps,
                                                                                                    loss, ep_rw))
            # plotter.plot_dist(obs=[ob])
            # if total_steps % train_config['summary_freq'] == 0:
        if total_steps % train_config['update_target_freq'] == 0:
            agent.update_target()

        if total_steps % train_config['save_freq'] == 0:
            save(sess=agent.sess, save_path='logs', var_list=get_trainable_variables(scope='target'))
        ep_rw = 0


if __name__ == '__main__':
    main()
