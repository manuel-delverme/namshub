import numpy as np
import tensorflow as tf

from agents.c51 import Agent
from bit_env.BitEnv import BitEnv
from commons.logger import Logger

# from commons.running_stats import ZFilter

tf.logging.set_verbosity(tf.logging.INFO)
network_config = {
    'units': (64, 32, 64),
    'topology': 'conv',  # 'fc',  # conv
    'act': tf.nn.relu,
    'lr': 1e-3,
    'max_clip': 40.

}

agent_config = {
    'gamma': 0.99,
    'v_min': -20.,
    'v_max': +20.,
    'nb_atoms': 51,
    '_lambda': 1.0,
    'network': network_config
}
train_config = {
    # 'env_name': 'CartPole-v0',
    'max_ep': np.inf,  # int(1e5),
    'max_steps': 1e6 * 10,
    'summary_freq': 100,
    'save_freq': 5000,
    'update_target_freq': 5,
    'train_freq': 1,
    'num_cpu': 1,
    'batch_size': 64,
    'buffer_size': int(1e5),
    'expl_fraction': .5,
    'final_eps': .01,
    'alpha_replay': .6,  # define the importance of the priority of the memory
    'beta_replay': .4,
    'use_replay': False
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


def main():
    env = BitEnv(**env_config)
    # import gym
    #
    # env = gym.make('CartPole-v0')
    agent = Agent(obs_dim=env.observation_space.shape[0], acts_dim=env.action_space.n, agent_config=agent_config,
                  train_config=train_config)
    # plotter = PlotMachine(agent=agent, v_min=agent_config['v_min'], v_max=agent_config['v_max'],
    #                       nb_atoms=agent_config['nb_atoms'],
    #                       n_actions=env.action_space.n, action_set=None)

    logger = Logger(log_dir='logs', var_list=agent.target._params, verbose=False, max_steps=None)
    ep = 0
    total_steps = 0
    # printer = PrintMachine(train_config['summary_freq'])
    # printer.restart()
    try:

        while ep < train_config['max_ep']:
            ep += 1
            done = False
            ep_rw = 0
            ob = env.reset()
            while not done:
                # env.render()

                act = agent.step(obs=[ob], schedule=total_steps)
                ob1, r, done, _ = env.step(action=act)
                agent.memory.add(step=(ob, act, r, ob1, float(done)))
                ob = ob1.copy()
                ep_rw += r
                total_steps += 1

            batch, idxs_ws = agent.sample(batch_size=train_config['batch_size'], t=total_steps)
            loss, feed_dict = agent.train(*batch, idxs_and_ws=idxs_ws)
            # plotter.plot_dist(obs=[ob])

            if ep % train_config['update_target_freq'] == 0:
                agent.update_target()
            if ep % train_config['summary_freq'] == 0:
                summary, global_step = agent.get_train_summary(feed_dict=feed_dict)
                ep_stats = env.print_stats(epsilon=agent.eps)
                # ep_stats = {
                #     'loss': loss,
                #     'agent_eps': agent.eps,
                #     'ep_rw': ep_rw,
                #     'total_ep': ep,
                #     'total_steps': total_steps
                # }
                logger.log(ep_stats, total_ep=ep)
                logger.dump(stats=ep_stats, tf_summary=summary, global_step=total_steps)
            if ep % train_config['save_freq'] == 0:
                logger.save_model(sess=agent.sess, global_step=total_steps)
    except KeyboardInterrupt:
        logger.save_model(sess=agent.sess, global_step=total_steps)
        agent.close()
        print('Closing experiment. File saved at {}'.format(logger.save_path))


if __name__ == '__main__':
    main()
