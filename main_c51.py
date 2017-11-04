import gym
import tensorflow as tf

from agents.c51 import Agent
from commons.tf_utils import load_model, save, get_trainable_variables
from commons.utils import PlotMachine

tf.logging.set_verbosity(tf.logging.INFO)
network_config = {
    'units': (32, 16),
    'topology': 'fc',  # conv
    'act': tf.nn.relu,
    'lr': 1e-3,
    'max_clip': 40.

}

agent_config = {
    'gamma': 0.95,
    'v_min': 0.,
    'v_max': 10.,
    'nb_atoms': 11.,
    '_lambda': 1.0,
    'network': network_config
}
train_config = {
    'env_name': 'CartPole-v0',
    'max_ep': int(1e6),
    'max_steps': 1e6 * 200,
    'summary_freq': 500,
    'save_freq': 50000,
    'update_target_freq': 1000,
    'num_cpu': 1,
    'batch_size': 64,
    'expl_fraction': .0002,
    'final_eps': .10,
}

"""
- add convolution
- add average statistics
- add prioritized replay memory
- add a way to manage network config, training config 
- add tensorboard
"""
MAX_STEPS = 100000
TRAIN_FREQ = 5
UPDATE_TARGET_FREQ = 500
BATCH_SIZE = 64
PRINT_FREQ = 500
BUFFER_SIZE = 50000


# def main():
#     env = gym.make('CartPole-v0')  # BitEnv.BitEnv(use_historic_data=True)
#     agent = Agent(obs_dim=env.observation_space.shape[0], acts_dim=env.action_space.n, agent_config=agent_config,
#                   train_config=train_config)
#     plotter = PlotMachine(agent=agent, v_min=agent_config['v_min'], v_max=agent_config['v_max'],
#                           nb_atoms=agent_config['nb_atoms'],
#                           n_actions=env.action_space.n,
#                           )
#     total_steps = 0
#     for ep in range(train_config['max_ep']):
#         done = False
#         ob = env.reset()
#         ep_r, ep_steps= 0, 0,
#         while not done:
#
#             if ep_r > 150:
#                 env.render()
#                 plotter.plot_dist(obs=[ob])
#
#             act = agent.step(obs=[ob], schedule=total_steps)
#             ob1, r, done, _ = env.step(action=act)
#             agent.memory.add(step=(ob, act, r, ob1, float(done)))
#             ob = ob1.copy()
#             ep_r += r
#             ep_steps += 1
#             total_steps += 1
#
#         batch = agent.memory.sample(batch_size=train_config['batch_size'])
#         loss = agent.train(*batch)
#
#         if total_steps % train_config['update_target_freq'] == 0:
#             agent.update_target()
#
#         if total_steps % train_config['summary_freq'] == 0:
#             tf.logging.info('Ep : {}, rw: {}, len {}'.format(ep, ep_r
#                                                              , ep_steps))
#             # print(loss)
#             # env.print_stats(epsilon=agent.eps, extra=loss)
#
#         if total_steps % train_config['save_freq'] == 0:
#             save(sess=agent.sess, save_path=train_config['save_dir'], var_list=agent.target._params)

def main():
    env = gym.make('CartPole-v0')
    agent = Agent(obs_dim=env.observation_space.shape[0], acts_dim=env.action_space.n, train_config=train_config,
                  agent_config=agent_config)  # max_steps=MAX_STEPS,
    # buffer_size=BUFFER_SIZE)
    plotter = PlotMachine(agent=agent, v_min=0, v_max=10, nb_atoms=11, n_actions=env.action_space.n)
    ob = env.reset()
    ep_rw = 0
    # from tf_utils import load_model

    load_model(sess=agent.sess, load_path='logs')
    for t in range(MAX_STEPS):
        if ep_rw > 150:
            env.render()
            plotter.plot_dist(obs=[ob])
        act = agent.step(obs=[ob], schedule=t)
        ob1, r, done, _ = env.step(action=act)
        agent.memory.add(step=(ob, act, r, ob1, float(done)))
        ob = ob1.copy()
        ep_rw += r
        if done:
            ob = env.reset()
            if t % TRAIN_FREQ == 0:
                batch = agent.memory.sample(batch_size=BATCH_SIZE)
                loss = agent.train(*batch)
                if t % PRINT_FREQ:
                    print('EP {}, loss {}, eps {}, rw {}'.format(t, loss, agent.eps, ep_rw))
                    # from tf_utils import save, get_trainable_variables
                    save(sess=agent.sess, save_path='logs', var_list=get_trainable_variables(scope='target'))
            if t % UPDATE_TARGET_FREQ == 0:
                agent.update_target()
            ep_rw = 0


if __name__ == '__main__':
    main()
