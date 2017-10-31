import tensorflow as tf
from tqdm import tqdm
import os
from network import ActorCritic
from async_agent import A3CGroupAgent
import datetime
import six
import sys
from collections import deque
from oldtf_utils import fc
import BitEnv
import random
import numpy as np


def main():
    DISCOUNT_FACTOR = 0.99
    # DEVICE = '/gpu:0'
    DEVICE = '/cpu:0'

    SAVE_PERIOD = 20000
    SUMMARY_PERIOD = 100

    LEARNING_RATE = 0.00025
    DECAY = 0.99
    GRAD_CLIP = 0.1
    ENTROPY_BETA = 0.01

    NUM_THREADS = 1
    AGENT_PER_THREADS = 1
    UNROLL_STEP = 5
    MAX_ITERATION = 1000000
    RANDOM_SEED = 185

    env = BitEnv.BitEnv()
    # Initialize Seed
    tf.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.reset_default_graph()

    action_space_size = env.action_space.n
    # define actor critic networks and environments
    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = tf.train.polynomial_decay(LEARNING_RATE, global_step, MAX_ITERATION // 2, LEARNING_RATE * 0.1)

    master_ac = ActorCritic(nA=action_space_size, state_shape=env.observation_space.shape,  device_name=DEVICE,
                            learning_rate=learning_rate, decay=DECAY, grad_clip=GRAD_CLIP, entropy_beta=ENTROPY_BETA)
    agents_group = []
    for i in range(NUM_THREADS):
        group_envs = [BitEnv.BitEnv() for _ in range(AGENT_PER_THREADS)]
        ac = ActorCritic(nA=action_space_size, state_shape=env.observation_space.shape, master=master_ac,
                         device_name=DEVICE, scope_name='Thread%02d' % i, learning_rate=learning_rate, decay=DECAY,
                         grad_clip=GRAD_CLIP, entropy_beta=ENTROPY_BETA)
        agentGroup = A3CGroupAgent(group_envs, ac, unroll_step=UNROLL_STEP, discount_factor=DISCOUNT_FACTOR, seed=i)
        agents_group.append(agentGroup)

    queue = tf.FIFOQueue(capacity=NUM_THREADS * 10, dtypes=[tf.float32, tf.float32, tf.float32], )

    train_ops = [g_agent.enqueue_op(queue) for g_agent in agents_group]
    qr = tf.train.QueueRunner(queue, train_ops)
    tf.train.queue_runner.add_queue_runner(qr)
    loss = queue.dequeue()

    # Miscellaneous(init op, summaries, etc.)
    increase_step = global_step.assign(global_step + 1)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    def _train_info():
        _total_eps = sum([g_agent.num_episodes() for g_agent in agents_group])
        _avg_r = sum([g_agent.reward_info()[0] for g_agent in agents_group]) / len(agents_group)
        _max_r = max([g_agent.reward_info()[1] for g_agent in agents_group])
        return _total_eps, _avg_r, _max_r

    train_info = tf.py_func(_train_info, [], [tf.int64, tf.float64, tf.float64], stateful=True)
    pl, el, vl = loss
    total_eps, avg_r, max_r = train_info

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('policy_loss', pl)
    tf.summary.scalar('entropy_loss', el)
    tf.summary.scalar('value_loss', vl)
    tf.summary.scalar('total_episodes', total_eps)
    tf.summary.scalar('average_rewards', avg_r)
    tf.summary.scalar('maximum_rewards', max_r)
    summary_op = tf.summary.merge_all()
    # config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

    # saver and sessions
    saver = tf.train.Saver(var_list=master_ac.train_vars, max_to_keep=3)

    sess = tf.Session()
    sess.graph.finalize()

    sess.run(init_op)
    master_ac.initialize(sess)
    for agent in agents_group:
        agent.master_actor_critic.initialize(sess)
    print('Initialize Complete...')

    LOG_DIR = "log/"
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    summary_writer_eps = tf.summary.FileWriter(os.path.join(LOG_DIR, 'per-eps'))
    # summary_writer.add_summary(sess.run(config_summary))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in tqdm(range(MAX_ITERATION)):
            if coord.should_stop():
                break

            (pl, el, vl), summary_str, (total_eps, avg_r, max_r), _ = sess.run(
                [loss, summary_op, train_info, increase_step])
            if step % SUMMARY_PERIOD == 0:
                summary_writer.add_summary(summary_str, step)
                summary_writer_eps.add_summary(summary_str, total_eps)
                tqdm.write('step(%7d) policy_loss:%1.5f,entropy_loss:%1.5f,value_loss:%1.5f, te:%5d avg_r:%2.1f '
                           'max_r:%2.1f' % (step, pl, el, vl, total_eps, avg_r, max_r))
            if (step + 1) % SAVE_PERIOD == 0:
                saver.save(sess, os.path.join(LOG_DIR, '/model.ckpt'), global_step=step + 1)
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

        saver.save(sess, LOG_DIR + '/last.ckpt')
        sess.close()
        # queue.close() #where should it go?
    # -----------------END----------------

    """
    stats = {
        'reward': deque(maxlen=env.sim_time_max),
        'acts': [0, 0, 0],
    }
    act = BitEnv.MarketActions.BUY
    stats['acts'][act] += 1
    stats['reward'].append(r)
    ep_stats = {
        'steps': steps,
        'avg_reward': np.array(stats['reward']).mean(),
        'sell': stats['acts'][0],
        'buy': stats['acts'][1],
        'noop': stats['acts'][2],
        'liquid': env.liquid_budget,
        'invested': env.invested_budget * env.observation[BitEnv.Observations.PRICE],
        'epsilon': agent.epsilon,
    }
    stats['reward'].clear()
    stats['acts'] = [0, 0, 0]
    """


if __name__ == "__main__":
    main()
