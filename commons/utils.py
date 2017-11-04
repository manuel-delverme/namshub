import numpy as np
import matplotlib.pyplot as plt
from commons.tf_utils import build_z

class LinearSchedule(object):
    def __init__(self, init_value, final_value, max_steps):
        self.max_steps = max_steps
        self.init_value = init_value
        self.final_value = final_value

    def value(self, t):
        fraction = min(float(t) / self.max_steps, 1.0)
        return self.init_value + fraction * (self.final_value - self.init_value)



class PlotMachine(object):
    def __init__(self, agent, v_min, v_max, nb_atoms, n_actions, action_set=None):
        z, self.dz = build_z(v_min, v_max, nb_atoms)
        self.z = agent.sess.run(z)
        # turn interactive mode on
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.bars = [self.ax.bar(self.z, np.ones_like(self.z) * .25, self.dz * .9) for _ in range(n_actions)]
        if action_set is not None:
            plt.legend(action_set, loc='upper left')
        self.agent = agent
        pass

    def make_pdf(self, obs):
        return self.agent.get_p(obs=obs)

    def plot_dist(self, obs):
        pdf_act = self.make_pdf(obs)

        for rects, sample in zip(self.bars, pdf_act):
            for rect, y in zip(rects, sample):
                rect.set_height(y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
