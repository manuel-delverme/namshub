import tensorflow as tf

from commons.tf_utils import fc, p_to_q, build_z, get_trainable_variables, get_pa


class DQN(object):
    def __init__(self, name, obs_dim, acts_dim, network_config, gamma=.95, v_min=-20., v_max=20., n_atoms=51):
        self.scope = name
        self.__init_ph(obs_dim=obs_dim, acts_dim=acts_dim, n_atoms=n_atoms)
        self.__build_graph(acts_dim=acts_dim, units=network_config['units'], act=network_config['act'], n_atoms=n_atoms,
                           topology=network_config['topology'])
        self.__predict_op(v_min=v_min, v_max=v_max, n_atoms=n_atoms)
        self.__train_op(lr=network_config['lr'], max_clip=network_config['max_clip'])
        if self.scope == 'target':
            self.__build_categorical(v_min, v_max, n_atoms, gamma)

    def __train_op(self, lr=1e-3, max_clip=40.):
        # cat_idx = tf.transpose(
        #     tf.reshape(tf.concat([tf.range(batch_size), self.acts], axis=0), shape=[2, batch_size]))
        # p_target = tf.gather_nd(params=self.p, indices=cat_idx)
        p_target = get_pa(p=self.p, acts=self.acts, batch_size=tf.shape(self.acts)[0])
        self.cross_entropy = tf.reduce_mean(tf.reduce_sum(-self.thtz * tf.log(0.0001 + p_target), axis=-1),
                                            name='empirical_cross_entropy')
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        self._params = get_trainable_variables(scope=self.scope)
        grads = tf.gradients(self.cross_entropy, self._params)
        grads, _ = tf.clip_by_global_norm(t_list=grads, clip_norm=max_clip)
        self.train_op = opt.apply_gradients(zip(grads, self._params))

    def __init_ph(self, obs_dim, acts_dim, n_atoms):
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim], name='obs')
        self.acts = tf.placeholder(dtype=tf.int32, shape=[None], name='acts')
        self.rws = tf.placeholder(dtype=tf.float32, shape=[None], name='rws')
        self.dones = tf.placeholder(dtype=tf.float32, shape=[None], name='dones')
        self.thtz = tf.placeholder(dtype=tf.float32, shape=[None, n_atoms], name='T_pi')

    def __predict_op(self, v_min, v_max, n_atoms):
        self.q_values = p_to_q(self.p, v_min, v_max, n_atoms)
        self.next_action = tf.cast(tf.argmax(self.q_values, axis=1), tf.int32)  # check shape (None,)

    def __build_graph(self, acts_dim, units, act, n_atoms, topology):
        with tf.variable_scope(self.scope):
            h = self.obs
            if topology == 'conv':
                from commons.tf_utils import conv1d
                conv1d(obs=h, scope=topology)
            else:
                for idx, size in enumerate(units):
                    h = fc(x=h, h_size=size, act=act, name='fc_{}'.format(idx))

            with tf.variable_scope('p_dist'):
                logits = fc(x=h, h_size=acts_dim * n_atoms, act=None, name='logits')
                logits = tf.reshape(logits, shape=(-1, acts_dim, n_atoms))
                self.p = tf.nn.softmax(logits, dim=-1)

    def __build_categorical(self, v_min, v_max, n_atoms, gamma):
        z, dz = build_z(v_min=v_min, v_max=v_max, n_atoms=n_atoms)
        batch_size = tf.shape(self.rws)[0]
        with tf.variable_scope('categorical'):
            # we need to build a categorical index for the bin. Thus we get the index, concat with the action which is of size
            # (batch_size,), reshape it of (2,batch_size) and transpose it. The final matrix is of size (batch_size,2) where the first columns has index and
            # seocond column has action
            # get probs of selected actions. p is of shape (batch_size, action_size, dict_size)
            # p_best (batch_size, nb_atoms)
            self.p_best = get_pa(p=self.p, acts=self.next_action, batch_size=batch_size)
            # replicates z, batch_size times, this is building the integrations support over the atom dimension
            Z = tf.reshape(tf.tile(z, [batch_size]), shape=[batch_size, n_atoms])
            # replicates rws (batch_size, ) over n_atoms, reshape it in n_atoms, batch_size, and traspose it. Final dim (batch_size, n_atoms)
            R = tf.transpose(tf.reshape(tf.tile(self.rws, [n_atoms]), shape=[n_atoms, batch_size]))
            # Apply bellman operator over the Z random variable
            Tz = tf.clip_by_value(R + gamma * tf.einsum('ij,i->ij', Z, 1. - self.dones), clip_value_min=v_min,
                                  clip_value_max=v_max)
            # Expanded over in the atom_dimension. Batch_size, n_atoms**2
            Tz = tf.reshape(tf.tile(Tz, [1, n_atoms]), (-1, n_atoms, n_atoms))
            Z = tf.reshape(tf.tile(Z, [1, n_atoms]), shape=(-1, n_atoms, n_atoms))
            # Rescale the bellman operator over the support of the Z random variable
            Tzz = tf.abs(Tz - tf.transpose(Z, perm=(0, 2, 1))) / dz
            Thz = tf.clip_by_value(1 - Tzz, 0, 1)
            # Integrate out the k column of the atom dimension
            self.ThTz = tf.einsum('ijk,ik->ij', Thz, self.p_best)
