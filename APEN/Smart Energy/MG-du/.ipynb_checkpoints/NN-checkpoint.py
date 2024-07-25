import tensorflow as tf
GLOBAL_NET_SCOPE = 'Global_Net'

class ACNet(object):
    def __init__(self, scope, sess,N_S,N_A, *args, **kwargs):
        self.sess = sess
        self.N_S=N_S
        self.N_A=N_A
        self.ENTROPY_BETA = kwargs.get('entropy_beta',0.9)
        self.LR_A = kwargs.get('lr_a',0.0001)  # learning rate for actor
        self.LR_C = kwargs.get('lr_c',0.001)  # learning rate for critic
        self.OPT_A=kwargs.get('opt_a',tf.train.RMSPropOptimizer(self.LR_A, name='RMSPropA'))
        self.OPT_C=kwargs.get('opt_c',tf.train.RMSPropOptimizer(self.LR_C, name='RMSPropA'))
        self.globalAC=kwargs.get('global_net',None)
        # 定义global net
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu, sigma + 1e-4
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.squeeze(normal_dist.sample(1), axis=0)
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, self.N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu, sigma + 1e-4
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = self.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.squeeze(normal_dist.sample(1), axis=0) 
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, self.globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, self.globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, self.globalAC.a_params))
                    self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, self.globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('critic'):
            cell_size = 64
            s = tf.expand_dims(self.s, axis=1, name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation
            l_c1 = tf.layers.dense(cell_out, 120, tf.nn.relu, kernel_initializer=w_init, name='lc1')
            l_c2 = tf.layers.dense(l_c1, 180, tf.nn.relu, kernel_initializer=w_init, name='lc2')
            l_c3 = tf.layers.dense(l_c2, 120, tf.nn.relu, kernel_initializer=w_init, name='lc3')
            v = tf.layers.dense(l_c3, 1, kernel_initializer=w_init, name='v')  # state value
        with tf.variable_scope('actor'):
            l_a1_mu = tf.layers.dense(cell_out, 120, tf.nn.relu6, kernel_initializer=w_init, name='la1_mu')
            l_a1_sigma = tf.layers.dense(cell_out, 120, tf.nn.relu6, kernel_initializer=w_init, name='la1_sigma')
            l_a2_mu = tf.layers.dense(l_a1_mu, 170, tf.nn.relu6, kernel_initializer=w_init, name='la2_mu')
            l_a2_sigma = tf.layers.dense(l_a1_sigma, 170, tf.nn.relu6, kernel_initializer=w_init, name='la2_sigma')
            l_a3_mu = tf.layers.dense(l_a2_mu, 120, tf.nn.relu6, kernel_initializer=w_init, name='la3_mu')
            l_a3_sigma = tf.layers.dense(l_a2_sigma, 120, tf.nn.relu6, kernel_initializer=w_init, name='la3_sigma')
            mu = tf.layers.dense(l_a3_mu, self.N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a3_sigma, self.N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state):  # run by a local
        s = s[tf.newaxis, :]
        a, cell_state = self.sess.run([self.A, self.final_state], {self.s: s, self.init_state: cell_state})
        return a[0], cell_state