import numpy as np
import tensorflow as tf
import config

FLAGS = config.flags.FLAGS

gamma = FLAGS.gamma  # reward discount factor
history_len = FLAGS.history_len
lr = FLAGS.lr    # learning rate 
h_nodes = 64
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class DQN:
    def __init__(self, sess, state_dim, sup_state_dim, action_dim, n_agents, nn_id, use_as_peer=False):
        self.sess = sess
        self.state_dim = state_dim
        self.sup_state_dim = sup_state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        global_scope = 'dqn_' + str(nn_id)

        # placeholders
        tf.compat.v1.disable_eager_execution()
        self.state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])
        self.next_state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents, state_dim])

        self.action_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, n_agents])
        self.reward_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents])
        self.is_not_terminal_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_agents])
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32)

        self.q_networks = []
        self.target_q_networks = []
        self.peer_q_networks = []
        self.td_errors = []
        self.train_networks = []
        self.update_slow_target_dqns = []
        self.update_peer_dqns = []

        for i in range(n_agents):
            with tf.GradientTape(persistent=True) as tape:
                scope = global_scope+"_"+str(i)
                with tf.compat.v1.variable_scope(scope):
                    q_network = self.generate_dqn(self.state_ph[:,i])
                self.q_networks.append(q_network)

                with tf.compat.v1.variable_scope('slow_target_'+scope):
                    target_q_network = self.generate_dqn(self.next_state_ph[:, i], False)
                self.target_q_networks.append(target_q_network)
                q_network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                target_q_network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_'+scope)

                discount = self.is_not_terminal_ph[:,i] * gamma
                a_onehot = tf.one_hot(self.action_ph[:,i], action_dim, 1.0, 0.0)
                target = self.reward_ph[:,i] + discount * tf.reduce_max(target_q_network, axis=1)
                Q_act  = tf.reduce_sum(q_network*a_onehot, axis=1)
                #calculate loss 
                td_error = tf.reduce_sum(tf.square(target - Q_act))
                tape.watch(q_network_vars)

                #apply gradients 
                #gradients = tape.gradient(td_error, q_network_vars)
                gradients = tape.gradient(td_error, q_network_vars)
                
                optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
                train_network = optimizer.apply_gradients(zip(gradients, q_network_vars))

            self.td_errors.append(td_error)
            self.train_networks.append(train_network)
            
            # copy weights from q_network to target q_network
            update_slow_target_ops = []
            for j in range(len(q_network_vars)):
                assign_op = tf.compat.v1.assign(target_q_network_vars[j], q_network_vars[j])

                update_slow_target_ops.append(assign_op)
            update_slow_target_dqn = tf.group(*update_slow_target_ops)
            self.update_slow_target_dqns.append(update_slow_target_dqn)

            if use_as_peer:
                with tf.compat.v1.variable_scope('peer_'+scope):
                    peer_q_network = self.generate_dqn(self.next_state_ph[:, i], False)
                    
                self.peer_q_networks.append(peer_q_network)
                peer_q_network_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='peer_'+scope)
                # copy weights from q_network to peer q_network
                update_peer_ops = []
                for j in range(len(q_network_vars)):
                    assign_op = tf.compat.v1.assign(peer_q_network_vars[j], q_network_vars[j])
                    update_peer_ops.append(assign_op)

                    #print tensor
                    """with tf.compat.v1.Session() as sess:
                        sess.run(tf.compat.v1.global_variables_initializer())
                        tensor_value = sess.run(assign_op)
                        print("Tensor value:", tensor_value)"""
                
                update_peer_dqn = tf.group(*update_peer_ops)
                self.update_peer_dqns.append(update_peer_dqn)

        self.concat_dqns = tf.reshape(tf.concat(self.q_networks, 1), (-1, n_agents, action_dim))
        self.concat_target_dqns = tf.reshape(tf.concat(self.target_q_networks, 1), (-1, n_agents, action_dim))

        if use_as_peer:
            self.concat_peer_dqns = tf.reshape(tf.concat(self.peer_q_networks, 1), (-1, n_agents, action_dim))

    def generate_dqn(self, s, trainable=True):

        side = int(np.sqrt((self.state_dim - self.sup_state_dim*history_len)//(history_len*3)))

        if self.sup_state_dim > 0:
            obs = tf.reshape(tf.cast(s, tf.float32), (-1, history_len, self.state_dim // history_len))
            sup = tf.reshape(obs[:,:,-1*self.sup_state_dim:], (-1, history_len*self.sup_state_dim))            
            obs = tf.reshape(obs[:,:,:-1*self.sup_state_dim], (-1, history_len, side*side*3))
            obs = tf.transpose(obs, perm=[0,2,1])
            obs = tf.reshape(obs, (-1,side,side,history_len*3))
        else:
            obs = tf.reshape(s, (-1, history_len, side*side*3))
            obs = tf.transpose(obs, perm=[0,2,1])
            obs = tf.reshape(obs, (-1,side,side,history_len*3))

        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu)(obs)

        # conv1 = tf.nn.pool(conv1, (3,3), "MAX", "VALID")
        conv1 = tf.keras.layers.Flatten()(conv1)

        if self.sup_state_dim > 0:
            concat = tf.concat([conv1, sup], axis=1)
        else:
            concat = conv1
        
        hidden = tf.keras.layers.Dense(h_nodes, activation=tf.nn.relu, name='dense_a1')(concat)

        hidden2 = tf.keras.layers.Dense(h_nodes, activation=tf.nn.relu, name='dense_a2')(hidden)

        q_values = tf.keras.layers.Dense(self.action_dim, trainable=trainable, name='qvals')(hidden2)

        return q_values

    def get_q_values(self, state_ph):
        return self.sess.run(self.concat_dqns, feed_dict={self.state_ph: state_ph})
    
    def training_qnet(self, state_ph, action_ph, reward_ph, is_not_terminal_ph, next_state_ph, lr=lr):
        return self.sess.run([self.td_errors, self.train_networks], 
            feed_dict={
                self.state_ph: state_ph,
                self.next_state_ph: next_state_ph,
                self.action_ph: action_ph,
                self.reward_ph: reward_ph,
                self.is_not_terminal_ph: is_not_terminal_ph,
                self.lr: lr})

    def training_target_qnet(self):
        self.sess.run(self.update_slow_target_dqns)
