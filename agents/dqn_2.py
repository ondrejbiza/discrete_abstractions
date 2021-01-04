import numpy as np
import tensorflow as tf
from agents.dqn import DQN
from agents import utils
import constants
import agents.tf_utils as U


class DQN2(DQN):

    def __init__(self, env, state_shape, num_actions, num_filters_list, filter_size_list, stride_list, hiddens,
                 learning_rate, batch_size, optimizer, exploration_fraction, init_explore, final_explore, max_timesteps,
                 target_net=True, target_update_freq=100, grad_norm_clipping=None, buffer_size=10000,
                 prioritized_replay=False, pr_alpha=0.6, pr_beta=0.4, pr_eps=1e-6, gamma=0.9, custom_graph=False,
                 mask_dones=True):

        # TODO: output shape needs to be (2, 1) to get 4 actions--very ugly hack
        super(DQN2, self).__init__(
            env, state_shape, (2, 1), num_filters_list, filter_size_list, stride_list, hiddens,
            learning_rate, batch_size, optimizer, exploration_fraction, init_explore, final_explore, max_timesteps,
            target_net=target_net, target_update_freq=target_update_freq, grad_norm_clipping=grad_norm_clipping,
            buffer_size=buffer_size, depth_mean=None, prioritized_replay=prioritized_replay, pr_alpha=pr_alpha,
            pr_beta=pr_beta, pr_eps=pr_eps, gamma=gamma, custom_graph=custom_graph, target_size=None
        )

        self.num_actions = num_actions
        self.mask_dones = mask_dones

    def act(self, observation, timestep):
        """
        Take a single step in the environment.
        :param observation:         An observation of teh current state.
        :param timestep:            The current time step (from the start of the training, sets the value of epsilon).
        :return:                    Current and next state of the environment with other useful information.
        """

        with self.tf_graph.as_default():
            with self.session.as_default():

                # get gamma values
                q_values = self.session.run(self.get_q, feed_dict={
                    self.depth_pl: np.expand_dims(observation, axis=0)
                })[0]

                action = np.argmax(q_values)

                if self.exploration_schedule is not None and np.random.rand() < \
                        self.exploration_schedule.value(timestep):
                    action = np.random.randint(self.num_actions)

                # execute action
                new_observation, reward, done, _ = self.env.step(action)

                # remember experience
                self.replay_buffer.add(
                    observation, action, reward, new_observation, done
                )

                return observation, q_values, None, action, new_observation, None, None, None, reward, done

    def act_v2(self, observation, timestep):

        with self.tf_graph.as_default():
            with self.session.as_default():

                # get gamma values
                q_values = self.session.run(self.get_q, feed_dict={
                    self.depth_pl: np.expand_dims(observation, axis=0)
                })[0]

                #q_values_noise = q_values + np.random.random(np.shape(q_values)) * 0.01
                action = np.argmax(q_values)

                if self.exploration_schedule is not None and np.random.rand() < \
                        self.exploration_schedule.value(timestep):
                    action = np.random.randint(self.num_actions)

                return action, action, q_values

    def learn(self, timestep):
        """
        Take a single training step.
        :return:        None.
        """

        with self.tf_graph.as_default():
            with self.session.as_default():

                # get batch
                if self.prioritized_replay:
                    beta = self.beta_schedule.value(timestep)
                    states, actions, rewards, next_states, dones, weights, batch_indexes = \
                        self.replay_buffer.sample(self.batch_size, beta)
                else:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    weights, batch_indexes = np.ones_like(rewards), None

                # calculate targets
                a = rewards

                if self.mask_dones:
                    b = (1 - dones) * self.gamma
                else:
                    b = np.ones_like(dones, dtype=np.float32) * self.gamma

                if self.target_net:
                    qs = self.session.run(self.get_target_q, feed_dict={
                        self.depth_pl: next_states
                    })
                else:
                    qs = self.session.run(self.get_q, feed_dict={
                        self.depth_pl: next_states
                    })

                c = np.max(qs, axis=1)

                targets = a + b * c

                # train
                td_error, _ = self.session.run([self.td_error, self.train_step], feed_dict={
                    self.depth_pl: states,
                    self.actions_pl: actions,
                    self.targets_pl: targets,
                    self.importance_weights_pl: weights
                })

                new_priorities = np.abs(td_error) + self.pr_eps

                if self.prioritized_replay:
                    self.replay_buffer.update_priorities(batch_indexes, new_priorities)

        # maybe update target network
        if self.target_net and timestep % self.target_update_freq == 0 and timestep > 0:
            self.session.run(self.target_update_op)

    def build_placeholders_(self):

        self.depth_pl = tf.placeholder(tf.float32, shape=(None, *self.state_shape), name="depth_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="labels_pl")
        self.targets_pl = tf.placeholder(tf.float32, shape=(None,), name="targets_pl")
        self.importance_weights_pl = tf.placeholder(tf.float32, shape=(None,), name="importance_weights_pl")

    def build_network_(self, namespace):

        with tf.variable_scope(namespace):

            x = self.depth_pl

            with tf.variable_scope("convs"):

                for i in range(len(self.num_filters_list)):

                    with tf.variable_scope("conv{:d}".format(i + 1)):

                        x = tf.layers.conv2d(
                            x, self.num_filters_list[i], self.filter_size_list[i], self.stride_list[i],
                            padding="SAME", activation=tf.nn.relu,
                            kernel_initializer=utils.get_mrsa_initializer()
                        )

            x = tf.layers.flatten(x, name="flatten")

            with tf.variable_scope("fcs"):

                for i in range(len(self.hiddens)):

                    with tf.variable_scope("fc{:d}".format(i + 1)):

                        x = tf.layers.dense(
                            x, self.hiddens[i], activation=tf.nn.relu
                        )

            with tf.variable_scope("logits"):

                logits = tf.layers.dense(x, self.num_actions)

                return logits

    def build_training_(self):

        mask = tf.one_hot(self.actions_pl, self.num_actions, dtype=tf.float32)
        qs = tf.reduce_sum(self.get_q * mask, axis=1)

        self.td_error = qs - tf.stop_gradient(self.targets_pl)
        self.errors = self.importance_weights_pl * U.huber_loss(self.td_error)

        if self.optimizer == constants.OPT_ADAM:
            opt = tf.train.AdamOptimizer(self.learning_rate, self.MOMENTUM)
        elif self.optimizer == constants.OPT_MOMENTUM:
            opt = tf.train.MomentumOptimizer(self.learning_rate, self.MOMENTUM)
        else:
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        if self.grad_norm_clipping is not None:
            # clip gradients to some norm
            gradients = opt.compute_gradients(self.errors)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
            self.train_step = opt.apply_gradients(gradients)
        else:
            self.train_step = opt.minimize(
                self.errors
            )
