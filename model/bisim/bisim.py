import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import agents.utils as agent_utils


class FullOracleModel:

    def __init__(self, env, input_shape, num_blocks, encoder_filters, encoder_filter_sizes, encoder_strides,
                 encoder_neurons, learning_rate, weight_decay, gamma, batch_norm=False):

        self.env = env
        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.batch_norm = batch_norm

    def encode(self, states, batch_size=100):

        assert states.shape[0]

        num_steps = int(np.ceil(states.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            embedding = self.session.run(self.state_block_t, feed_dict={
                self.states_pl: states[batch_slice]
            })

            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def build(self):

        self.build_placeholders_and_constants()
        self.build_model()
        self.build_training()

    def build_placeholders_and_constants(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.r_c = tf.constant(self.env.r, dtype=tf.float32)
        self.p_c = tf.constant(self.env.p, dtype=tf.float32)

    def build_model(self):

        self.state_block_t = self.build_encoder(self.states_pl)
        self.next_state_block_t = self.build_encoder(self.next_states_pl, share_weights=True)

    def build_training(self):

        r_t = tf.gather(self.r_c, self.actions_pl)
        p_t = tf.gather(self.p_c, self.actions_pl)
        dones_t = tf.cast(self.dones_pl, tf.float32)

        self.reward_loss_t = tf.square(self.rewards_pl - tf.reduce_sum(self.state_block_t * r_t, axis=1))

        self.transition_loss_t = tf.reduce_sum(
            tf.square(tf.stop_gradient(self.next_state_block_t) - tf.matmul(tf.expand_dims(self.state_block_t, axis=1), p_t)[:, 0, :]),
            axis=1
        ) * (1 - dones_t)

        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

        self.loss_t = tf.reduce_mean(
            (1 / 2) * (self.reward_loss_t + self.gamma * self.transition_loss_t), axis=0
        ) + self.regularization_loss_t

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_t)

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.train_step = tf.group(self.train_step, self.update_op)

    def build_encoder(self, input_t, share_weights=False):

        x = tf.expand_dims(input_t, axis=-1)

        with tf.variable_scope("encoder", reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):
                x = tf.layers.dense(
                    x, self.num_blocks, activation=tf.nn.softmax,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

        return x

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()


class PartialOracleModel:

    ENCODER_NAMESPACE = "encoder"
    TARGET_ENCODER_NAMESPACE = "target_encoder"

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_model, weight_decay, gamma,
                 optimizer_encoder, optimizer_model, max_steps, batch_norm=True, target_network=False,
                 add_hand_state=False, add_entropy=False, entropy_from=10000, entropy_start=0.0, entropy_end=0.1,
                 ce_transitions=False):

        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_model = learning_rate_model
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.batch_norm = batch_norm
        self.target_network = target_network
        self.add_hand_state = add_hand_state
        self.add_entropy = add_entropy
        self.entropy_from = entropy_from
        self.entropy_start = entropy_start
        self.entropy_end = entropy_end
        self.ce_transitions = ce_transitions

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, states, batch_size=100, hand_states=None):

        assert states.shape[0]

        num_steps = int(np.ceil(states.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice],
                self.is_training_pl: False
            }

            if hand_states is not None:
                feed_dict[self.hand_states_pl] = hand_states[batch_slice]

            embedding = self.session.run(self.state_block_t, feed_dict=feed_dict)

            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        if self.add_hand_state:
            self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
            self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_model(self):

        self.state_block_t = self.build_encoder(self.states_pl, hand_state=self.hand_states_pl)

        if self.target_network:
            self.next_state_block_t = self.build_encoder(
                self.next_states_pl, share_weights=False, namespace=self.TARGET_ENCODER_NAMESPACE,
                hand_state=self.next_hand_states_pl
            )
            self.build_target_update()
        else:
            self.next_state_block_t = self.build_encoder(
                self.next_states_pl, share_weights=True, hand_state=self.next_hand_states_pl
            )

        self.r_t = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )
        self.p_t = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )

    def build_training(self):

        self.global_step = tf.train.get_or_create_global_step()

        r_t = tf.gather(self.r_t, self.actions_pl)
        p_t = tf.gather(self.p_t, self.actions_pl)
        dones_t = tf.cast(self.dones_pl, tf.float32)

        self.reward_loss_t = (1 / 2) * tf.square(self.rewards_pl - tf.reduce_sum(self.state_block_t * r_t, axis=1))

        if self.ce_transitions:
            # treat p_t as log probabilities
            p_t = tf.nn.softmax(p_t, axis=-1)

            # predict next state
            next_state = tf.matmul(tf.expand_dims(self.state_block_t, axis=1), p_t)[:, 0, :]

            # cross entropy between next state probs and predicted probs
            self.transition_loss_t = - self.next_state_block_t * tf.log(next_state + 1e-7)

            self.transition_loss_t = tf.reduce_sum(self.transition_loss_t, axis=-1)
            self.transition_loss_t = self.transition_loss_t * (1 - dones_t)
        else:
            self.transition_loss_t = (1 / 2) * tf.reduce_sum(
                tf.square(tf.stop_gradient(self.next_state_block_t) -
                          tf.matmul(tf.expand_dims(self.state_block_t, axis=1), p_t)[:, 0, :]),
                axis=1
            ) * (1 - dones_t)

        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

        self.loss_t = tf.reduce_mean(
            self.reward_loss_t + self.gamma * self.transition_loss_t, axis=0
        ) + self.regularization_loss_t

        if self.add_entropy:
            plogs = self.state_block_t * tf.log(self.state_block_t + 1e-7)
            self.entropy_loss_t = tf.reduce_mean(tf.reduce_sum(- plogs, axis=1), axis=0)

            f = tf.maximum(0.0, tf.cast(self.global_step - self.entropy_from, tf.float32)) / \
                (self.max_steps - self.entropy_from)
            f = f * (self.entropy_end - self.entropy_start) + self.entropy_start

            self.loss_t += f * self.entropy_loss_t

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        model_variables = [self.r_t, self.p_t]

        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)
        model_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_model)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

        self.model_train_step = model_optimizer.minimize(
            self.loss_t, var_list=model_variables
        )

        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def build_encoder(self, input_t, share_weights=False, namespace=ENCODER_NAMESPACE, hand_state=None):

        x = tf.expand_dims(input_t, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            if hand_state is not None:
                x = tf.concat([x, hand_state], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):
                x = tf.layers.dense(
                    x, self.num_blocks, activation=tf.nn.softmax,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

        return x

    def build_target_update(self):

        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ENCODER_NAMESPACE)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_ENCODER_NAMESPACE)

        assert len(source_vars) == len(target_vars) and len(source_vars) > 0

        update_ops = []
        for source_var, target_var in zip(source_vars, target_vars):
            update_ops.append(tf.assign(target_var, source_var))

        self.target_update_op = tf.group(*update_ops)

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()


class GumbelModel:

    ENCODER_NAMESPACE = "encoder"
    TARGET_ENCODER_NAMESPACE = "target_encoder"

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_p,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, batch_norm=True,
                 target_network=True, gamma_schedule=None, straight_through=False, kl=False, kl_weight=1.0,
                 oracle_r=None, oracle_p=None, transitions_mse=False, correct_ce=False):

        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_r = learning_rate_r
        self.learning_rate_p = learning_rate_p
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.batch_norm = batch_norm
        self.target_network = target_network
        self.gamma_schedule = gamma_schedule
        self.straight_through = straight_through
        self.kl = kl
        self.kl_weight = kl_weight
        self.oracle_r = oracle_r
        self.oracle_p = oracle_p
        self.transitions_mse = transitions_mse
        self.correct_ce = correct_ce

        if self.gamma_schedule is not None:
            assert len(self.gamma) == len(self.gamma_schedule) + 1

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, states, batch_size=100, hand_states=None):

        assert states.shape[0]

        num_steps = int(np.ceil(states.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice],
                self.is_training_pl: False
            }

            if hand_states is not None:
                feed_dict[self.hand_states_pl] = hand_states[batch_slice]

            embedding = self.session.run(self.state_block_samples_t, feed_dict=feed_dict)

            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

        self.temperature_pl = tf.placeholder(tf.float32, shape=[], name="temperature_pl")

    def build_model(self):

        # encode state block
        self.state_block_logits_t = self.build_encoder(self.states_pl, hand_state=self.hand_states_pl)

        agent_utils.summarize(self.state_block_logits_t, "state_block_logits_t")

        self.state_block_cat_dist = tf.contrib.distributions.OneHotCategorical(
            logits=self.state_block_logits_t
        )
        self.state_block_samples_t = tf.cast(self.state_block_cat_dist.sample(), tf.int32)

        self.state_block_sg_dist = tf.contrib.distributions.RelaxedOneHotCategorical(
            self.temperature_pl, logits=self.state_block_logits_t
        )
        self.state_block_sg_samples_t = self.state_block_sg_dist.sample()

        agent_utils.summarize(self.state_block_sg_samples_t, "state_block_sg_samples_t")

        if self.straight_through:

            # hard sample
            self.state_block_sg_samples_hard_t = \
                tf.cast(tf.one_hot(tf.argmax(self.state_block_sg_samples_t, -1), self.num_blocks), tf.float32)

            # fake gradients for the hard sample
            self.state_block_sg_samples_t = \
                tf.stop_gradient(self.state_block_sg_samples_hard_t - self.state_block_sg_samples_t) + \
                self.state_block_sg_samples_t

            agent_utils.summarize(self.state_block_sg_samples_hard_t, "state_block_sg_samples_hard_t")

        # encode next state block
        if self.target_network:
            self.next_state_block_logits_t = self.build_encoder(
                self.next_states_pl, share_weights=False, namespace=self.TARGET_ENCODER_NAMESPACE,
                hand_state=self.next_hand_states_pl
            )
            self.build_target_update()
        else:
            self.next_state_block_logits_t = self.build_encoder(
                self.next_states_pl, share_weights=True, hand_state=self.next_hand_states_pl
            )

        self.next_state_block_cat_dist = tf.contrib.distributions.OneHotCategorical(
            logits=self.next_state_block_logits_t
        )
        self.next_state_block_samples_t = tf.cast(self.next_state_block_cat_dist.sample(), tf.float32)

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )
        self.r_t = self.r_v
        self.p_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )

        if not self.transitions_mse:
            self.p_t = tf.nn.softmax(self.p_v, axis=-1)
        else:
            self.p_t = self.p_v

    def build_training(self):

        # set up global step variable
        self.global_step = tf.train.get_or_create_global_step()

        # gather reward and transition matrices for each action
        if self.oracle_r is not None:
            r_t = tf.gather(self.oracle_r, self.actions_pl)
        else:
            r_t = tf.gather(self.r_t, self.actions_pl)

        if self.oracle_p is not None:
            p_t = tf.gather(self.oracle_p, self.actions_pl)
        else:
            p_t = tf.gather(self.p_t, self.actions_pl)

        dones_t = tf.cast(self.dones_pl, tf.float32)

        # reward loss
        self.reward_loss_t = tf.square(self.rewards_pl - tf.reduce_sum(self.state_block_sg_samples_t * r_t, axis=1))

        # transition loss
        next_state = tf.matmul(tf.expand_dims(self.state_block_sg_samples_t, axis=1), p_t)[:, 0, :]

        if self.transitions_mse:
            self.transition_loss_t = tf.reduce_sum(
                tf.square(tf.stop_gradient(self.next_state_block_samples_t) - next_state),
                axis=1
            ) * (1 - dones_t)
        else:
            if self.correct_ce:
                self.transition_loss_t = tf.reduce_sum(
                    - tf.stop_gradient(tf.nn.softmax(self.next_state_block_logits_t)) * tf.log(next_state + 1e-7),
                    axis=1
                ) * (1 - dones_t)
            else:
                self.transition_loss_t = tf.reduce_sum(
                    - tf.stop_gradient(self.next_state_block_samples_t) * tf.log(next_state + 1e-7),
                    axis=1
                ) * (1 - dones_t)

        # weight decay regularizer
        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0.0

        # kl divergence regularizer
        if self.kl:
            prior_logits_t = tf.ones_like(self.state_block_logits_t) / self.num_blocks
            prior_cat_dist = tf.contrib.distributions.OneHotCategorical(logits=prior_logits_t)
            kl_divergence_t = tf.contrib.distributions.kl_divergence(self.state_block_cat_dist, prior_cat_dist)
            self.kl_loss_t = tf.reduce_mean(kl_divergence_t)
        else:
            self.kl_loss_t = 0.0

        # final loss
        if self.gamma_schedule is not None:
            gamma = tf.train.piecewise_constant(self.global_step, self.gamma_schedule, self.gamma)
        else:
            gamma = self.gamma

        self.loss_t = tf.reduce_mean(
            self.reward_loss_t + gamma * self.transition_loss_t, axis=0
        ) + self.regularization_loss_t + self.kl_weight * self.kl_loss_t

        # encoder optimizer
        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

        # add batch norm updates
        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

        # model optimizer if not full oracle
        if self.oracle_r is None or self.oracle_p is None:

            r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)
            r_step = r_optimizer.minimize(
                self.reward_loss_t, var_list=[self.r_v]
            )

            p_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_p)
            p_step = p_optimizer.minimize(
                self.transition_loss_t, var_list=[self.p_v]
            )

            self.model_train_step = tf.group(r_step, p_step)

            self.train_step = tf.group(self.encoder_train_step, self.model_train_step)
        else:
            self.train_step = self.encoder_train_step

    def build_encoder(self, input_t, share_weights=False, namespace=ENCODER_NAMESPACE, hand_state=None):

        x = tf.expand_dims(input_t, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            if hand_state is not None:
                x = tf.concat([x, hand_state], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):
                x = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

        return x

    def build_target_update(self):

        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ENCODER_NAMESPACE)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_ENCODER_NAMESPACE)

        assert len(source_vars) == len(target_vars) and len(source_vars) > 0

        update_ops = []
        for source_var, target_var in zip(source_vars, target_vars):
            update_ops.append(tf.assign(target_var, source_var))

        self.target_update_op = tf.group(*update_ops)

    def build_summaries(self):

        # losses
        tf.summary.scalar("loss", self.loss_t)
        tf.summary.scalar("transition_loss", tf.reduce_mean(self.transition_loss_t))
        tf.summary.scalar("reward_loss", tf.reduce_mean(self.reward_loss_t))

        # logits grads
        grad_t = tf.gradients(tf.reduce_mean(self.transition_loss_t), self.state_block_logits_t)
        grad_r = tf.gradients(tf.reduce_mean(self.reward_loss_t), self.state_block_logits_t)

        norm_grad_t = tf.norm(grad_t, ord=2, axis=-1)[0]
        norm_grad_r = tf.norm(grad_r, ord=2, axis=-1)[0]

        agent_utils.summarize(norm_grad_t, "logits_grad_t")
        agent_utils.summarize(norm_grad_r, "logits_grad_r")

        # samples grads
        grad_t = tf.gradients(tf.reduce_mean(self.transition_loss_t), self.state_block_sg_samples_t)
        grad_r = tf.gradients(tf.reduce_mean(self.reward_loss_t), self.state_block_sg_samples_t)

        norm_grad_t = tf.norm(grad_t, ord=2, axis=-1)[0]
        norm_grad_r = tf.norm(grad_r, ord=2, axis=-1)[0]

        agent_utils.summarize(norm_grad_t, "sg_samples_grad_t")
        agent_utils.summarize(norm_grad_r, "sg_samples_grad_r")

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()


class ExpectationModel:

    ENCODER_NAMESPACE = "encoder"
    TARGET_ENCODER_NAMESPACE = "target_encoder"

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, batch_norm=True,
                 target_network=True, oracle_r=None, oracle_t=None, propagate_next_state=False,
                 z_transform=False, abs_z_transform=False, sigsoftmax=False,  encoder_tau=1.0, model_tau=1.0,
                 no_tau_target_encoder=False, kl_penalty=False, kl_penalty_weight=0.01, small_r_init=False,
                 small_t_init=False):

        if propagate_next_state:
            assert not target_network

        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_r = learning_rate_r
        self.learning_rate_t = learning_rate_t
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.batch_norm = batch_norm
        self.target_network = target_network
        self.oracle_r = oracle_r
        self.oracle_t = oracle_t
        self.propagate_next_state = propagate_next_state
        self.z_transform = z_transform
        self.abs_z_transform = abs_z_transform
        self.sigsoftmax = sigsoftmax
        self.encoder_tau = encoder_tau
        self.model_tau = model_tau
        self.no_tau_target_encoder = no_tau_target_encoder
        self.kl_penalty = kl_penalty
        self.kl_penalty_weight = kl_penalty_weight
        self.small_r_init = small_r_init
        self.small_t_init = small_t_init

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, depths, hand_states, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }

            embedding = self.session.run(self.state_softmax_t, feed_dict=feed_dict)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def validate(self, depths, hand_states, actions, rewards, next_depths, next_hand_states, dones, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2 = self.session.run([self.full_transition_loss_t, self.full_reward_loss_t], feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, next_depths, next_hand_states, dones,
                            batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2 = self.session.run([
                self.state_softmax_t, self.full_transition_loss_t, self.full_reward_loss_t], feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_model(self):

        self.state_logits_t, self.state_softmax_t = self.build_encoder(
            self.states_pl, self.hand_states_pl, self.encoder_tau
        )

        self.perplexity_t = tf.constant(2, dtype=tf.float32) ** (
            - tf.reduce_mean(
                tf.reduce_sum(
                    self.state_softmax_t * tf.log(self.state_softmax_t + 1e-7) /
                    tf.log(tf.constant(2, dtype=self.state_softmax_t.dtype)),
                    axis=1
                ),
                axis=0
            )
        )

        if self.no_tau_target_encoder:
            target_tau = 1.0
        else:
            target_tau = self.encoder_tau

        if self.target_network:
            self.next_state_logits_t, self.next_state_softmax_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, target_tau, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_logits_t, self.next_state_softmax_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, target_tau, share_weights=True
            )

        self.build_r_model()
        self.build_t_model()

    def build_r_model(self):

        if self.small_r_init:
            r_init = tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32)
        else:
            r_init = tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=r_init
        )
        self.r_t = self.r_v

    def build_t_model(self):

        if self.small_t_init:
            t_init = tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32)
        else:
            t_init = tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=t_init
        )

        self.t_softmax_t = tf.nn.softmax(self.t_v / self.model_tau, axis=2)
        self.t_logsoftmax_t = tf.nn.log_softmax(self.t_v / self.model_tau, axis=2)

    def build_training(self):

        # prep
        self.global_step = tf.train.get_or_create_global_step()
        self.gather_matrices()
        self.dones_float_t = tf.cast(self.dones_pl, tf.float32)

        # build losses
        self.build_reward_loss()
        self.build_transition_loss()
        self.build_regularization_loss()
        self.build_kl_penalty()

        # build full loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False)

        self.loss_t = self.reward_loss_t + tf.stop_gradient(self.gamma_v) * self.transition_loss_t + \
                      self.regularization_loss_t + self.kl_penalty_weight_v * self.kl_penalty_t

        # build training
        self.build_encoder_training()
        self.build_model_training()

        # integrate training into a single op
        if self.model_train_step is None:
            self.train_step = self.encoder_train_step
        else:
            self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def gather_matrices(self):

        if self.oracle_r is not None:
            self.r_gather_t = tf.gather(self.oracle_r, self.actions_pl)
        else:
            self.r_gather_t = tf.gather(self.r_t, self.actions_pl)

        if self.oracle_t is not None:
            self.t_logsoftmax_gather_t = tf.gather(self.oracle_t, self.actions_pl)
        else:
            self.t_logsoftmax_gather_t = tf.gather(self.t_logsoftmax_t, self.actions_pl)

    def build_reward_loss(self):

        term1 = tf.square(self.rewards_pl[:, tf.newaxis] - self.r_gather_t)
        term2 = term1 * self.state_softmax_t

        self.full_reward_loss_t = (1 / 2) * tf.reduce_sum(term2, axis=1)
        self.reward_loss_t = (1 / 2) * tf.reduce_mean(tf.reduce_sum(term2, axis=1), axis=0)

    def build_transition_loss(self):

        if self.propagate_next_state:
            self.transition_term1 = self.state_softmax_t[:, :, tf.newaxis] * self.next_state_softmax_t[:, tf.newaxis, :]
        else:
            self.transition_term1 = self.state_softmax_t[:, :, tf.newaxis] * tf.stop_gradient(
                self.next_state_softmax_t[:, tf.newaxis, :]
            )

        self.transition_term2 = self.transition_term1 * self.t_logsoftmax_gather_t

        self.full_transition_loss_t = - tf.reduce_sum(self.transition_term2, axis=[1, 2]) * (1 - self.dones_float_t)

        self.transition_loss_t = tf.reduce_sum(self.full_transition_loss_t, axis=0) / tf.reduce_max(
            [1.0, tf.reduce_sum(1 - self.dones_float_t)]
        )

    def build_regularization_loss(self):

        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

    def build_kl_penalty(self):

        self.kl_penalty_weight_v = tf.Variable(self.kl_penalty_weight, trainable=False, dtype=tf.float32)
        self.kl_penalty_weight_pl = tf.placeholder(tf.float32, shape=[], name="kl_penalty_weight_pl")
        self.kl_penalty_weight_assign = tf.assign(self.kl_penalty_weight_v, self.kl_penalty_weight_pl)

        if self.kl_penalty:
            log_softmax = tf.nn.log_softmax(self.state_logits_t, axis=-1)
            self.kl_penalty_t = tf.reduce_mean(tf.reduce_sum(self.state_softmax_t * log_softmax, axis=-1), axis=0)
        else:
            self.kl_penalty_t = 0.0

    def build_encoder_training(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

    def build_model_training(self):

        model_train_step = []

        if self.oracle_r is None:
            r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)
            self.r_step = r_optimizer.minimize(
                self.reward_loss_t, var_list=[self.r_v]
            )
            model_train_step.append(self.r_step)

        if self.oracle_t is None:
            t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
            self.t_step = t_optimizer.minimize(
                self.transition_loss_t, var_list=[self.t_v]
            )
            model_train_step.append(self.t_step)

        if len(model_train_step) > 0:
            self.model_train_step = tf.group(*model_train_step)
        else:
            self.model_train_step = None

    def build_encoder(self, depth_pl, hand_state_pl, tau, share_weights=False, namespace=ENCODER_NAMESPACE):

        x = tf.expand_dims(depth_pl, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):
                x = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

        if self.z_transform:
            dist = x - tf.reduce_min(x, axis=1)[:, tf.newaxis]
            dist = dist / tf.reduce_sum(dist, axis=1)[:, tf.newaxis]
        elif self.abs_z_transform:
            abs_x = tf.abs(x)
            dist = abs_x / tf.reduce_sum(abs_x, axis=1)[:, tf.newaxis]
        elif self.sigsoftmax:
            e_x = tf.exp(x - tf.reduce_max(x, axis=1)[:, tf.newaxis])
            sig_e_x = e_x * tf.nn.sigmoid(x)
            dist = sig_e_x / tf.reduce_sum(sig_e_x, axis=1)[:, tf.newaxis]
        else:
            dist = tf.nn.softmax(x / tau)

        return x, dist

    def build_target_update(self):

        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ENCODER_NAMESPACE)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_ENCODER_NAMESPACE)

        assert len(source_vars) == len(target_vars) and len(source_vars) > 0

        update_ops = []
        for source_var, target_var in zip(source_vars, target_vars):
            update_ops.append(tf.assign(target_var, source_var))

        self.target_update_op = tf.group(*update_ops)

    def build_summaries(self):

        # losses
        tf.summary.scalar("loss", self.loss_t)
        tf.summary.scalar("transition_loss", tf.reduce_mean(self.transition_loss_t))
        tf.summary.scalar("reward_loss", tf.reduce_mean(self.reward_loss_t))

        # logits and softmax
        agent_utils.summarize(self.state_logits_t, "logits")
        agent_utils.summarize(self.state_softmax_t, "softmax")

        # gradients
        self.grad_norms_d = dict()

        for target, target_name in zip(
            [self.loss_t, self.transition_loss_t, self.reward_loss_t], ["total_loss", "t_loss", "r_loss"]
        ):
            for source, source_name in zip([self.state_logits_t, self.state_softmax_t], ["logits", "softmax"]):

                grads = tf.gradients(tf.reduce_mean(target), source)
                grad_norms = tf.norm(grads, ord=1, axis=-1)[0]
                name = "state_{}_grad_{}".format(source_name, target_name)
                self.grad_norms_d[name] = tf.reduce_mean(grad_norms)

                agent_utils.summarize(grad_norms, name)

    def set_gamma(self, value):

        self.session.run(tf.assign(self.gamma_v, value))

    def set_kl_penalty_weight(self, value):

        self.session.run(self.kl_penalty_weight_assign, feed_dict={self.kl_penalty_weight_pl: value})

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()

    def save_matrices_as_images(self, step, save_dir, ext="pdf"):

        r, p = self.session.run([self.r_t, self.t_softmax_t])

        r = np.reshape(r, (r.shape[0], -1))
        p = np.reshape(p, (p.shape[0], -1))

        r_path = os.path.join(save_dir, "r_{:d}.{}".format(step, ext))
        p_path = os.path.join(save_dir, "p_{:d}.{}".format(step, ext))

        plt.clf()
        plt.imshow(r, vmin=-0.5, vmax=1.5)
        plt.colorbar()
        plt.savefig(r_path)

        plt.clf()
        plt.imshow(p, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(p_path)


class ExpectationModelGaussian(ExpectationModel):

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, batch_norm=True,
                 target_network=True, oracle_r=None, oracle_t=None, propagate_next_state=False,
                 z_transform=False, abs_z_transform=False, sigsoftmax=False, encoder_tau=1.0, model_tau=1.0,
                 no_tau_target_encoder=False, kl_penalty=False, kl_penalty_weight=0.01, small_r_init=False,
                 small_t_init=False):

        super(ExpectationModelGaussian, self).__init__(
            input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes, encoder_strides,
            encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, weight_decay, gamma,
            optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm, target_network=target_network,
            oracle_r=oracle_r, oracle_t=oracle_t, propagate_next_state=propagate_next_state, z_transform=z_transform,
            abs_z_transform=abs_z_transform, sigsoftmax=sigsoftmax, encoder_tau=encoder_tau, model_tau=model_tau,
            no_tau_target_encoder=no_tau_target_encoder, kl_penalty=kl_penalty,
            kl_penalty_weight=kl_penalty_weight, small_r_init=small_r_init, small_t_init=small_t_init
        )

    def build_model(self):

        self.state_mu_t, self.state_sd_t, self.state_var_t, self.state_logits_t, self.state_softmax_t = \
            self.build_encoder(
                self.states_pl, self.hand_states_pl, self.encoder_tau, namespace=self.ENCODER_NAMESPACE
            )

        self.perplexity_t = tf.constant(2, dtype=tf.float32) ** (
            - tf.reduce_mean(
                tf.reduce_sum(
                    self.state_softmax_t * tf.log(self.state_softmax_t + 1e-7) /
                    tf.log(tf.constant(2, dtype=self.state_softmax_t.dtype)),
                    axis=1
                ),
                axis=0
            )
        )

        if self.no_tau_target_encoder:
            target_tau = 1.0
        else:
            target_tau = self.encoder_tau

        if self.target_network:
            self.next_state_mu_t, self.next_state_sd_t, self.next_state_var_t, self.next_state_logits_t, \
                self.next_state_softmax_t = \
                self.build_encoder(
                    self.next_states_pl, self.next_hand_states_pl, target_tau, share_weights=False,
                    namespace=self.TARGET_ENCODER_NAMESPACE
                )
            self.build_target_update()
        else:
            self.next_state_mu_t, self.next_state_sd_t, self.next_state_var_t, self.next_state_logits_t, \
                self.next_state_softmax_t = \
                self.build_encoder(
                    self.next_states_pl, self.next_hand_states_pl, target_tau, share_weights=True,
                    namespace=self.ENCODER_NAMESPACE
                )

        self.build_r_model()
        self.build_t_model()

    def build_encoder(self, depth_pl, hand_state_pl, tau, share_weights=False, namespace=None):

        x = tf.expand_dims(depth_pl, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):
                mu_t = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                log_var_t = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

        var_t = tf.exp(log_var_t)
        sd_t = tf.sqrt(var_t)
        noise_t = tf.random_normal(
            shape=(tf.shape(mu_t)[0], self.num_blocks), mean=0, stddev=1.0
        )
        sample_t = mu_t + sd_t * noise_t

        dist_t = tf.nn.softmax(sample_t / tau)

        return mu_t, sd_t, var_t, sample_t, dist_t

    def build_kl_penalty(self):

        self.kl_penalty_weight_v = tf.Variable(self.kl_penalty_weight, trainable=False, dtype=tf.float32)
        self.kl_penalty_weight_pl = tf.placeholder(tf.float32, shape=[], name="kl_penalty_weight_pl")
        self.kl_penalty_weight_assign = tf.assign(self.kl_penalty_weight_v, self.kl_penalty_weight_pl)

        if self.kl_penalty:
            kl_divergence_t = 0.5 * (tf.square(self.state_mu_t) + self.state_var_t - tf.log(self.state_var_t) - 1.0)
            self.kl_penalty_t = tf.reduce_mean(tf.reduce_sum(kl_divergence_t, axis=1), axis=0)
        else:
            self.kl_penalty_t = 0.0


class ExpectationModelGaussianWithQ(ExpectationModelGaussian):

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, learning_rate_q,
                 weight_decay, reward_gamma, transition_gamma, optimizer_encoder, optimizer_model, max_steps, batch_norm=True,
                 target_network=True, oracle_r=None, oracle_t=None, propagate_next_state=False,
                 z_transform=False, abs_z_transform=False, sigsoftmax=False, encoder_tau=1.0, model_tau=1.0,
                 no_tau_target_encoder=False, kl_penalty=False, kl_penalty_weight=0.01, small_r_init=False,
                 small_t_init=False, small_q_init=False):

        super(ExpectationModelGaussianWithQ, self).__init__(
            input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes, encoder_strides,
            encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, weight_decay, transition_gamma,
            optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm, target_network=target_network,
            oracle_r=oracle_r, oracle_t=oracle_t, propagate_next_state=propagate_next_state, z_transform=z_transform,
            abs_z_transform=abs_z_transform, sigsoftmax=sigsoftmax, encoder_tau=encoder_tau, model_tau=model_tau,
            no_tau_target_encoder=no_tau_target_encoder, kl_penalty=kl_penalty, kl_penalty_weight=kl_penalty_weight,
            small_r_init=small_r_init, small_t_init=small_t_init
        )

        self.learning_rate_q = learning_rate_q
        self.small_q_init = small_q_init
        self.reward_gamma = reward_gamma

    def validate(self, depths, hand_states, actions, rewards, q_values, next_depths, next_hand_states, dones,
                 batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2, l3 = self.session.run(
                [self.full_q_loss_t, self.full_transition_loss_t, self.full_reward_loss_t], feed_dict=feed_dict
            )

            losses.append(np.transpose(np.array([l1, l2, l3]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, q_values, next_depths, next_hand_states, dones,
                            batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2, l3 = self.session.run([
                self.state_softmax_t, self.full_q_loss_t, self.full_transition_loss_t, self.full_reward_loss_t],
                feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2, l3]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="q_values_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_model(self):

        self.state_mu_t, self.state_sd_t, self.state_var_t, self.state_logits_t, self.state_softmax_t = \
            self.build_encoder(
                self.states_pl, self.hand_states_pl, self.encoder_tau, namespace=self.ENCODER_NAMESPACE
            )

        self.perplexity_t = tf.constant(2, dtype=tf.float32) ** (
            - tf.reduce_mean(
                tf.reduce_sum(
                    self.state_softmax_t * tf.log(self.state_softmax_t + 1e-7) /
                    tf.log(tf.constant(2, dtype=self.state_softmax_t.dtype)),
                    axis=1
                ),
                axis=0
            )
        )

        if self.no_tau_target_encoder:
            target_tau = 1.0
        else:
            target_tau = self.encoder_tau

        if self.target_network:
            self.next_state_mu_t, self.next_state_sd_t, self.next_state_var_t, self.next_state_logits_t, \
                self.next_state_softmax_t = \
                self.build_encoder(
                    self.next_states_pl, self.next_hand_states_pl, target_tau, share_weights=False,
                    namespace=self.TARGET_ENCODER_NAMESPACE
                )
            self.build_target_update()
        else:
            self.next_state_mu_t, self.next_state_sd_t, self.next_state_var_t, self.next_state_logits_t, \
                self.next_state_softmax_t = \
                self.build_encoder(
                    self.next_states_pl, self.next_hand_states_pl, target_tau, share_weights=True,
                    namespace=self.ENCODER_NAMESPACE
                )

        self.build_r_model()
        self.build_t_model()
        self.build_q_model()

    def build_q_model(self):

        if self.small_q_init:
            q_init = tf.random_normal_initializer(mean=0, stddev=0.1, dtype=tf.float32)
        else:
            q_init = tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)

        self.q_v = tf.get_variable(
            "q_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=q_init
        )
        self.q_t = self.q_v

    def build_training(self):

        # prep
        self.global_step = tf.train.get_or_create_global_step()
        self.gather_matrices()
        self.dones_float_t = tf.cast(self.dones_pl, tf.float32)

        # build losses
        self.build_q_loss()
        self.build_reward_loss()
        self.build_transition_loss()
        self.build_regularization_loss()
        self.build_kl_penalty()

        # build full loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False)

        self.loss_t = self.q_loss_t + self.reward_gamma * self.reward_loss_t + \
            tf.stop_gradient(self.gamma_v) * self.transition_loss_t + \
            self.regularization_loss_t + self.kl_penalty_weight_v * self.kl_penalty_t

        # build training
        self.build_encoder_training()
        self.build_model_training()
        self.build_q_model_training()

        if self.model_train_step is None:
            self.model_train_step = self.q_step
        else:
            self.model_train_step = tf.group(self.model_train_step, self.q_step)

        # integrate training into a single op
        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def build_q_loss(self):

        term1 = tf.square(self.q_values_pl[:, :, tf.newaxis] - self.q_t[tf.newaxis, :, :])
        term1 = tf.reduce_sum(term1, axis=1)
        term2 = term1 * self.state_softmax_t

        self.full_q_loss_t = (1 / 2) * tf.reduce_sum(term2, axis=1)
        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)

    def build_q_model_training(self):

        q_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_q)
        self.q_step = q_optimizer.minimize(
            self.q_loss_t, var_list=[self.q_v]
        )


class ExpectationModelContinuous:

    ENCODER_NAMESPACE = "encoder"
    TARGET_ENCODER_NAMESPACE = "target_encoder"

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, batch_norm=True,
                 target_network=True, propagate_next_state=False, no_sample=False, softplus=False,
                 beta=0.0, zero_embedding_variance=False, old_bn_settings=False, bn_momentum=0.99):

        if propagate_next_state:
            assert not target_network

        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_r = learning_rate_r
        self.learning_rate_t = learning_rate_t
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.batch_norm = batch_norm
        self.target_network = target_network
        self.propagate_next_state = propagate_next_state
        self.no_sample = no_sample
        self.softplus = softplus
        self.beta = beta
        self.zero_embedding_variance = zero_embedding_variance
        self.old_bn_settings = old_bn_settings
        self.bn_momentum = bn_momentum

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, depths, hand_states, batch_size=100, zero_sd=False):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }
            if zero_sd:
                feed_dict[self.state_sd_t] = np.zeros(
                    (len(depths[batch_slice]), self.num_blocks), dtype=np.float32
                )

            embedding = self.session.run(self.state_mu_t, feed_dict=feed_dict)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def predict_next_states(self, depths, hand_states, actions, batch_size=100, zero_sd=False):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        next_states = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.actions_pl: actions[batch_slice],
                self.is_training_pl: False
            }
            if zero_sd:
                feed_dict[self.state_sd_t] = np.zeros(
                    (len(depths[batch_slice]), self.num_blocks), dtype=np.float32
                )

            tmp_next_states = self.session.run(self.transformed_logits, feed_dict=feed_dict)
            next_states.append(tmp_next_states)

        next_states = np.concatenate(next_states, axis=0)
        return next_states

    def predict_rewards(self, depths, hand_states, actions, batch_size=100, zero_sd=False):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        rewards = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.actions_pl: actions[batch_slice],
                self.is_training_pl: False
            }
            if zero_sd:
                feed_dict[self.state_sd_t] = np.zeros(
                    (len(depths[batch_slice]), self.num_blocks), dtype=np.float32
                )

            tmp_rewards = self.session.run(self.reward_prediction_t, feed_dict=feed_dict)
            rewards.append(tmp_rewards)

        rewards = np.concatenate(rewards, axis=0)
        return rewards

    def validate(self, depths, hand_states, actions, rewards, next_depths, next_hand_states, dones, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2 = self.session.run([self.full_transition_loss_t, self.full_reward_loss_t], feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, next_depths, next_hand_states, dones,
                            batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2 = self.session.run([
                self.state_mu_t, self.full_transition_loss_t, self.full_reward_loss_t], feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

        self.saver = tf.train.Saver()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_model(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl)

        if self.target_network:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True
            )

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)

        # gather appropriate transition matrices
        self.gather_r_t = tf.gather(self.r_v, self.actions_pl)
        self.gather_t_t = tf.gather(self.t_v, self.actions_pl)

        # build losses
        self.build_reward_loss()
        self.build_transition_loss()
        self.build_weight_decay_loss()
        self.build_kl_loss()

        # build the whole loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False)

        self.loss_t = self.reward_loss_t + tf.stop_gradient(self.gamma_v) * self.transition_loss_t + \
            self.regularization_loss_t + self.beta * self.kl_loss_t

        # build training
        self.build_encoder_training()
        self.build_r_model_training()
        self.build_t_model_training()

        self.model_train_step = tf.group(self.r_step, self.t_step)
        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def build_reward_loss(self):

        self.reward_prediction_t = tf.reduce_sum(self.state_sample_t * self.gather_r_t, axis=1)

        term1 = tf.square(
            self.rewards_pl - self.reward_prediction_t
        )

        self.full_reward_loss_t = (1 / 2) * term1
        self.reward_loss_t = tf.reduce_mean(self.full_reward_loss_t, axis=0)

    def build_transition_loss(self):

        self.transformed_logits = tf.matmul(self.state_sample_t[:, tf.newaxis, :], self.gather_t_t)
        self.transformed_logits = self.transformed_logits[:, 0, :]

        if self.propagate_next_state:
            term1 = tf.reduce_sum(tf.square(self.next_state_sample_t - self.transformed_logits), axis=1)
        else:
            term1 = tf.reduce_sum(tf.square(tf.stop_gradient(self.next_state_sample_t) - self.transformed_logits),
                                  axis=1)

        self.full_transition_loss_t = (1 / 2) * term1 * (1 - self.float_dones_t)
        self.transition_loss_t = tf.reduce_sum(self.full_transition_loss_t, axis=0) / tf.reduce_max(
            [1.0, tf.reduce_sum(1 - self.float_dones_t)])

    def build_weight_decay_loss(self):

        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

    def build_kl_loss(self):

        self.kl_loss_t = 0.0
        if self.beta is not None and self.beta > 0.0:
            self.kl_divergence_t = 0.5 * (
                        tf.square(self.state_mu_t) + self.state_var_t - tf.log(self.state_var_t + 1e-5) - 1.0)
            self.kl_loss_t = tf.reduce_mean(tf.reduce_sum(self.kl_divergence_t, axis=1))

    def build_encoder_training(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

    def build_r_model_training(self):

        r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)
        self.r_step = r_optimizer.minimize(
            self.reward_loss_t, var_list=[self.r_v]
        )

    def build_t_model_training(self):

        t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
        self.t_step = t_optimizer.minimize(
            self.transition_loss_t, var_list=[self.t_v]
        )

    def build_encoder(self, depth_pl, hand_state_pl, share_weights=False, namespace=ENCODER_NAMESPACE):

        if len(depth_pl.shape) == 3:
            x = tf.expand_dims(depth_pl, axis=-1)
        elif len(depth_pl.shape) != 4:
            raise ValueError("Weird depth shape?")
        else:
            x = depth_pl

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        if self.old_bn_settings:
                            x = tf.layers.batch_normalization(
                                x, training=self.is_training_pl, trainable=not share_weights, momentum=self.bn_momentum
                            )

                        else:
                            x = tf.layers.batch_normalization(
                                x, training=self.is_training_pl if not share_weights else False,
                                trainable=not share_weights, momentum=self.bn_momentum
                            )

                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                if self.old_bn_settings:
                    x = tf.layers.batch_normalization(
                        x, training=self.is_training_pl, trainable=not share_weights, momentum=self.bn_momentum
                    )

                else:
                    x = tf.layers.batch_normalization(
                        x, training=self.is_training_pl if not share_weights else False,
                        trainable=not share_weights, momentum=self.bn_momentum
                    )

                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        if self.old_bn_settings:
                            x = tf.layers.batch_normalization(
                                x, training=self.is_training_pl, trainable=not share_weights, momentum=self.bn_momentum
                            )

                        else:
                            x = tf.layers.batch_normalization(
                                x, training=self.is_training_pl if not share_weights else False,
                                trainable=not share_weights, momentum=self.bn_momentum
                            )

                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):

                mu = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                sigma = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                if self.no_sample:
                    sample = mu
                    var_t = None
                else:

                    noise = tf.random_normal(
                        shape=(tf.shape(mu)[0], self.num_blocks), mean=0, stddev=1.0
                    )

                    if self.softplus:
                        # log var is sd
                        sd = tf.nn.softplus(sigma)

                        if self.zero_embedding_variance:
                            sd = sd * 0.0

                        sd_noise_t = noise * sd
                        sample = mu + sd_noise_t
                        var_t = tf.square(sd)
                    else:
                        var_t = tf.exp(sigma)

                        if self.zero_embedding_variance:
                            var_t = var_t * 0.0

                        sd = tf.sqrt(var_t)

                        sd_noise_t = noise * sd
                        sample = mu + sd_noise_t

        return mu, var_t, sd, sample

    def build_target_update(self):

        source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ENCODER_NAMESPACE)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.TARGET_ENCODER_NAMESPACE)

        assert len(source_vars) == len(target_vars) and len(source_vars) > 0

        update_ops = []
        for source_var, target_var in zip(source_vars, target_vars):
            update_ops.append(tf.assign(target_var, source_var))

        self.target_update_op = tf.group(*update_ops)

    def build_summaries(self):

        # losses
        tf.summary.scalar("loss", self.loss_t)
        tf.summary.scalar("transition_loss", tf.reduce_mean(self.transition_loss_t))
        tf.summary.scalar("reward_loss", tf.reduce_mean(self.reward_loss_t))

        # logits and softmax
        self.summarize(self.state_mu_t, "means")
        self.summarize(self.state_var_t, "vars")
        self.summarize(self.state_sample_t, "samples")

        # matrices
        self.summarize(self.r_v, "R")
        self.summarize(self.t_v, "T")

    def set_gamma(self, value):

        self.session.run(tf.assign(self.gamma_v, value))

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()

    def load(self, path):

        self.saver.restore(self.session, path)

    def save(self, path):

        path_dir = os.path.dirname(path)

        if len(path_dir) > 0 and not os.path.isdir(path_dir):
            os.makedirs(path_dir)

        self.saver.save(self.session, path)

    def save_matrices_as_images(self, step, save_dir, ext="pdf"):

        r, p = self.session.run([self.r_v, self.t_v])

        r = np.reshape(r, (r.shape[0], -1))
        p = np.reshape(p, (p.shape[0], -1))

        r_path = os.path.join(save_dir, "r_{:d}.{}".format(step, ext))
        p_path = os.path.join(save_dir, "p_{:d}.{}".format(step, ext))

        plt.clf()
        plt.imshow(r, vmin=-0.5, vmax=1.5)
        plt.colorbar()
        plt.savefig(r_path)

        plt.clf()
        plt.imshow(p, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(p_path)

    def summarize(self, var, name):

        tf.summary.scalar(name + "_mean", tf.reduce_mean(var))
        tf.summary.scalar(name + "_min", tf.reduce_min(var))
        tf.summary.scalar(name + "_max", tf.reduce_max(var))

        tf.summary.histogram(name + "_hist", var)


class ExpectationModelVQ(ExpectationModelContinuous):

    def __init__(self, input_shape, num_embeddings, dimensionality, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 weight_decay, gamma_1, optimizer_encoder, optimizer_model,
                 max_steps, batch_norm=True, target_network=True, propagate_next_state=False, no_sample=False,
                 softplus=False, alpha=0.1, beta=0.0):

        ExpectationModelContinuous.__init__(
            self, input_shape, dimensionality, num_actions, encoder_filters, encoder_filter_sizes, encoder_strides,
            encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, weight_decay, gamma_1,
            optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm, target_network=target_network,
            propagate_next_state=propagate_next_state, no_sample=no_sample, softplus=softplus, beta=beta
        )

        self.num_embeddings = num_embeddings
        self.dimensionality = dimensionality
        self.aplha = alpha

    def build_model(self):

        self.state_mu_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl, namespace=self.ENCODER_NAMESPACE)

        self.embeds = tf.get_variable(
            "embeddings", [self.num_embeddings, self.dimensionality],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        self.state_sample_t, self.state_classes_t = self.quantize(self.state_mu_t)

        if self.target_network:
            self.next_state_mu_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_mu_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True, namespace=self.ENCODER_NAMESPACE
            )

        self.next_state_sample_t, self.next_state_classes_t = self.quantize(self.next_state_mu_t)

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.dimensionality), dtype=tf.float32)
        )

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.dimensionality, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.dimensionality), dtype=tf.float32)
        )

    def quantize(self, prediction):

        diff = prediction[:, tf.newaxis, :] - self.embeds[tf.newaxis, :, :]
        norm = tf.norm(diff, axis=2)
        classes = tf.argmin(norm, axis=1)

        return tf.gather(self.embeds, classes), classes

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)

        # gather appropriate transition matrices
        self.gather_r_t = tf.gather(self.r_v, self.actions_pl)
        self.gather_t_t = tf.gather(self.t_v, self.actions_pl)

        # build losses
        self.build_reward_loss()
        self.build_transition_loss()
        self.build_left_and_right_embedding_losses()
        self.build_weight_decay_loss()

        # build the whole loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False)

        self.main_loss_t = self.reward_loss_t + tf.stop_gradient(self.gamma_v) * self.transition_loss_t + \
            self.regularization_loss_t
        self.loss_t = self.main_loss_t + self.right_loss_t

        # build training
        self.build_encoder_and_embedding_training()
        self.build_r_model_training()
        self.build_t_model_training()

        self.model_train_step = tf.group(self.r_step, self.t_step)
        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def build_left_and_right_embedding_losses(self):

        self.left_loss_t = tf.reduce_mean(
            tf.norm(tf.stop_gradient(self.state_mu_t) - self.state_sample_t, axis=1) ** 2
        )

        self.right_loss_t = tf.reduce_mean(
            tf.norm(self.state_mu_t - tf.stop_gradient(self.state_sample_t), axis=1) ** 2
        )

    def build_reward_loss(self):

        self.reward_prediction_t = tf.reduce_sum(self.state_sample_t * self.gather_r_t, axis=1)

        term1 = tf.square(
            self.rewards_pl - self.reward_prediction_t
        )

        self.full_reward_loss_t = (1 / 2) * term1
        self.reward_loss_t = tf.reduce_mean(self.full_reward_loss_t, axis=0)

    def build_transition_loss(self):

        self.transformed_logits = tf.matmul(self.state_sample_t[:, tf.newaxis, :], self.gather_t_t)
        self.transformed_logits = self.transformed_logits[:, 0, :]

        if self.propagate_next_state:
            term1 = tf.reduce_mean(tf.square(self.next_state_sample_t - self.transformed_logits), axis=1)
        else:
            term1 = tf.reduce_mean(tf.square(tf.stop_gradient(self.next_state_sample_t) - self.transformed_logits),
                                  axis=1)

        self.full_transition_loss_t = (1 / 2) * term1 * (1 - self.float_dones_t)
        self.transition_loss_t = tf.reduce_sum(self.full_transition_loss_t, axis=0) / tf.reduce_max(
            [1.0, tf.reduce_sum(1 - self.float_dones_t)])

    def build_encoder_and_embedding_training(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        grad_z = tf.gradients(self.main_loss_t, self.state_sample_t)
        encoder_grads = [(tf.gradients(self.state_mu_t, var, grad_z)[0] + self.aplha *
                          tf.gradients(self.right_loss_t, var)[0], var) for var in encoder_variables]

        embed_grads = list(zip(tf.gradients(self.left_loss_t, self.embeds), [self.embeds]))

        self.encoder_train_step = encoder_optimizer.apply_gradients(
            encoder_grads + embed_grads
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

    def build_encoder(self, depth_pl, hand_state_pl, share_weights=False, namespace=None):

        x = tf.expand_dims(depth_pl, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):

                mu = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

        return mu


class ExpectationModelContinuousNNTransitions(ExpectationModelContinuous):

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 transition_neurons, weight_decay, gamma_1, optimizer_encoder, optimizer_model,
                 max_steps, batch_norm=True, target_network=True, propagate_next_state=False, no_sample=False,
                 softplus=False, beta=0.0):

        ExpectationModelContinuous.__init__(
            self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes, encoder_strides,
            encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, weight_decay, gamma_1,
            optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm, target_network=target_network,
            propagate_next_state=propagate_next_state, no_sample=no_sample, softplus=softplus, beta=beta
        )

        self.transition_neurons = transition_neurons

    def build_model(self):

        # TODO: throw out t_v; either accept mu_t or sample_t

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl)

        if self.target_network:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True
            )

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

    def build_transition_nn(self, embedding):

        # TODO: needs to accept actions as well

        x = embedding

        with tf.variable_scope("transition"):
            for idx, neurons in enumerate(self.transition_neurons):
                with tf.variable_scope("fc{:d}".format(idx)):

                    if idx == len(self.transition_neurons) - 1:

                        x = tf.layers.dense(
                            x, neurons, activation=None,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer()
                        )

                    else:

                        x = tf.layers.dense(
                            x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            use_bias=not self.batch_norm
                        )

                        if self.batch_norm:
                            x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                            x = tf.nn.relu(x)

        assert embedding.shape[1] == x.shape[1]

        return x


class ExpectationModelContinuousWithQ(ExpectationModelContinuous):

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 learning_rate_q, weight_decay, gamma_1, gamma_2, optimizer_encoder, optimizer_model,
                 max_steps, batch_norm=True, target_network=True, propagate_next_state=False, no_sample=False,
                 softplus=False, beta=0.0, old_bn_settings=False, bn_momentum=0.99):

        ExpectationModelContinuous.__init__(
            self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes, encoder_strides,
            encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, weight_decay, gamma_1,
            optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm, target_network=target_network,
            propagate_next_state=propagate_next_state, no_sample=no_sample, softplus=softplus, beta=beta,
            old_bn_settings=old_bn_settings, bn_momentum=bn_momentum
        )

        self.gamma_2 = gamma_2
        self.learning_rate_q = learning_rate_q

    def validate(self, depths, hand_states, actions, rewards, q_values, next_depths, next_hand_states, dones,
                 batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2, l3 = self.session.run(
                [self.full_q_loss_t, self.full_transition_loss_t, self.full_reward_loss_t], feed_dict=feed_dict
            )

            losses.append(np.transpose(np.array([l1, l2, l3]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, q_values, next_depths, next_hand_states, dones,
                            batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2, l3 = self.session.run([
                self.state_mu_t, self.full_q_loss_t, self.full_transition_loss_t, self.full_reward_loss_t],
                feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2, l3]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="q_values_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_model(self):

        self.build_encoders()
        self.build_linear_models()

    def build_encoders(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl)

        if self.target_network:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True
            )

    def build_linear_models(self):

        self.q_v = tf.get_variable(
            "q_values_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)

        # gather appropriate transition matrices
        self.gather_r_t = tf.gather(self.r_v, self.actions_pl)
        self.gather_t_t = tf.gather(self.t_v, self.actions_pl)

        # build losses
        self.build_q_loss()
        self.build_reward_loss()
        self.build_transition_loss()
        self.build_weight_decay_loss()
        self.build_kl_loss()

        # build the whole loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False)

        self.loss_t = self.q_loss_t + self.gamma_2 * self.reward_loss_t + \
                      tf.stop_gradient(self.gamma_v) * self.transition_loss_t + \
                      self.regularization_loss_t + self.beta * self.kl_loss_t

        # build training
        self.build_encoder_training()
        self.build_q_model_training()
        self.build_r_model_training()
        self.build_t_model_training()

        self.model_train_step = tf.group(self.q_step, self.r_step, self.t_step)
        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def build_q_loss(self):

        self.q_prediction_t = tf.reduce_sum(self.state_sample_t[:, tf.newaxis, :] * self.q_v[tf.newaxis, :, :], axis=2)

        term1 = tf.reduce_mean(tf.square(self.q_values_pl - self.q_prediction_t), axis=1)

        self.full_q_loss_t = (1 / 2) * term1
        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)

    def build_q_model_training(self):

        q_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_q)
        self.q_step = q_optimizer.minimize(
            self.q_loss_t, var_list=[self.q_v]
        )


class ExpectationModelVQWithQ(ExpectationModelVQ):

    def __init__(self, input_shape, num_blocks, dimensionality, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 learning_rate_q, weight_decay, gamma_1, gamma_2, optimizer_encoder, optimizer_model,
                 max_steps, batch_norm=True, target_network=True, propagate_next_state=False, no_sample=False,
                 softplus=False, alpha=0.1, beta=0.0):

        ExpectationModelVQ.__init__(
            self, input_shape, num_blocks, dimensionality, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, weight_decay,
            gamma_1, optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm,
            target_network=target_network, propagate_next_state=propagate_next_state, no_sample=no_sample,
            softplus=softplus, alpha=alpha, beta=beta
        )

        self.gamma_2 = gamma_2
        self.learning_rate_q = learning_rate_q

    def validate(self, depths, hand_states, actions, rewards, q_values, next_depths, next_hand_states, dones,
                 batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2, l3 = self.session.run(
                [self.full_q_loss_t, self.full_transition_loss_t, self.full_reward_loss_t], feed_dict=feed_dict
            )

            losses.append(np.transpose(np.array([l1, l2, l3]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, q_values, next_depths, next_hand_states, dones,
                            batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2, l3 = self.session.run([
                self.state_mu_t, self.full_q_loss_t, self.full_transition_loss_t, self.full_reward_loss_t],
                feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2, l3]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="q_values_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_model(self):

        self.state_mu_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl, namespace=self.ENCODER_NAMESPACE)

        self.embeds = tf.get_variable(
            "embeddings", [self.num_embeddings, self.dimensionality],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        self.state_sample_t, self.state_classes_t = self.quantize(self.state_mu_t)

        if self.target_network:
            self.next_state_mu_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_mu_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True, namespace=self.ENCODER_NAMESPACE
            )

        self.next_state_sample_t, self.next_state_classes_t = self.quantize(self.next_state_mu_t)

        self.q_v = tf.get_variable(
            "q_values_matrix", shape=(self.num_actions, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.dimensionality), dtype=tf.float32)
        )

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.dimensionality), dtype=tf.float32)
        )

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.dimensionality, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.dimensionality), dtype=tf.float32)
        )

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)

        # gather appropriate transition matrices
        self.gather_r_t = tf.gather(self.r_v, self.actions_pl)
        self.gather_t_t = tf.gather(self.t_v, self.actions_pl)

        # build losses
        self.build_q_loss()
        self.build_reward_loss()
        self.build_transition_loss()
        self.build_left_and_right_embedding_losses()
        self.build_weight_decay_loss()

        # build the whole loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False)

        self.main_loss_t = self.q_loss_t + self.gamma_2 * self.reward_loss_t + \
            tf.stop_gradient(self.gamma_v) * self.transition_loss_t + self.regularization_loss_t
        self.loss_t = self.main_loss_t + self.right_loss_t

        # build training
        self.build_encoder_and_embedding_training()
        self.build_q_model_training()
        self.build_r_model_training()
        self.build_t_model_training()

        self.model_train_step = tf.group(self.q_step, self.r_step, self.t_step)
        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def build_q_loss(self):

        self.q_prediction_t = tf.reduce_sum(self.state_sample_t[:, tf.newaxis, :] * self.q_v[tf.newaxis, :, :], axis=2)

        term1 = tf.reduce_mean(tf.square(self.q_values_pl - self.q_prediction_t), axis=1)

        self.full_q_loss_t = (1 / 2) * term1
        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)

    def build_q_model_training(self):

        q_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_q)
        self.q_step = q_optimizer.minimize(
            self.q_loss_t, var_list=[self.q_v]
        )


class ExpectationModelVQWithQNeural(ExpectationModelVQWithQ):

    def __init__(self, input_shape, num_blocks, dimensionality, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, q_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 learning_rate_q, weight_decay, gamma_1, gamma_2, optimizer_encoder, optimizer_model,
                 max_steps, batch_norm=True, target_network=True, propagate_next_state=False, no_sample=False,
                 softplus=False, alpha=0.1, beta=0.0):

        ExpectationModelVQWithQ.__init__(
            self, input_shape, num_blocks, dimensionality, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, learning_rate_q,
            weight_decay, gamma_1, gamma_2, optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm,
            target_network=target_network, propagate_next_state=propagate_next_state, no_sample=no_sample,
            softplus=softplus, alpha=alpha, beta=beta
        )

        self.q_neurons = q_neurons

    def build_model(self):

        self.state_mu_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl, namespace=self.ENCODER_NAMESPACE)

        self.embeds = tf.get_variable(
            "embeddings", [self.num_embeddings, self.dimensionality],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        self.state_sample_t, self.state_classes_t = self.quantize(self.state_mu_t)

        if self.target_network:
            self.next_state_mu_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_mu_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True, namespace=self.ENCODER_NAMESPACE
            )

        self.next_state_sample_t, self.next_state_classes_t = self.quantize(self.next_state_mu_t)

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.dimensionality), dtype=tf.float32)
        )

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.dimensionality, self.dimensionality), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.dimensionality), dtype=tf.float32)
        )

    def build_q_loss(self):

        self.q_prediction_t = self.build_q_model(self.state_sample_t)

        term1 = tf.reduce_mean(tf.square(self.q_values_pl - self.q_prediction_t), axis=1)

        self.full_q_loss_t = (1 / 2) * term1
        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)

    def build_q_model_training(self):

        q_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_q)
        self.q_step = q_optimizer.minimize(
            self.q_loss_t, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_model")
        )

    def build_q_model(self, embedding):

        x = embedding

        with tf.variable_scope("q_model"):
            for idx, neurons in enumerate(self.q_neurons):
                with tf.variable_scope("fc{:d}".format(idx)):

                    if idx == len(self.q_neurons) - 1:

                        x = tf.layers.dense(
                            x, neurons, activation=None,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer()
                        )

                    else:

                        x = tf.layers.dense(
                            x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            use_bias=not self.batch_norm
                        )

                        if self.batch_norm:
                            x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                            x = tf.nn.relu(x)

        return x


class ExpectationModelHierarchy:

    L1_NAMESPACE = "l1"
    L1_TARGET_NAMESPACE = "target_l1"
    L2_NAMESPACE = "l2"
    L2_TARGET_NAMESPACE = "target_l2"

    def __init__(self, input_shape, l1_num_blocks, l2_num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, l1_learning_rate_encoder, l2_learning_rate_encoder, learning_rate_r, learning_rate_t,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, l2_hiddens, batch_norm=True,
                 target_network=True, propagate_next_state=False, no_sample=False):

        if propagate_next_state:
            assert not target_network

        self.input_shape = input_shape
        self.l1_num_blocks = l1_num_blocks
        self.l2_num_blocks = l2_num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.l1_learning_rate_encoder = l1_learning_rate_encoder
        self.l2_learning_rate_encoder = l2_learning_rate_encoder
        self.learning_rate_r = learning_rate_r
        self.learning_rate_t = learning_rate_t
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.l2_hiddens = l2_hiddens
        self.batch_norm = batch_norm
        self.target_network = target_network
        self.propagate_next_state = propagate_next_state
        self.no_sample = no_sample

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, depths, hand_states, level, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []

        if level == self.L1_NAMESPACE:
            to_run = self.l1_state_mu_t
        else:
            to_run = self.l2_softmax_t

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }

            embedding = self.session.run(to_run, feed_dict=feed_dict)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def validate(self, depths, hand_states, actions, rewards, next_depths, next_hand_states, dones, level,
                 batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        if level == self.L1_NAMESPACE:
            to_run = [self.l1_full_transition_loss_t, self.l1_full_reward_loss_t]
        else:
            to_run = [self.l2_full_transition_loss_t, self.l2_full_reward_loss_t]

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2 = self.session.run(to_run, feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, next_depths, next_hand_states, dones,
                            level, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        if level == self.L1_NAMESPACE:
            to_run = [self.l1_state_mu_t, self.l1_full_transition_loss_t, self.l1_full_reward_loss_t]
        else:
            to_run = [self.l2_softmax_t, self.l2_full_transition_loss_t, self.l2_full_reward_loss_t]

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2 = self.session.run(to_run, feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_model(self):

        # level one
        self.l1_state_mu_t, self.l1_state_log_var_t, self.l1_state_sample_t = \
            self.build_l1(self.states_pl, self.hand_states_pl, namespace=self.L1_NAMESPACE)

        if self.target_network:
            self.l1_next_state_mu_t, self.l1_next_state_log_var_t, self.l1_next_state_sample_t = self.build_l1(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.L1_TARGET_NAMESPACE
            )
        else:
            self.l1_next_state_mu_t, self.l1_next_state_log_var_t, self.l1_next_state_sample_t = self.build_l1(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True, namespace=self.L1_NAMESPACE
            )

        self.l1_r_v = tf.get_variable(
            "reward_matrix_l1", shape=(self.num_actions, self.l1_num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.l1_num_blocks), dtype=tf.float32)
        )
        self.l1_r_t = self.l1_r_v

        self.l1_t_v = tf.get_variable(
            "transition_matrix_l1", shape=(self.num_actions, self.l1_num_blocks, self.l1_num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.l1_num_blocks), dtype=tf.float32)
        )
        self.l1_t_t = self.l1_t_v

        # level two
        self.l2_logits_t, self.l2_softmax_t = self.build_l2(self.l1_state_mu_t, self.L2_NAMESPACE)

        if self.target_network:
            self.l2_next_logits_t, self.l2_next_softmax_t = self.build_l2(
                self.l1_state_mu_t, namespace=self.L2_TARGET_NAMESPACE, share_weights=False
            )
        else:
            self.l2_next_logits_t, self.l2_next_softmax_t = self.build_l2(
                self.l1_state_mu_t, namespace=self.L2_NAMESPACE, share_weights=True
            )

        self.l2_r_v = tf.get_variable(
            "reward_matrix_l2", shape=(self.num_actions, self.l2_num_blocks), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )
        self.l2_r_t = self.l2_r_v

        self.l2_t_v = tf.get_variable(
            "transition_matrix_l2", shape=(self.num_actions, self.l2_num_blocks, self.l2_num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.l2_num_blocks), dtype=tf.float32)
        )
        self.l2_t_t = tf.nn.softmax(self.l2_t_v, axis=-1)

        # build target updates
        if self.target_network:
            self.build_target_update()

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()

        # gather appropriate transition matrices
        l1_r_per_action_t = tf.gather(self.l1_r_t, self.actions_pl)
        l2_r_per_action_t = tf.gather(self.l2_r_t, self.actions_pl)
        l1_t_per_action_t = tf.gather(self.l1_t_t, self.actions_pl)
        l2_t_per_action_t = tf.gather(tf.nn.log_softmax(self.l2_t_v), self.actions_pl)

        # reward cross-entropy
        dones_t = tf.cast(self.dones_pl, tf.float32)

        self.l1_full_reward_loss_t = (1 / 2) * tf.square(
            self.rewards_pl - tf.reduce_sum(self.l1_state_sample_t * l1_r_per_action_t, axis=1)
        )
        self.l1_reward_loss_t = tf.reduce_mean(self.l1_full_reward_loss_t, axis=0)

        self.l2_full_reward_loss_t = (1 / 2) * tf.reduce_sum(
            tf.square(self.rewards_pl[:, tf.newaxis] - l2_r_per_action_t) * self.l2_softmax_t, axis=1
        )
        self.l2_reward_loss_t = tf.reduce_mean(self.l2_full_reward_loss_t, axis=0)

        # transition cross-entropy
        l1_transformed_logits = tf.matmul(self.l1_state_sample_t[:, tf.newaxis, :], l1_t_per_action_t)
        l1_transformed_logits = l1_transformed_logits[:, 0, :]

        self.l1_full_transition_loss_t = (1 / 2) * tf.reduce_sum(
            tf.square(tf.stop_gradient(self.l1_next_state_sample_t) - l1_transformed_logits), axis=1
        ) * (1 - dones_t)
        self.l1_transition_loss_t = tf.reduce_sum(self.l1_full_transition_loss_t, axis=0) / tf.reduce_max([1.0, tf.reduce_sum(1 - dones_t)])

        term1 = self.l2_softmax_t[:, :, tf.newaxis] * tf.stop_gradient(self.l2_next_softmax_t[:, tf.newaxis, :]) * \
                l2_t_per_action_t

        self.l2_full_transition_loss_t = - tf.reduce_sum(term1, axis=[1, 2]) * (1 - dones_t)
        self.l2_transition_loss_t = tf.reduce_sum(self.l2_full_transition_loss_t, axis=0) / tf.reduce_max([1.0, tf.reduce_sum(1 - dones_t)])

        # regularization
        l1_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.L1_NAMESPACE)
        l2_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.L2_NAMESPACE)

        self.l1_regularization_loss_t = 0
        self.l2_regularization_loss_t = 0
        if len(l1_reg) > 0:
            self.l1_regularization_loss_t = tf.add_n(l1_reg)
        if len(l2_reg) > 0:
            self.l2_regularization_loss_t = tf.add_n(l2_reg)

        # full loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False, dtype=tf.float32)

        self.l1_loss_t = self.l1_reward_loss_t + \
                         tf.stop_gradient(self.gamma_v) * self.l1_transition_loss_t + \
                         self.l1_regularization_loss_t
        self.l2_loss_t = self.l2_reward_loss_t + \
                         tf.stop_gradient(self.gamma_v) * self.l2_transition_loss_t + \
                         self.l2_regularization_loss_t

        # optimizers
        l1_encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.L1_NAMESPACE)
        l2_encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.L2_NAMESPACE)
        l1_encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.l1_learning_rate_encoder)
        l2_encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.l2_learning_rate_encoder)

        self.l1_encoder_train_step = l1_encoder_optimizer.minimize(
            self.l1_loss_t, global_step=self.global_step, var_list=l1_encoder_variables
        )
        self.l2_encoder_train_step = l2_encoder_optimizer.minimize(
            self.l2_loss_t, global_step=self.global_step, var_list=l2_encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.l1_encoder_train_step = tf.group(self.l1_encoder_train_step, self.update_op)

        l1_r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)
        l2_r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)

        self.l1_r_step = l1_r_optimizer.minimize(
            self.l1_reward_loss_t, var_list=[self.l1_r_v]
        )
        self.l2_r_step = l2_r_optimizer.minimize(
            self.l2_reward_loss_t, var_list=[self.l2_r_v]
        )

        l1_t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
        l2_t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
        self.l1_t_step = l1_t_optimizer.minimize(
            self.l1_transition_loss_t, var_list=[self.l1_t_v]
        )
        self.l2_t_step = l2_t_optimizer.minimize(
            self.l2_transition_loss_t, var_list=[self.l2_t_v]
        )

        self.l1_model_train_step = tf.group(self.l1_r_step, self.l1_t_step)
        self.l2_model_train_step = tf.group(self.l2_r_step, self.l2_t_step)

        self.l1_train_step = tf.group(self.l1_encoder_train_step, self.l1_model_train_step)
        self.l2_train_step = tf.group(self.l2_encoder_train_step, self.l2_model_train_step)

    def build_l1(self, depth_pl, hand_state_pl, share_weights=False, namespace=L1_NAMESPACE):

        x = tf.expand_dims(depth_pl, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):

                mu = tf.layers.dense(
                    x, self.l1_num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                log_var = tf.layers.dense(
                    x, self.l1_num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                if self.no_sample:
                    sample = mu
                else:

                    noise = tf.random_normal(
                        shape=(tf.shape(mu)[0], self.l1_num_blocks), mean=0, stddev=1.0
                    )

                    var_t = tf.exp(log_var)
                    sd = tf.sqrt(var_t)

                    sd_noise_t = noise * sd
                    sample = mu + sd_noise_t

        return mu, log_var, sample

    def build_l2(self, l1_output, namespace, share_weights=False):

        x = l1_output

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx, neurons in enumerate(self.l2_hiddens):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer()
                    )

            l2_logits = tf.layers.dense(
                x, self.l2_num_blocks, activation=None,
                kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=agent_utils.get_mrsa_initializer()
            )
            l2_softmax = tf.nn.softmax(l2_logits)

        return l2_logits, l2_softmax

    def build_target_update(self):

        ops = []

        for source, target in zip([self.L1_NAMESPACE, self.L2_NAMESPACE],
                                  [self.L1_TARGET_NAMESPACE, self.L2_TARGET_NAMESPACE]):

            source_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=source)
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target)

            assert len(source_vars) == len(target_vars) and len(source_vars) > 0

            for source_var, target_var in zip(source_vars, target_vars):
                ops.append(tf.assign(target_var, source_var))

        self.target_update_op = tf.group(*ops)

    def set_gamma(self, value):

        self.session.run(tf.assign(self.gamma_v, value))

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()


class RobsTwoStepModel:

    ENCODER_NAMESPACE = "encoder"
    TARGET_ENCODER_NAMESPACE = "target_encoder"

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, batch_norm=True,
                 tau=1.0, stop_step_two_reward_gradients=False, normal_t_init=False):

        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_r = learning_rate_r
        self.learning_rate_t = learning_rate_t
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.batch_norm = batch_norm
        self.tau = tau
        self.stop_step_two_reward_gradients = stop_step_two_reward_gradients
        self.normal_t_init = normal_t_init

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, depths, hand_states, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }

            embedding = self.session.run(self.state_softmax_t, feed_dict=feed_dict)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def validate(self, depths, hand_states, actions, rewards, next_actions, next_rewards, dones, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_actions_pl: next_actions[batch_slice],
                self.next_rewards_pl: next_rewards[batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2 = self.session.run(
                [self.full_step_one_reward_loss_t, self.full_step_two_reward_loss_t], feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, next_actions, next_rewards, dones,
                            batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_actions_pl: next_actions[batch_slice],
                self.next_rewards_pl: next_rewards[batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2 = self.session.run([
                self.state_softmax_t, self.full_step_one_reward_loss_t, self.full_step_two_reward_loss_t],
                feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_actions_pl = tf.placeholder(tf.int32, shape=(None,), name="next_actions_pl")
        self.next_rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="next_rewards_pl")

        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

    def build_model(self):

        self.state_logits_t, self.state_softmax_t = self.build_encoder(self.states_pl, self.hand_states_pl)

        self.perplexity_t = tf.constant(2, dtype=tf.float32) ** (
            - tf.reduce_mean(
                tf.reduce_sum(
                    self.state_softmax_t * tf.log(self.state_softmax_t + 1e-7) /
                    tf.log(tf.constant(2, dtype=self.state_softmax_t.dtype)),
                    axis=1
                ),
                axis=0
            )
        )

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )
        self.r_t = self.r_v

        if self.normal_t_init:
            init = tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks))
        else:
            init = tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=init
        )
        self.t_t = tf.nn.softmax(self.t_v, axis=-1)

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()

        # gather appropriate transition matrices
        r_per_action_t = tf.gather(self.r_t, self.actions_pl)
        t_per_action_t = tf.gather(self.t_t, self.actions_pl)

        next_r_per_action_t = tf.gather(self.r_t, self.next_actions_pl)

        if self.stop_step_two_reward_gradients:
            next_r_per_action_t = tf.stop_gradient(next_r_per_action_t)

        # one step reward loss
        self.full_step_one_reward_loss_t, self.step_one_reward_loss_t = \
            self.build_reward_loss(r_per_action_t, self.rewards_pl, self.state_softmax_t)

        # two step reward loss
        self.transformed_next_state_softmax_t = \
            tf.matmul(self.state_softmax_t[:, tf.newaxis, :], t_per_action_t)[:, 0, :]

        self.full_step_two_reward_loss_t, self.step_two_reward_loss_t = \
            self.build_reward_loss(
                next_r_per_action_t, self.next_rewards_pl, self.transformed_next_state_softmax_t, dones=self.dones_pl
            )

        # regularization
        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

        # full loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False, dtype=tf.float32)

        self.loss_t = self.step_one_reward_loss_t + tf.stop_gradient(self.gamma_v) * self.step_two_reward_loss_t + \
                      self.regularization_loss_t

        # optimizers
        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

        model_train_step = []

        r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)
        self.r_step = r_optimizer.minimize(
            self.loss_t, var_list=[self.r_v]
        )
        model_train_step.append(self.r_step)

        t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
        self.t_step = t_optimizer.minimize(
            self.loss_t, var_list=[self.t_v]
        )
        model_train_step.append(self.t_step)

        self.model_train_step = tf.group(*model_train_step)

        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    @staticmethod
    def build_reward_loss(expected_rewards, observed_rewards, encodings, dones=None):

        # compute the reward loss
        term1_t = tf.square(observed_rewards[:, tf.newaxis] - expected_rewards)
        term2_t = term1_t * encodings

        full_reward_loss_t = (1 / 2) * tf.reduce_sum(term2_t, axis=1)

        if dones is not None:
            dones = (1.0 - tf.cast(dones, dtype=tf.float32))
            full_reward_loss_t = full_reward_loss_t * dones
            reward_loss_t = tf.reduce_sum(full_reward_loss_t) / tf.reduce_max([tf.reduce_sum(dones), 1.0])
        else:
            reward_loss_t = tf.reduce_mean(full_reward_loss_t)

        return full_reward_loss_t, reward_loss_t

    def build_encoder(self, depth_pl, hand_state_pl, share_weights=False, namespace=ENCODER_NAMESPACE):

        x = tf.expand_dims(depth_pl, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):
                x = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

        dist = tf.nn.softmax(x / self.tau)

        return x, dist

    def build_summaries(self):

        # losses
        tf.summary.scalar("loss", self.loss_t)
        tf.summary.scalar("step_one_reward_loss", tf.reduce_mean(self.step_one_reward_loss_t))
        tf.summary.scalar("step_two_reward_loss", tf.reduce_mean(self.step_two_reward_loss_t))

        # logits and softmax
        agent_utils.summarize(self.state_logits_t, "logits")
        agent_utils.summarize(self.state_softmax_t, "softmax")

        # grad norms
        self.build_gradient_norm_summary(self.loss_t, self.state_logits_t, "state_logits_grad_norm")
        self.build_gradient_norm_summary(self.loss_t, self.state_softmax_t, "state_softmax_grad_norm")
        self.build_gradient_norm_summary(self.loss_t, self.r_v, "r_grad_norm", matrix_grad=True)
        self.build_gradient_norm_summary(self.loss_t, self.t_v, "t_grad_norm", matrix_grad=True)

    def build_gradient_norm_summary(self, y, x, name, matrix_grad=False):

        # TODO: different dimensions of r_t and t_t
        grad_t = tf.gradients(y, x)

        if matrix_grad:
            grad_t = grad_t[0]

        norm_t = tf.norm(grad_t, ord=2, axis=-1)[0]
        agent_utils.summarize(norm_t, name)

    def set_gamma(self, value):

        self.session.run(tf.assign(self.gamma_v, value))

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()

    def save_matrices_as_images(self, step, save_dir, ext="pdf"):

        r, p = self.session.run([self.r_t, self.t_t])

        r = np.reshape(r, (r.shape[0], -1))
        p = np.reshape(p, (p.shape[0], -1))

        r_path = os.path.join(save_dir, "r_{:d}.{}".format(step, ext))
        p_path = os.path.join(save_dir, "p_{:d}.{}".format(step, ext))

        plt.clf()
        plt.imshow(r, vmin=-0.5, vmax=1.5)
        plt.colorbar()
        plt.savefig(r_path)

        plt.clf()
        plt.imshow(p, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(p_path)


class RobsTwoStepModelContinuous:

    ENCODER_NAMESPACE = "encoder"
    TARGET_ENCODER_NAMESPACE = "target_encoder"

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, batch_norm=True,
                 stop_step_two_reward_gradients=False, model_init_std=0.1, no_sample=False):

        self.input_shape = input_shape
        self.num_blocks = num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_r = learning_rate_r
        self.learning_rate_t = learning_rate_t
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.batch_norm = batch_norm
        self.stop_step_two_reward_gradients = stop_step_two_reward_gradients
        self.model_init_std = model_init_std
        self.no_sample = no_sample

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, depths, hand_states, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }

            # TODO: I could sample
            embedding = self.session.run(self.mu_t, feed_dict=feed_dict)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def validate(self, depths, hand_states, actions, rewards, next_actions, next_rewards, dones, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_actions_pl: next_actions[batch_slice],
                self.next_rewards_pl: next_rewards[batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2 = self.session.run(
                [self.full_step_one_reward_loss_t, self.full_step_two_reward_loss_t], feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, next_actions, next_rewards, dones,
                            batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_actions_pl: next_actions[batch_slice],
                self.next_rewards_pl: next_rewards[batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings, l1, l2 = self.session.run([
                self.mu_t, self.full_step_one_reward_loss_t, self.full_step_two_reward_loss_t],
                feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_actions_pl = tf.placeholder(tf.int32, shape=(None,), name="next_actions_pl")
        self.next_rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="next_rewards_pl")

        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

    def build_model(self):

        self.mu_t, self.log_var_t, self.sample_t = self.build_encoder(self.states_pl, self.hand_states_pl)

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )
        self.r_t = self.r_v

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )
        self.t_t = self.t_v
        #self.t_t = tf.nn.softmax(self.t_v, axis=-1)

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()

        # gather appropriate transition matrices
        r_per_action_t = tf.gather(self.r_t, self.actions_pl)
        t_per_action_t = tf.gather(self.t_t, self.actions_pl)

        next_r_per_action_t = tf.gather(self.r_t, self.next_actions_pl)

        if self.stop_step_two_reward_gradients:
            next_r_per_action_t = tf.stop_gradient(next_r_per_action_t)

        # one step reward loss
        self.full_step_one_reward_loss_t, self.step_one_reward_loss_t = \
            self.build_reward_loss(r_per_action_t, self.rewards_pl, self.sample_t)

        # two step reward loss
        self.transformed_next_state_sample_t = \
            tf.matmul(self.sample_t[:, tf.newaxis, :], t_per_action_t)[:, 0, :]

        self.full_step_two_reward_loss_t, self.step_two_reward_loss_t = \
            self.build_reward_loss(
                next_r_per_action_t, self.next_rewards_pl, self.transformed_next_state_sample_t, dones=self.dones_pl
            )

        # regularization
        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

        # full loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False, dtype=tf.float32)

        self.loss_t = self.step_one_reward_loss_t + tf.stop_gradient(self.gamma_v) * self.step_two_reward_loss_t + \
                      self.regularization_loss_t

        # optimizers
        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

        model_train_step = []

        r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)
        self.r_step = r_optimizer.minimize(
            self.loss_t, var_list=[self.r_v]
        )
        model_train_step.append(self.r_step)

        t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
        self.t_step = t_optimizer.minimize(
            self.loss_t, var_list=[self.t_v]
        )
        model_train_step.append(self.t_step)

        self.model_train_step = tf.group(*model_train_step)

        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    @staticmethod
    def build_reward_loss(expected_rewards, observed_rewards, encodings, dones=None):

        # compute the reward loss
        term1_t = tf.square(
            observed_rewards - tf.reduce_sum(encodings * expected_rewards, axis=1)
        )

        full_reward_loss_t = (1 / 2) * term1_t

        if dones is not None:
            dones = (1.0 - tf.cast(dones, dtype=tf.float32))
            full_reward_loss_t = full_reward_loss_t * dones
            reward_loss_t = tf.reduce_sum(full_reward_loss_t) / tf.reduce_max([tf.reduce_sum(dones), 1.0])
        else:
            reward_loss_t = tf.reduce_mean(full_reward_loss_t)

        return full_reward_loss_t, reward_loss_t

    def build_encoder(self, depth_pl, hand_state_pl, share_weights=False, namespace=ENCODER_NAMESPACE):

        x = tf.expand_dims(depth_pl, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):

                mu = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                log_var = tf.layers.dense(
                    x, self.num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                if self.no_sample:
                    sample = mu
                else:

                    noise = tf.random_normal(
                        shape=(tf.shape(mu)[0], self.num_blocks), mean=0, stddev=1.0
                    )

                    var_t = tf.exp(log_var)
                    sd = tf.sqrt(var_t)

                    sd_noise_t = noise * sd
                    sample = mu + sd_noise_t

        return mu, log_var, sample

    def build_summaries(self):

        # losses
        tf.summary.scalar("loss", self.loss_t)
        tf.summary.scalar("step_one_reward_loss", tf.reduce_mean(self.step_one_reward_loss_t))
        tf.summary.scalar("step_two_reward_loss", tf.reduce_mean(self.step_two_reward_loss_t))

        # means and variances
        agent_utils.summarize(self.mu_t, "means")
        agent_utils.summarize(self.log_var_t, "log_variances")

        # grad norms
        self.build_gradient_norm_summary(self.loss_t, self.mu_t, "means_grad_norm")
        self.build_gradient_norm_summary(self.loss_t, self.log_var_t, "log_variances_grad_norm")
        self.build_gradient_norm_summary(self.loss_t, self.r_v, "r_grad_norm", matrix_grad=True)
        self.build_gradient_norm_summary(self.loss_t, self.t_v, "t_grad_norm", matrix_grad=True)

    def build_gradient_norm_summary(self, y, x, name, matrix_grad=False):

        # TODO: different dimensions of r_t and t_t
        grad_t = tf.gradients(y, x)

        if matrix_grad:
            grad_t = grad_t[0]

        norm_t = tf.norm(grad_t, ord=2, axis=-1)[0]
        agent_utils.summarize(norm_t, name)

    def set_gamma(self, value):

        self.session.run(tf.assign(self.gamma_v, value))

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()

    def save_matrices_as_images(self, step, save_dir, ext="pdf"):

        r, p = self.session.run([self.r_t, self.t_t])

        r = np.reshape(r, (r.shape[0], -1))
        p = np.reshape(p, (p.shape[0], -1))

        r_path = os.path.join(save_dir, "r_{:d}.{}".format(step, ext))
        p_path = os.path.join(save_dir, "p_{:d}.{}".format(step, ext))

        plt.clf()
        plt.imshow(r, vmin=-0.5, vmax=1.5)
        plt.colorbar()
        plt.savefig(r_path)

        plt.clf()
        plt.imshow(p, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(p_path)


class RobsTwoStepModelHierarchy:

    L1_NAMESPACE = "l1"
    L2_NAMESPACE = "l2"

    def __init__(self, input_shape, l1_num_blocks, l2_num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, l1_learning_rate_encoder, l2_learning_rate_encoder,
                 learning_rate_r, learning_rate_t,
                 weight_decay, gamma, optimizer_encoder, optimizer_model, max_steps, l2_hiddens, batch_norm=True,
                 stop_step_two_reward_gradients=False, model_init_std=0.1, no_sample=False):

        self.input_shape = input_shape
        self.l1_num_blocks = l1_num_blocks
        self.l2_num_blocks = l2_num_blocks
        self.num_actions = num_actions
        self.encoder_filters = encoder_filters
        self.encoder_filter_sizes = encoder_filter_sizes
        self.encoder_strides = encoder_strides
        self.encoder_neurons = encoder_neurons
        self.l1_learning_rate_encoder = l1_learning_rate_encoder
        self.l2_learning_rate_encoder = l2_learning_rate_encoder
        self.learning_rate_r = learning_rate_r
        self.learning_rate_t = learning_rate_t
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_model = optimizer_model
        self.max_steps = max_steps
        self.l2_hiddens = l2_hiddens
        self.batch_norm = batch_norm
        self.stop_step_two_reward_gradients = stop_step_two_reward_gradients
        self.model_init_std = model_init_std
        self.no_sample = no_sample

        self.hand_states_pl, self.next_hand_states_pl = None, None

    def encode(self, depths, hand_states, level, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }

            if level == self.L1_NAMESPACE:
                to_run = self.l1_mu_t
            else:
                to_run = self.l2_softmax_t

            embedding = self.session.run(to_run, feed_dict=feed_dict)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def validate(self, depths, hand_states, actions, rewards, next_actions, next_rewards, dones, level, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_actions_pl: next_actions[batch_slice],
                self.next_rewards_pl: next_rewards[batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            if level == self.L1_NAMESPACE:
                to_run = [self.l1_full_step_one_reward_loss_t, self.l1_full_step_two_reward_loss_t]
            else:
                to_run = [self.l2_full_step_one_reward_loss_t, self.l2_full_step_two_reward_loss_t]

            l1, l2 = self.session.run(to_run, feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def validate_and_encode(self, depths, hand_states, actions, rewards, next_actions, next_rewards, dones,
                            level, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_actions_pl: next_actions[batch_slice],
                self.next_rewards_pl: next_rewards[batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            if level == self.L1_NAMESPACE:
                to_run = [self.l1_mu_t, self.l1_full_step_one_reward_loss_t, self.l1_full_step_two_reward_loss_t]
            else:
                to_run = [self.l2_softmax_t, self.l2_full_step_one_reward_loss_t, self.l2_full_step_two_reward_loss_t]

            tmp_embeddings, l1, l2 = self.session.run(to_run, feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))
            embeddings.append(tmp_embeddings)

        losses = np.concatenate(losses, axis=0)
        embeddings = np.concatenate(embeddings)

        return losses, embeddings

    def build(self):

        self.build_placeholders()
        self.build_model()
        self.build_training()

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_actions_pl = tf.placeholder(tf.int32, shape=(None,), name="next_actions_pl")
        self.next_rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="next_rewards_pl")

        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

    def build_model(self):

        # build level one model
        self.l1_mu_t, self.l1_log_var_t, self.l1_sample_t = self.build_l1(self.states_pl, self.hand_states_pl)

        self.l1_r_v = tf.get_variable(
            "reward_matrix_l1", shape=(self.num_actions, self.l1_num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.l1_num_blocks), dtype=tf.float32)
        )
        self.l1_r_t = self.l1_r_v

        self.l1_t_v = tf.get_variable(
            "transition_matrix_l1", shape=(self.num_actions, self.l1_num_blocks, self.l1_num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.l1_num_blocks), dtype=tf.float32)
        )
        self.l1_t_t = self.l1_t_v

        # build level two model
        self.l2_logits_t, self.l2_softmax_t = self.build_l2(self.l1_mu_t)

        self.l2_r_v = tf.get_variable(
            "reward_matrix_l2", shape=(self.num_actions, self.l2_num_blocks), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )
        self.l2_r_t = self.l2_r_v

        self.l2_t_v = tf.get_variable(
            "transition_matrix_l2", shape=(self.num_actions, self.l2_num_blocks, self.l2_num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.l2_num_blocks), dtype=tf.float32)
        )
        self.l2_t_t = tf.nn.softmax(self.l2_t_v, axis=-1)

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()

        # gather appropriate transition matrices
        l1_r_per_action_t = tf.gather(self.l1_r_t, self.actions_pl)
        l2_r_per_action_t = tf.gather(self.l2_r_t, self.actions_pl)
        l1_t_per_action_t = tf.gather(self.l1_t_t, self.actions_pl)
        l2_t_per_action_t = tf.gather(self.l2_t_t, self.actions_pl)

        next_l1_r_per_action_t = tf.gather(self.l1_r_t, self.next_actions_pl)
        next_l2_r_per_action_t = tf.gather(self.l2_r_t, self.next_actions_pl)

        if self.stop_step_two_reward_gradients:
            next_l1_r_per_action_t = tf.stop_gradient(next_l1_r_per_action_t)
            next_l2_r_per_action_t = tf.stop_gradient(next_l2_r_per_action_t)

        # one step reward loss
        self.l1_full_step_one_reward_loss_t, self.l1_step_one_reward_loss_t = \
            self.build_reward_loss_continuous(l1_r_per_action_t, self.rewards_pl, self.l1_sample_t)

        self.l2_full_step_one_reward_loss_t, self.l2_step_one_reward_loss_t = \
            self.build_reward_loss_discrete(l2_r_per_action_t, self.rewards_pl, self.l2_softmax_t)

        # two step reward loss
        self.transformed_l1_next_state_sample_t = \
            tf.matmul(self.l1_sample_t[:, tf.newaxis, :], l1_t_per_action_t)[:, 0, :]

        self.transformed_l2_next_state_sample_t = \
            tf.matmul(self.l2_softmax_t[:, tf.newaxis, :], l2_t_per_action_t)[:, 0, :]

        self.l1_full_step_two_reward_loss_t, self.l1_step_two_reward_loss_t = \
            self.build_reward_loss_continuous(
                next_l1_r_per_action_t, self.next_rewards_pl, self.transformed_l1_next_state_sample_t,
                dones=self.dones_pl
            )

        self.l2_full_step_two_reward_loss_t, self.l2_step_two_reward_loss_t = \
            self.build_reward_loss_discrete(
                next_l2_r_per_action_t, self.next_rewards_pl, self.transformed_l2_next_state_sample_t,
                dones=self.dones_pl
            )

        # regularization
        l1_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.L1_NAMESPACE)
        l2_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.L2_NAMESPACE)

        self.l1_regularization_loss_t = 0
        self.l2_regularization_loss_t = 0
        if len(l1_reg) > 0:
            self.l1_regularization_loss_t = tf.add_n(l1_reg)
        if len(l2_reg) > 0:
            self.l2_regularization_loss_t = tf.add_n(l2_reg)

        # full loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False, dtype=tf.float32)

        self.l1_loss_t = self.l1_step_one_reward_loss_t + \
                         tf.stop_gradient(self.gamma_v) * self.l1_step_two_reward_loss_t + \
                         self.l1_regularization_loss_t
        self.l2_loss_t = self.l2_step_one_reward_loss_t + \
                         tf.stop_gradient(self.gamma_v) * self.l2_step_two_reward_loss_t + \
                         self.l2_regularization_loss_t

        # optimizers
        l1_encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.L1_NAMESPACE)
        l2_encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.L2_NAMESPACE)
        l1_encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.l1_learning_rate_encoder)
        l2_encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.l2_learning_rate_encoder)

        self.l1_encoder_train_step = l1_encoder_optimizer.minimize(
            self.l1_loss_t, global_step=self.global_step, var_list=l1_encoder_variables
        )
        self.l2_encoder_train_step = l2_encoder_optimizer.minimize(
            self.l2_loss_t, global_step=self.global_step, var_list=l2_encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.l1_encoder_train_step = tf.group(self.l1_encoder_train_step, self.update_op)

        l1_r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)
        l2_r_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_r)

        self.l1_r_step = l1_r_optimizer.minimize(
            self.l1_loss_t, var_list=[self.l1_r_v]
        )
        self.l2_r_step = l2_r_optimizer.minimize(
            self.l2_loss_t, var_list=[self.l2_r_v]
        )

        l1_t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
        l2_t_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.learning_rate_t)
        self.l1_t_step = l1_t_optimizer.minimize(
            self.l1_loss_t, var_list=[self.l1_t_v]
        )
        self.l2_t_step = l2_t_optimizer.minimize(
            self.l2_loss_t, var_list=[self.l2_t_v]
        )

        self.l1_model_train_step = tf.group(self.l1_r_step, self.l1_t_step)
        self.l2_model_train_step = tf.group(self.l2_r_step, self.l2_t_step)

        self.l1_train_step = tf.group(self.l1_encoder_train_step, self.l1_model_train_step)
        self.l2_train_step = tf.group(self.l2_encoder_train_step, self.l2_model_train_step)

    @staticmethod
    def build_reward_loss_continuous(expected_rewards, observed_rewards, encodings, dones=None):

        # compute the reward loss
        term1_t = tf.square(
            observed_rewards - tf.reduce_sum(encodings * expected_rewards, axis=1)
        )

        full_reward_loss_t = (1 / 2) * term1_t

        if dones is not None:
            dones = (1.0 - tf.cast(dones, dtype=tf.float32))
            full_reward_loss_t = full_reward_loss_t * dones
            reward_loss_t = tf.reduce_sum(full_reward_loss_t) / tf.reduce_max([tf.reduce_sum(dones), 1.0])
        else:
            reward_loss_t = tf.reduce_mean(full_reward_loss_t)

        return full_reward_loss_t, reward_loss_t

    @staticmethod
    def build_reward_loss_discrete(expected_rewards, observed_rewards, encodings, dones=None):

        # compute the reward loss
        term1_t = tf.square(observed_rewards[:, tf.newaxis] - expected_rewards)
        term2_t = term1_t * encodings

        full_reward_loss_t = (1 / 2) * tf.reduce_sum(term2_t, axis=1)

        if dones is not None:
            dones = (1.0 - tf.cast(dones, dtype=tf.float32))
            full_reward_loss_t = full_reward_loss_t * dones
            reward_loss_t = tf.reduce_sum(full_reward_loss_t) / tf.reduce_max([tf.reduce_sum(dones), 1.0])
        else:
            reward_loss_t = tf.reduce_mean(full_reward_loss_t)

        return full_reward_loss_t, reward_loss_t

    def build_l1(self, depth_pl, hand_state_pl, share_weights=False, namespace=L1_NAMESPACE):

        x = tf.expand_dims(depth_pl, axis=-1)

        with tf.variable_scope(namespace, reuse=share_weights):

            for idx in range(len(self.encoder_filters)):
                with tf.variable_scope("conv{:d}".format(idx + 1)):
                    x = tf.layers.conv2d(
                        x, self.encoder_filters[idx], self.encoder_filter_sizes[idx], self.encoder_strides[idx],
                        padding="SAME", activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm and idx != len(self.encoder_filters) - 1:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            x = tf.layers.flatten(x)

            if self.batch_norm:
                x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                x = tf.nn.relu(x)

            x = tf.concat([x, hand_state_pl], axis=1)

            for idx, neurons in enumerate(self.encoder_neurons):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer(),
                        use_bias=not self.batch_norm
                    )

                    if self.batch_norm:
                        x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                        x = tf.nn.relu(x)

            with tf.variable_scope("predict"):

                mu = tf.layers.dense(
                    x, self.l1_num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                log_var = tf.layers.dense(
                    x, self.l1_num_blocks, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

                if self.no_sample:
                    sample = mu
                else:

                    noise = tf.random_normal(
                        shape=(tf.shape(mu)[0], self.l1_num_blocks), mean=0, stddev=1.0
                    )

                    var_t = tf.exp(log_var)
                    sd = tf.sqrt(var_t)

                    sd_noise_t = noise * sd
                    sample = mu + sd_noise_t

        return mu, log_var, sample

    def build_l2(self, l1_output):

        x = l1_output

        with tf.variable_scope(self.L2_NAMESPACE):

            for idx, neurons in enumerate(self.l2_hiddens):
                with tf.variable_scope("fc{:d}".format(idx + 1)):
                    x = tf.layers.dense(
                        x, neurons, activation=tf.nn.relu,
                        kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                        kernel_initializer=agent_utils.get_mrsa_initializer()
                    )

            l2_logits = tf.layers.dense(
                x, self.l2_num_blocks, activation=None,
                kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                kernel_initializer=agent_utils.get_mrsa_initializer()
            )
            l2_softmax = tf.nn.softmax(l2_logits)

        return l2_logits, l2_softmax

    def set_gamma(self, value):

        self.session.run(tf.assign(self.gamma_v, value))

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()


class ExpectationModelContinuousVampPrior(ExpectationModelContinuous):

    def __init__(self, input_shape, num_blocks, num_actions, num_pseudo_inputs, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t,
                 pseudo_inputs_learning_rate, weight_decay, beta1, beta2, gamma, optimizer_encoder, optimizer_model,
                 max_steps, batch_norm=True, target_network=True, propagate_next_state=False, no_sample=False,
                 softplus=False):

        super(ExpectationModelContinuousVampPrior, self).__init__(
            input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes, encoder_strides,
            encoder_neurons, learning_rate_encoder, learning_rate_r, learning_rate_t, weight_decay, gamma,
            optimizer_encoder, optimizer_model, max_steps, batch_norm=batch_norm, target_network=target_network,
            propagate_next_state=propagate_next_state, no_sample=no_sample, softplus=softplus, beta=0.0
        )

        self.pseudo_inputs_learning_rate = pseudo_inputs_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.num_pseudo_inputs = num_pseudo_inputs

    def validate(self, depths, hand_states, actions, rewards, next_depths, next_hand_states, dones, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2, l3, l4 = self.session.run(
                [self.full_transition_loss_t, self.full_reward_loss_t, self.full_entropy_loss_t,
                 self.full_prior_loss_t],
                feed_dict=feed_dict)

            losses.append(np.transpose(np.array([l1, l2, l3, l4]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def get_pseudo_inputs(self):

        return self.session.run([self.pseudo_depths_v, self.pseudo_hand_states_t, self.pseudo_mu_t, self.pseudo_sample_t], feed_dict={
            self.is_training_pl: False
        })

    def build_model(self):

        # encode states and next states
        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl)

        if self.target_network:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=False,
                namespace=self.TARGET_ENCODER_NAMESPACE
            )
            self.build_target_update()
        else:
            self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True
            )

        # encode pseudo inputs
        self.pseudo_depths_v = tf.get_variable(
            "pseudo_depths", shape=(self.num_pseudo_inputs, *self.input_shape),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32)
        )
        self.pseudo_hand_states_v = tf.get_variable(
            "pseudo_hand_states", shape=(self.num_pseudo_inputs, 1),
            initializer=tf.random_normal_initializer(mean=0.5, stddev=0.1, dtype=tf.float32)
        )
        self.pseudo_hand_states_t = agent_utils.hardtanh(self.pseudo_hand_states_v)

        self.pseudo_mu_t, self.pseudo_var_t, self.pseudo_sd_t, self.pseudo_sample_t = self.build_encoder(
            self.pseudo_depths_v, self.pseudo_hand_states_t, share_weights=True
        )

        self.r_v = tf.get_variable(
            "reward_matrix", shape=(self.num_actions, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_blocks, self.num_blocks), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / self.num_blocks), dtype=tf.float32)
        )

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)

        # gather appropriate transition matrices
        self.gather_r_t = tf.gather(self.r_v, self.actions_pl)
        self.gather_t_t = tf.gather(self.t_v, self.actions_pl)

        # build losses
        self.build_reward_loss()
        self.build_transition_loss()
        self.build_weight_decay_loss()
        self.build_entropy_loss()
        self.build_prior_loss()

        # build the whole loss
        self.gamma_v = tf.Variable(initial_value=self.gamma, trainable=False)

        self.loss_t = self.reward_loss_t + tf.stop_gradient(self.gamma_v) * self.transition_loss_t + \
            self.regularization_loss_t + self.beta1 * self.entropy_loss_t + self.beta2 * self.prior_loss_t

        # build training
        self.build_encoder_training()
        self.build_pseudo_inputs_train()
        self.build_r_model_training()
        self.build_t_model_training()

        self.encoder_train_step = tf.group(self.encoder_train_step, self.pseudo_input_train_step)
        self.model_train_step = tf.group(self.r_step, self.t_step)
        self.train_step = tf.group(self.encoder_train_step, self.model_train_step)

    def build_entropy_loss(self):

        self.full_entropy_loss_t = - (1 / 2) * tf.reduce_sum(tf.log(self.state_var_t), axis=1) - \
            (self.num_blocks / 2) * (1 + np.log(2 * np.pi))
        self.entropy_loss_t = tf.reduce_mean(self.full_entropy_loss_t, axis=0)

    def build_prior_loss(self):

        self.sample_probs_t = agent_utils.many_multivariate_normals_log_pdf(
            self.state_sample_t, self.pseudo_mu_t, self.pseudo_var_t, tf.log(self.pseudo_var_t)
        )
        self.sample_probs_t -= np.log(self.num_pseudo_inputs)

        d_max = tf.reduce_max(self.sample_probs_t, axis=1)
        self.pseudo_expectation_t = d_max + tf.log(
            tf.reduce_sum(tf.exp(self.sample_probs_t - d_max[:, tf.newaxis]), axis=1)
        )

        self.full_prior_loss_t = - self.pseudo_expectation_t
        self.prior_loss_t = tf.reduce_mean(self.full_prior_loss_t, axis=0)

    def build_pseudo_inputs_train(self):

        pseudo_input_optimizer = agent_utils.get_optimizer(self.optimizer_model, self.pseudo_inputs_learning_rate)
        self.pseudo_input_train_step = pseudo_input_optimizer.minimize(
            self.prior_loss_t, var_list=[self.pseudo_depths_v, self.pseudo_hand_states_v]
        )


class QPredictionModel(ExpectationModelContinuousWithQ):

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, weight_decay, optimizer_encoder,
                 max_steps, batch_norm=True, no_sample=False, softplus=False, only_one_q_value=False,
                 old_bn_settings=False, bn_momentum=0.99):

        super(QPredictionModel, self).__init__(
            input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, None, None,
            None, weight_decay, None, None, optimizer_encoder, None,
            max_steps, batch_norm=batch_norm, target_network=False, propagate_next_state=False,
            no_sample=no_sample, softplus=softplus, beta=None, old_bn_settings=old_bn_settings,
            bn_momentum=bn_momentum
        )

        self.only_one_q_value = only_one_q_value

    def validate(self, depths, hand_states, actions, q_values, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.is_training_pl: False
            }

            l1 = self.session.run([self.full_q_loss_t], feed_dict=feed_dict)

            losses.append(l1)

        losses = np.concatenate(losses, axis=0)

        return losses

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="q_values_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

    def build_model(self):

        self.build_encoders()
        self.build_predictor()

    def build_encoders(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(self.states_pl, self.hand_states_pl)

    def build_predictor(self):

        with tf.variable_scope(self.ENCODER_NAMESPACE):
            with tf.variable_scope("q_predictions"):
                self.q_prediction_t = tf.layers.dense(
                    self.state_sample_t, self.num_actions,
                    kernel_initializer=agent_utils.get_xavier_initializer(),
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay)
                )

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.num_actions, dtype=tf.float32)

        # build losses
        self.build_q_loss()
        self.build_weight_decay_loss()

        # build the whole loss
        self.loss_t = self.q_loss_t + self.regularization_loss_t

        # build training
        self.build_encoder_training()
        self.train_step = self.encoder_train_step

    def build_q_loss(self):

        if self.only_one_q_value:
            self.full_q_loss_t = (1 / 2) * tf.reduce_sum(
                tf.square(self.q_values_pl - self.q_prediction_t) * self.actions_mask_t, axis=1
            )
        else:
            self.full_q_loss_t = (1 / 2) * tf.reduce_mean(tf.square(self.q_values_pl - self.q_prediction_t), axis=1)

        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)


class QPredictionModelGMMPrior(QPredictionModel):

    def __init__(self, input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, weight_decay, optimizer_encoder,
                 max_steps, batch_norm=True, no_sample=False, softplus=False, only_one_q_value=False,
                 old_bn_settings=False):

        super(QPredictionModelGMMPrior, self).__init__(
            input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, weight_decay, optimizer_encoder,
            max_steps, batch_norm=batch_norm, no_sample=no_sample, softplus=softplus, only_one_q_value=only_one_q_value,
            old_bn_settings=old_bn_settings
        )

        self.num_components = num_components
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2

    def encode(self, depths, hand_states, batch_size=100, zero_sd=False):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []
        sample_probs = []

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }
            if zero_sd:
                feed_dict[self.state_sd_t] = np.zeros(
                    (len(depths[batch_slice]), self.num_blocks), dtype=np.float32
                )

            embedding, sample_prob = self.session.run([self.state_mu_t, self.z1_log_cond_t], feed_dict=feed_dict)
            embeddings.append(embedding)
            sample_probs.append(sample_prob)

        embeddings = np.concatenate(embeddings, axis=0)
        sample_probs = np.concatenate(sample_probs, axis=0)

        return embeddings, sample_probs

    def validate(self, depths, hand_states, actions, q_values, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.is_training_pl: False
            }

            tmp_losses = self.session.run(
                [self.full_q_loss_t, self.full_prior_log_likelihood_t, self.full_encoder_entropy_t], feed_dict=feed_dict
            )

            losses.append(np.transpose(tmp_losses))

        losses = np.concatenate(losses, axis=0)

        return losses

    def build_model(self):

        self.build_encoders()
        self.build_predictor()
        self.build_mixture_components()

    def build_mixture_components(self):

        self.mixtures_mu_v = tf.get_variable(
            "mixtures_mu", initializer=tf.random_normal_initializer(
                mean=0.0, stddev=0.1, dtype=tf.float32
            ), shape=(self.num_components, self.num_blocks)
        )
        self.mixtures_logvar_v = tf.get_variable(
            "mixtures_var", initializer=tf.random_normal_initializer(
                mean=-1.0, stddev=0.1, dtype=tf.float32
            ), shape=(self.num_components, self.num_blocks)
        )
        self.mixtures_var_t = tf.exp(self.mixtures_logvar_v)

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.num_actions, dtype=tf.float32)

        # build losses
        self.build_q_loss()
        self.build_encoder_entropy_loss()
        self.build_prior_likelihood()
        self.build_weight_decay_loss()

        # build the whole loss
        self.prior_loss_t = - self.prior_log_likelihood
        self.loss_t = self.beta0 * self.q_loss_t + self.regularization_loss_t - self.beta1 * self.encoder_entropy_t - \
            self.beta2 * self.prior_log_likelihood

        # build training
        self.build_encoder_training()
        self.build_prior_training()
        self.train_step = tf.group(self.encoder_train_step, self.prior_train_step)

    def build_encoder_entropy_loss(self):

        self.full_encoder_entropy_t = (1 / 2) * tf.reduce_sum(tf.log(self.state_var_t), axis=1) + \
            (self.num_blocks / 2) * (1 + np.log(2 * np.pi))
        self.encoder_entropy_t = tf.reduce_mean(self.full_encoder_entropy_t, axis=0)

    def build_prior_likelihood(self):

        self.z1_log_probs_t = self.many_multivariate_normals_log_pdf(
            self.state_sample_t, self.mixtures_mu_v, self.mixtures_var_t, self.mixtures_logvar_v
        )
        self.z1_log_joint_t = self.z1_log_probs_t - np.log(self.num_components)

        self.full_prior_log_likelihood_t = tf.reduce_logsumexp(self.z1_log_joint_t, axis=1)
        self.prior_log_likelihood = tf.reduce_mean(self.full_prior_log_likelihood_t, axis=0)

        self.z1_log_cond_t = self.z1_log_joint_t - self.full_prior_log_likelihood_t[:, tf.newaxis]

    def build_encoder_training(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.ENCODER_NAMESPACE))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

    def build_prior_training(self):

        mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v]
        mixture_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.prior_train_step = mixture_optimizer.minimize(
            self.prior_loss_t, global_step=self.global_step, var_list=mixture_variables
        )

    def build_summaries(self):

        # losses
        tf.summary.scalar("loss", tf.reduce_mean(self.loss_t))
        tf.summary.scalar("q loss", tf.reduce_mean(self.q_loss_t))
        tf.summary.scalar("reg loss", tf.reduce_mean(self.regularization_loss_t))
        tf.summary.scalar("entropy loss", tf.reduce_mean(- self.encoder_entropy_t))
        tf.summary.scalar("prior loss", tf.reduce_mean(- self.prior_log_likelihood))
        tf.summary.scalar("entropy loss weighted", tf.reduce_mean(- self.beta1 * self.encoder_entropy_t))
        tf.summary.scalar("prior loss weighted", tf.reduce_mean(- self.beta2 * self.prior_log_likelihood))

        # encoder outputs
        self.summarize(self.state_mu_t, "means")
        self.summarize(self.state_var_t, "vars")
        self.summarize(self.state_sample_t, "samples")

        # mixture
        self.summarize(self.mixtures_mu_v, "mixture_means")
        self.summarize(self.mixtures_var_t, "mixture_vars")

        # encoder gradients
        self.summarize(
            tf.norm(tf.gradients(self.q_loss_t, self.state_sample_t), ord=2, axis=1),
            "q_state_sample_grads"
        )
        self.summarize(
            tf.norm(tf.gradients(- self.beta1 * self.encoder_entropy_t, self.state_var_t), ord=2, axis=1),
            "entropy_state_var_grads"
        )
        self.summarize(
            tf.norm(tf.gradients(- self.beta2 * self.prior_log_likelihood, self.state_sample_t), ord=2, axis=1),
            "prior_state_sample_grads"
        )

    @staticmethod
    def many_multivariate_normals_log_pdf(x, mu, var, logvar):

        x = x[:, tf.newaxis, :]
        mu = mu[tf.newaxis, :, :]
        var = var[tf.newaxis, :, :]
        logvar = logvar[tf.newaxis, :, :]

        term1 = - (mu.shape[2].value / 2) * np.log(2 * np.pi)
        term2 = - (1 / 2) * tf.reduce_sum(logvar, axis=2)
        term3 = - (1 / 2) * tf.reduce_sum(tf.square(x - mu) / var, axis=2)

        return term1 + term2 + term3


class ClusterQPredictionModelGMMPrior(QPredictionModelGMMPrior):

    def __init__(self, input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, weight_decay, optimizer_encoder,
                 max_steps, batch_norm=True, no_sample=False, softplus=False, only_one_q_value=False,
                 mixtures_logvar_mean=-1.0):

        super(ClusterQPredictionModelGMMPrior, self).__init__(
            input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, weight_decay,
            optimizer_encoder, max_steps, batch_norm=batch_norm, no_sample=no_sample, softplus=softplus,
            only_one_q_value=only_one_q_value
        )

        self.mixtures_logvar_mean = mixtures_logvar_mean

    def build_model(self):

        self.build_encoders()
        self.build_mixture_components()

    def build_mixture_components(self):

        self.mixtures_mu_v = tf.get_variable(
            "mixtures_mu", initializer=tf.random_normal_initializer(
                mean=0.0, stddev=0.1, dtype=tf.float32
            ), shape=(self.num_components, self.num_blocks)
        )
        self.mixtures_logvar_v = tf.get_variable(
            "mixtures_var", initializer=tf.random_normal_initializer(
                mean=self.mixtures_logvar_mean, stddev=0.1, dtype=tf.float32
            ), shape=(self.num_components, self.num_blocks)
        )
        self.mixtures_var_t = tf.exp(self.mixtures_logvar_v)

        self.mixtures_qs = tf.get_variable(
            "mixtures_qs", initializer=tf.random_normal_initializer(
                mean=0.0, stddev=0.1, dtype=tf.float32
            ), shape=(self.num_components, self.num_actions)
        )

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.num_actions, dtype=tf.float32)

        # build losses
        self.build_encoder_entropy_loss()
        self.build_prior_likelihood()
        self.build_q_loss()
        self.build_weight_decay_loss()

        # build the whole loss
        self.prior_loss_t = - self.prior_log_likelihood
        self.loss_t = self.beta0 * self.q_loss_t + self.regularization_loss_t - self.beta1 * self.encoder_entropy_t - \
            self.beta2 * self.prior_log_likelihood

        # build training
        self.build_encoder_training()
        self.build_prior_training()
        self.train_step = tf.group(self.encoder_train_step, self.prior_train_step)

    def build_prior_likelihood(self):

        self.sample_probs_t = self.many_multivariate_normals_log_pdf(
            self.state_sample_t, self.mixtures_mu_v, self.mixtures_var_t, self.mixtures_logvar_v
        )
        self.sample_probs_t -= np.log(self.num_components)

        d_max = tf.reduce_max(self.sample_probs_t, axis=1)
        self.full_prior_log_likelihood_t = d_max + tf.log(
            tf.reduce_sum(tf.exp(self.sample_probs_t - d_max[:, tf.newaxis]), axis=1)
        )
        self.prior_log_likelihood = tf.reduce_mean(self.full_prior_log_likelihood_t, axis=0)

        self.sample_log_cond_t = self.sample_probs_t - self.full_prior_log_likelihood_t[:, tf.newaxis]

    def build_q_loss(self):

        # B x A
        self.q_prediction_t = tf.matmul(self.sample_log_cond_t, self.mixtures_qs)

        if self.only_one_q_value:
            self.full_q_loss_t = (1 / 2) * tf.reduce_sum(
                tf.square(self.q_values_pl - self.q_prediction_t) * self.actions_mask_t, axis=1
            )
        else:
            self.full_q_loss_t = (1 / 2) * tf.reduce_mean(tf.square(self.q_values_pl - self.q_prediction_t), axis=1)

        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)

    def build_prior_training(self):

        mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v, self.mixtures_qs]
        mixture_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.prior_train_step = mixture_optimizer.minimize(
            self.prior_loss_t, global_step=self.global_step, var_list=mixture_variables
        )


class QPredictionModelGMMPriorWithTransitions(QPredictionModelGMMPrior):

    def __init__(self, input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, beta3, weight_decay,
                 optimizer_encoder, max_steps, batch_norm=True, no_sample=False, softplus=False,
                 only_one_q_value=False, transition_hiddens=tuple()):

        super(QPredictionModelGMMPriorWithTransitions, self).__init__(
            input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, weight_decay,
            optimizer_encoder, max_steps, batch_norm=batch_norm, no_sample=no_sample, softplus=softplus,
            only_one_q_value=only_one_q_value
        )

        self.beta3 = beta3
        self.transition_hiddens = transition_hiddens

    def validate(self, depths, hand_states, actions, q_values, next_depths, next_hand_states, dones, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            l1, l2 = self.session.run([self.full_q_loss_t, self.full_transition_loss_t], feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="q_values_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_encoders(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(
                self.states_pl, self.hand_states_pl, share_weights=False, namespace=self.ENCODER_NAMESPACE
            )

        self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = \
            self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True, namespace=self.ENCODER_NAMESPACE
            )

    def build_predictor(self):

        with tf.variable_scope(self.ENCODER_NAMESPACE):
            with tf.variable_scope("q_predictions"):
                self.q_prediction_t = tf.layers.dense(
                    self.state_sample_t, self.num_actions,
                    kernel_initializer=agent_utils.get_xavier_initializer(),
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay)
                )

        with tf.variable_scope(self.ENCODER_NAMESPACE):
            with tf.variable_scope("transition_predictions"):

                x = self.state_sample_t

                for idx, neurons in enumerate(self.transition_hiddens):
                    with tf.variable_scope("fc{:d}".format(idx)):

                        x = tf.layers.dense(
                            x, neurons, activation=tf.nn.relu if not self.batch_norm else None,
                            kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                            kernel_initializer=agent_utils.get_mrsa_initializer(),
                            use_bias=not self.batch_norm
                        )

                        if self.batch_norm:
                            x = tf.layers.batch_normalization(x, training=self.is_training_pl)
                            x = tf.nn.relu(x)

                self.transition_prediction_t = tf.layers.dense(
                    x, self.num_blocks * self.num_actions,
                    kernel_initializer=agent_utils.get_xavier_initializer(),
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay)
                )
                self.transition_prediction_t = tf.reshape(
                    self.transition_prediction_t, (-1, self.num_actions, self.num_blocks)
                )

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.num_actions, dtype=tf.float32)
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)

        # build losses
        self.build_q_loss()
        self.build_encoder_entropy_loss()
        self.build_prior_likelihood()
        self.build_transition_loss()
        self.build_weight_decay_loss()

        # build the whole loss
        self.loss_t = self.beta0 * self.q_loss_t + self.regularization_loss_t - self.beta1 * self.encoder_entropy_t - \
            self.beta2 * self.prior_log_likelihood + self.beta3 * self.transition_loss_t

        # build training
        self.build_encoder_training()
        self.train_step = self.encoder_train_step

    def build_transition_loss(self):

        self.masked_transition_prediction_t = tf.reduce_sum(
            self.transition_prediction_t * self.actions_mask_t[:, :, tf.newaxis], axis=1
        )

        if self.propagate_next_state:
            term1 = tf.reduce_sum(
                tf.square(self.next_state_sample_t - self.masked_transition_prediction_t), axis=1
            )
        else:
            term1 = tf.reduce_sum(
                tf.square(tf.stop_gradient(self.next_state_sample_t) - self.masked_transition_prediction_t), axis=1
            )

        self.full_transition_loss_t = (1 / 2) * term1 * (1 - self.float_dones_t)
        self.transition_loss_t = tf.reduce_sum(self.full_transition_loss_t, axis=0) / tf.reduce_max(
            [1.0, tf.reduce_sum(1 - self.float_dones_t)])


class QPredictionModelFakeHMMPrior(QPredictionModelGMMPrior):

    def __init__(self, input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, weight_decay,
                 optimizer_encoder, max_steps, batch_norm=True, no_sample=False, softplus=False,
                 only_one_q_value=False):

        super(QPredictionModelFakeHMMPrior, self).__init__(
            input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, weight_decay,
            optimizer_encoder, max_steps, batch_norm=batch_norm, no_sample=no_sample, softplus=softplus,
            only_one_q_value=only_one_q_value
        )

    def predict_next_states(self, depths, hand_states, actions, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.is_training_pl: False
            }

            tmp_embeddings = self.session.run(self.next_state_observations, feed_dict=feed_dict)
            embeddings.append(tmp_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="q_values_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_encoders(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(
                self.states_pl, self.hand_states_pl, share_weights=False, namespace=self.ENCODER_NAMESPACE
            )

        self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = \
            self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True, namespace=self.ENCODER_NAMESPACE
            )

    def build_model(self):

        self.build_encoders()
        self.build_predictor()
        self.build_mixture_components()
        self.build_transition_matrix()

    def build_transition_matrix(self):

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_components, self.num_components), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32)
        )
        self.t_softmax_t = tf.nn.softmax(self.t_v, axis=2)
        self.t_logsoftmax_t = tf.nn.log_softmax(self.t_v, axis=2)

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.num_actions, dtype=tf.float32)
        self.gather_t_logsoftmax_t = tf.gather(self.t_logsoftmax_t, self.actions_pl)

        # build losses
        self.build_q_loss()
        self.build_encoder_entropy_loss()
        self.build_prior_likelihood()
        self.build_weight_decay_loss()

        # build the whole loss
        self.loss_t = self.beta0 * self.q_loss_t + self.regularization_loss_t - self.beta1 * self.encoder_entropy_t - \
            self.beta2 * self.prior_log_likelihood

        # build training
        self.build_encoder_training()
        self.train_step = self.encoder_train_step

    def build_prior_likelihood(self):

        self.z1_ll, self.z1_log_probs, self.d1 = self.build_sample_likelihood(self.state_sample_t)
        self.z2_ll, self.z2_log_probs, self.d2 = self.build_sample_likelihood(self.next_state_sample_t)
        self.sample_probs_t = self.z1_log_probs

        self.z1_log_cond = self.z1_log_probs - self.z1_ll[:, tf.newaxis]
        self.z2_log_cond = self.z2_log_probs - self.z2_ll[:, tf.newaxis]

        self.prior_log_likelihood = self.z1_ll[:, tf.newaxis, tf.newaxis] + self.gather_t_logsoftmax_t + \
            tf.stop_gradient(self.z1_log_cond[:, :, tf.newaxis]) + tf.stop_gradient(self.z2_log_cond[:, tf.newaxis, :])
        self.full_prior_log_likelihood = tf.reduce_logsumexp(self.prior_log_likelihood, axis=[1, 2])
        self.prior_log_likelihood = tf.reduce_mean(self.full_prior_log_likelihood, axis=0)

        self.next_cluster_prediction = tf.reduce_logsumexp(
            self.z1_log_cond[:, :, tf.newaxis] + self.gather_t_logsoftmax_t, axis=1
        )
        self.next_cluster_assignment = tf.argmax(self.next_cluster_prediction, axis=1, output_type=tf.int32)
        self.next_state_observations = self.d2.sample(tf.shape(self.next_cluster_assignment)[0])
        self.next_cluster_mask = tf.one_hot(self.next_cluster_assignment, self.num_components, dtype=tf.float32)
        self.next_state_observations = tf.reduce_sum(
            self.next_state_observations * self.next_cluster_mask[:, :, tf.newaxis], axis=1
        )

    def build_sample_likelihood(self, sample_t):

        d = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.mixtures_mu_v, scale_diag=tf.sqrt(self.mixtures_var_t)
        )
        sample_probs_t = d.log_prob(sample_t[:, tf.newaxis, :])
        sample_probs_t -= np.log(self.num_components)

        d_max = tf.reduce_max(sample_probs_t, axis=1)
        full_sample_log_likelihood_t = d_max + tf.log(
            tf.reduce_sum(tf.exp(sample_probs_t - d_max[:, tf.newaxis]), axis=1)
        )

        return full_sample_log_likelihood_t, sample_probs_t, d

    def build_encoder_training(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v, self.t_v]
        all_variables = encoder_variables + mixture_variables

        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=all_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)


class QPredictionModelHMMPrior(QPredictionModelGMMPrior):

    def __init__(self, input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2,
                 weight_decay, optimizer_encoder, max_steps, batch_norm=True, no_sample=False, softplus=False,
                 only_one_q_value=False, beta3=0.0, alpha=0.1, cluster_predict_qs=False,
                 cluster_predict_qs_weight=0.1, old_bn_settings=False, fix_prior_training=True,
                 learning_rate_steps=None, learning_rate_values=None, model_learning_rate=None):

        super(QPredictionModelHMMPrior, self).__init__(
            input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2, weight_decay,
            optimizer_encoder, max_steps, batch_norm=batch_norm, no_sample=no_sample, softplus=softplus,
            only_one_q_value=only_one_q_value, old_bn_settings=old_bn_settings
        )

        self.beta3 = beta3
        self.alpha = alpha
        self.cluster_predict_qs = cluster_predict_qs
        self.cluster_predict_qs_weight = cluster_predict_qs_weight
        self.fix_prior_training = fix_prior_training
        self.learning_rate_steps = learning_rate_steps
        self.learning_rate_values = learning_rate_values
        self.learning_rate_model = model_learning_rate

    def encode(self, depths, hand_states, batch_size=100, zero_sd=False, new_means=None):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []
        sample_probs = []

        for i in range(num_steps):

            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[batch_slice],
                self.is_training_pl: False
            }

            if zero_sd:
                feed_dict[self.state_sd_t] = np.zeros(
                    (len(depths[batch_slice]), self.num_blocks), dtype=np.float32
                )

            if new_means is not None:
                feed_dict[self.mixtures_mu_v] = new_means

            embedding, sample_prob = self.session.run([self.state_mu_t, self.z1_log_cond], feed_dict=feed_dict)
            embeddings.append(embedding)
            sample_probs.append(sample_prob)

        embeddings = np.concatenate(embeddings, axis=0)
        sample_probs = np.concatenate(sample_probs, axis=0)

        return embeddings, sample_probs

    def validate(self, depths, hand_states, actions, q_values, next_depths, next_hand_states, dones, batch_size=100):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.next_states_pl: next_depths[batch_slice],
                self.next_hand_states_pl: next_hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.q_values_pl: q_values[batch_slice],
                self.dones_pl: dones[batch_slice],
                self.is_training_pl: False
            }

            tmp_losses = self.session.run(
                [self.full_q_loss_t, self.full_prior_log_likelihood, self.full_encoder_entropy_t], feed_dict=feed_dict
            )

            losses.append(np.transpose(tmp_losses))

        losses = np.concatenate(losses, axis=0)

        return losses

    def predict_next_states(self, depths, hand_states, actions, batch_size=100, zero_sd=False, new_means=None):

        num_steps = int(np.ceil(depths.shape[0] / batch_size))
        embeddings = []
        predictions = []
        assignments = []
        assignments_hard = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: depths[batch_slice],
                self.hand_states_pl: hand_states[:, np.newaxis][batch_slice],
                self.actions_pl: actions[batch_slice],
                self.is_training_pl: False
            }
            if zero_sd:
                feed_dict[self.state_sd_t] = np.zeros(
                    (len(depths[batch_slice]), self.num_blocks), dtype=np.float32
                )

            if new_means is not None:
                feed_dict[self.mixtures_mu_v] = new_means

            tmp_embeddings, tmp_prediction, tmp_assignment, tmp_assignment_hard = self.session.run(
                [self.next_state_observations, self.next_cluster_prediction, self.next_cluster_assignment,
                 self.next_cluster_assignment_hard],
                feed_dict=feed_dict
            )
            embeddings.append(tmp_embeddings)
            predictions.append(tmp_prediction)
            assignments.append(tmp_assignment)
            assignments_hard.append(tmp_assignment_hard)

        embeddings = np.concatenate(embeddings, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        assignments = np.concatenate(assignments, axis=0)
        assignments_hard = np.concatenate(assignments_hard, axis=0)

        return embeddings, predictions, assignments, assignments_hard

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_actions), name="q_values_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_encoders(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t = \
            self.build_encoder(
                self.states_pl, self.hand_states_pl, share_weights=False, namespace=self.ENCODER_NAMESPACE
            )

        self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t = \
            self.build_encoder(
                self.next_states_pl, self.next_hand_states_pl, share_weights=True, namespace=self.ENCODER_NAMESPACE
            )

    def build_model(self):

        self.build_encoders()
        self.build_predictor()
        self.build_mixture_components()
        self.build_hmm_variables()

    def build_hmm_variables(self):

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.num_actions, self.num_components, self.num_components), dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0, dtype=tf.float32)
        )
        self.t_softmax_t = tf.nn.softmax(self.t_v, axis=2)
        self.t_logsoftmax_t = tf.nn.log_softmax(self.t_v, axis=2)

        self.prior_v = tf.get_variable(
            "prior", shape=(self.num_components,),
            initializer=tf.constant_initializer(value=0, dtype=tf.float32)
        )
        self.prior_softmax_t = tf.nn.softmax(self.prior_v, axis=0)
        self.prior_logsoftmax_t = tf.nn.log_softmax(self.prior_v, axis=0)

    def build_training(self):

        # create global training step variable
        self.global_step = tf.train.get_or_create_global_step()
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.num_actions, dtype=tf.float32)
        self.gather_t_logsoftmax_t = tf.gather(self.t_logsoftmax_t, self.actions_pl)

        # setup learning rate
        if self.learning_rate_steps is not None and self.learning_rate_values is not None:
            assert len(self.learning_rate_steps) == len(self.learning_rate_values) - 1
            self.learning_rate_steps_to_tensor()

        # build losses
        self.build_q_loss()
        self.build_encoder_entropy_loss()
        self.build_prior_likelihood()
        self.build_weight_decay_loss()

        self.beta1_v = tf.get_variable(
            "beta1", initializer=tf.constant_initializer(self.beta1, dtype=tf.float32), shape=[]
        )

        # build the whole loss
        self.loss_t = self.beta0 * self.q_loss_t + self.regularization_loss_t - \
            tf.stop_gradient(self.beta1_v) * self.encoder_entropy_t - \
            self.beta2 * self.prior_log_likelihood
        self.prior_loss_t = - self.prior_log_likelihood

        # predict q values for clusters as well
        if self.cluster_predict_qs:
            self.build_cluster_qs_predictor()
            self.loss_t += self.cluster_predict_qs_weight * self.cluster_q_loss_t

        # build training
        if self.fix_prior_training:
            self.build_encoder_training_fixed()
            self.train_step = tf.group(self.encoder_train_step, self.prior_train_step)
        else:
            self.build_encoder_training()
            self.train_step = self.encoder_train_step

    def build_cluster_qs_predictor(self):

        self.cluster_qs_v = tf.get_variable(
            "cluster_qs", shape=(self.num_components, self.num_actions),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1)
        )
        self.cluster_q_prediction_t = tf.reduce_sum(
            self.z1_log_cond[:, :, tf.newaxis] * self.cluster_qs_v[tf.newaxis, :, :], axis=1
        )

        if self.only_one_q_value:
            self.full_cluster_q_loss_t = (1 / 2) * tf.reduce_sum(
                tf.square(self.q_values_pl - self.cluster_q_prediction_t) * self.actions_mask_t, axis=1
            )
        else:
            self.full_cluster_q_loss_t = (1 / 2) * tf.reduce_mean(
                tf.square(self.q_values_pl - self.cluster_q_prediction_t), axis=1
            )

        self.cluster_q_loss_t = tf.reduce_mean(self.full_cluster_q_loss_t, axis=0)

    def build_prior_likelihood(self):

        self.z2_prior = tf.reduce_logsumexp(
            self.gather_t_logsoftmax_t + self.prior_logsoftmax_t[tf.newaxis, :, tf.newaxis], axis=1
        )

        self.z1_ll, self.z1_log_probs, self.d1 = self.build_sample_likelihood(
            self.state_sample_t, self.prior_logsoftmax_t[tf.newaxis, :]
        )
        self.z2_ll, self.z2_log_probs, self.d2 = self.build_sample_likelihood(
            self.next_state_sample_t, self.z2_prior
        )
        self.sample_probs_t = self.z1_log_probs

        self.z1_log_cond = self.z1_log_probs + self.prior_logsoftmax_t - self.z1_ll[:, tf.newaxis]
        self.z2_log_cond = self.z2_log_probs + self.z2_prior - self.z2_ll[:, tf.newaxis]

        self.build_dirichlet_prior()

        self.prior_log_likelihood = self.z1_log_probs[:, :, tf.newaxis] + \
                                    self.prior_logsoftmax_t[tf.newaxis, :, tf.newaxis] + \
                                    self.beta3 * self.dirichlet_prior_log_prob + \
            (
                tf.stop_gradient(self.z2_log_probs[:, tf.newaxis, :]) +
                self.gather_t_logsoftmax_t
            ) * (1 - self.float_dones_t)[:, tf.newaxis, tf.newaxis]
        self.full_prior_log_likelihood = tf.reduce_logsumexp(self.prior_log_likelihood, axis=[1, 2])
        self.prior_log_likelihood = tf.reduce_mean(self.full_prior_log_likelihood, axis=0)

        self.next_cluster_prediction = tf.reduce_logsumexp(
            self.z1_log_cond[:, :, tf.newaxis] + self.gather_t_logsoftmax_t, axis=1
        )
        self.next_cluster_prediction_hard = tf.reduce_logsumexp(
            tf.one_hot(
                tf.argmax(self.z1_log_cond, axis=1), self.z1_log_cond.shape[1].value, dtype=tf.float32
            )[:, :, tf.newaxis] + self.gather_t_logsoftmax_t, axis=1
        )
        self.next_cluster_assignment = tf.argmax(self.next_cluster_prediction, axis=1, output_type=tf.int32)
        self.next_cluster_assignment_hard = tf.argmax(self.next_cluster_prediction_hard, axis=1, output_type=tf.int32)
        self.next_state_observations = self.d2.sample(tf.shape(self.next_cluster_assignment)[0])
        self.next_cluster_mask = tf.one_hot(self.next_cluster_assignment, self.num_components, dtype=tf.float32)
        self.next_state_observations = tf.reduce_sum(
            self.next_state_observations * self.next_cluster_mask[:, :, tf.newaxis], axis=1
        )

    def build_dirichlet_prior(self):

        self.dirichlet_prior_d = tf.contrib.distributions.Dirichlet(
            np.ones(self.num_components, dtype=np.float32) * self.alpha, allow_nan_stats=False
        )
        self.dirichlet_prior_log_prob = self.dirichlet_prior_d.log_prob(self.prior_softmax_t)

    def build_sample_likelihood(self, sample_t, log_prior_t):

        d = tf.contrib.distributions.MultivariateNormalDiag(
            loc=self.mixtures_mu_v, scale_diag=tf.sqrt(self.mixtures_var_t)
        )
        sample_log_probs_t = d.log_prob(sample_t[:, tf.newaxis, :])
        sample_log_joint_t = sample_log_probs_t + log_prior_t

        d_max = tf.reduce_max(sample_log_joint_t, axis=1)
        full_sample_log_likelihood_t = d_max + tf.log(
            tf.reduce_sum(tf.exp(sample_log_joint_t - d_max[:, tf.newaxis]), axis=1)
        )

        return full_sample_log_likelihood_t, sample_log_probs_t, d

    def build_encoder_training(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v, self.t_v]
        all_variables = encoder_variables + mixture_variables

        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)
        hmm_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)
        T_and_prior_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)
        prior_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=all_variables
        )
        self.hmm_train_step = hmm_optimizer.minimize(
            self.loss_t, var_list=[self.t_v, self.prior_v, self.mixtures_mu_v, self.mixtures_logvar_v]
        )
        self.T_and_prior_train_step = T_and_prior_optimizer.minimize(
            self.loss_t, var_list=[self.t_v, self.prior_v]
        )
        self.prior_train_step = prior_optimizer.minimize(
            self.loss_t, var_list=[self.prior_v]
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

    def build_encoder_training_fixed(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v, self.t_v]
        all_mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v, self.t_v, self.prior_v]

        encoder_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_encoder)
        prior_optimizer = agent_utils.get_optimizer(self.optimizer_encoder, self.learning_rate_model)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )
        self.prior_train_step = prior_optimizer.minimize(
            self.prior_loss_t, global_step=self.global_step, var_list=mixture_variables
        )

        self.hmm_train_step = prior_optimizer.minimize(
            self.prior_loss_t, global_step=self.global_step, var_list=all_mixture_variables
        )

        if self.batch_norm:
            self.update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.encoder_train_step = tf.group(self.encoder_train_step, self.update_op)

    def learning_rate_steps_to_tensor(self):

        self.learning_rate_encoder = tf.train.piecewise_constant(
            self.global_step, self.learning_rate_steps, self.learning_rate_values
        )
