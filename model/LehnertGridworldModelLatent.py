import numpy as np
import tensorflow as tf
import agents.utils as agent_utils
import config_constants as cc
from model.model import Model


class LehnertGridworldModelLatent(Model):

    ENCODER_NAMESPACE = "encoder"
    NUM_ACTIONS = 4

    def __init__(self, config):

        super(LehnertGridworldModelLatent, self).__init__()

        self.height = config[cc.HEIGHT]
        self.width = config[cc.WIDTH]
        self.num_blocks = config[cc.NUM_BLOCKS]
        self.weight_decay = config[cc.WEIGHT_DECAY]
        self.optimizer = config[cc.OPTIMIZER]
        self.learning_rate = config[cc.LEARNING_RATE]
        self.beta = config[cc.BETA]
        self.one_q_value = config[cc.ONE_Q_VALUE]
        self.fc_only = config[cc.FC_ONLY]

    def validate(self, states, actions, qs, batch_size=100):

        num_steps = int(np.ceil(len(states) / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice],
                self.actions_pl: actions[batch_slice],
                self.qs_pl: qs[batch_slice]
            }

            l1, l2 = self.session.run(
                [self.full_q_loss_t, self.full_encoder_entropy_t],
                feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def predict(self, states, actions, batch_size=100):

        num_steps = int(np.ceil(len(states) / batch_size))
        qs = []
        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice],
                self.actions_pl: actions[batch_slice]
            }

            tmp_qs = self.session.run(self.masked_q_prediction_t, feed_dict=feed_dict)
            qs.append(tmp_qs)

        qs = np.concatenate(qs, axis=0)

        return qs

    def encode(self, states, batch_size=100):

        num_steps = int(np.ceil(len(states) / batch_size))
        encodings = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice]
            }

            tmp_encodings = self.session.run(self.state_sample_t, feed_dict=feed_dict)
            encodings.append(tmp_encodings)

        encodings = np.concatenate(encodings, axis=0)

        return encodings

    def build(self):

        self.build_placeholders_()
        self.build_model()
        self.build_predictors_()
        self.build_training_()

        self.saver = tf.train.Saver()

    def build_placeholders_(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, self.height, self.width), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.qs_pl = tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS), name="rewards_pl")
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.NUM_ACTIONS, dtype=tf.float32)

    def build_model(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t, self.state_noise_t = \
            self.build_encoder_(self.states_pl, self.ENCODER_NAMESPACE)

    def build_predictors_(self):

        with tf.variable_scope(self.ENCODER_NAMESPACE):

            with tf.variable_scope("predict_q"):
                self.q_prediction_t = tf.layers.dense(
                    self.state_sample_t, self.NUM_ACTIONS, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )
            self.masked_q_prediction_t = tf.reduce_sum(self.actions_mask_t * self.q_prediction_t, axis=1)

    def build_encoder_(self, x, namespace, share_weights=False):

        if self.fc_only:
            return self.build_encoder_fc_(x, namespace, share_weights=share_weights)
        else:
            return self.build_encoder_conv_(x, namespace, share_weights=share_weights)

    def build_encoder_fc_(self, x, namespace, share_weights=False):

        with tf.variable_scope(namespace, reuse=share_weights):

            x = tf.layers.flatten(x)

            with tf.variable_scope("fc1"):
                x = tf.layers.dense(
                    x, 128, activation=tf.nn.relu,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

            with tf.variable_scope("fc2"):
                x = tf.layers.dense(
                    x, 128, activation=tf.nn.relu,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

            with tf.variable_scope("latent"):
                mu = tf.layers.dense(
                    x, 32, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )
                sigma = tf.layers.dense(
                    x, 32, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )
                sd = tf.nn.softplus(sigma)
                noise = tf.random_normal(
                    shape=(tf.shape(mu)[0], 32), mean=0, stddev=1.0
                )
                sd_noise_t = noise * sd
                sample = mu + sd_noise_t
                var = tf.square(sd)

        return mu, var, sd, sample, noise

    def build_encoder_conv_(self, x, namespace, share_weights=False):
        # same network as for minatar, more or less

        with tf.variable_scope(namespace, reuse=share_weights):

            x = x[:, :, :, tf.newaxis]

            with tf.variable_scope("conv1"):
                x = tf.layers.conv2d(
                    x, 16, 3, (1, 1), padding="same", activation=tf.nn.relu,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

            x = tf.layers.flatten(x)

            with tf.variable_scope("fc1"):
                x = tf.layers.dense(
                    x, 128, activation=tf.nn.relu,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )

            with tf.variable_scope("latent"):
                mu = tf.layers.dense(
                    x, 32, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )
                sigma = tf.layers.dense(
                    x, 32, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )
                sd = tf.nn.softplus(sigma)
                noise = tf.random_normal(
                    shape=(tf.shape(mu)[0], 32), mean=0, stddev=1.0
                )
                sd_noise_t = noise * sd
                sample = mu + sd_noise_t
                var = tf.square(sd)

        return mu, var, sd, sample, noise

    def build_training_(self):

        self.global_step = tf.train.get_or_create_global_step()

        self.build_q_loss_()
        self.build_encoder_entropy_loss_()
        self.build_weight_decay_loss_()

        self.loss_t = self.q_loss_t - self.beta * self.encoder_entropy_t + self.regularization_loss_t

        self.build_model_training_()

    def build_q_loss_(self):

        if self.one_q_value:
            self.full_q_loss_t = (1 / 2) * tf.reduce_sum(
                tf.square(self.qs_pl - self.q_prediction_t) * self.actions_mask_t, axis=1
            )
        else:
            self.full_q_loss_t = (1 / 2) * tf.reduce_mean(
                tf.square(self.qs_pl - self.q_prediction_t), axis=1
            )

        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t)

    def build_encoder_entropy_loss_(self):

        self.full_encoder_entropy_t = (1 / 2) * tf.reduce_sum(tf.log(self.state_var_t), axis=1) + \
            (self.num_blocks / 2) * (1 + np.log(2 * np.pi))
        self.encoder_entropy_t = tf.reduce_mean(self.full_encoder_entropy_t, axis=0)

    def build_weight_decay_loss_(self):

        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

    def build_model_training_(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer, self.learning_rate)

        self.train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )
