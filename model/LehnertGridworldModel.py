import numpy as np
import tensorflow as tf
import agents.utils as agent_utils
import config_constants as cc
from model.model import Model


class LehnertGridworldModel(Model):

    MODEL_NAMESPACE = "model"
    NUM_ACTIONS = 4

    def __init__(self, config):

        super(LehnertGridworldModel, self).__init__()

        self.height = config[cc.HEIGHT]
        self.width = config[cc.WIDTH]
        self.weight_decay = config[cc.WEIGHT_DECAY]
        self.optimizer = config[cc.OPTIMIZER]
        self.learning_rate = config[cc.LEARNING_RATE]
        self.fc_only = config[cc.FC_ONLY]
        self.dropout_prob = config[cc.DROPOUT_PROB]

    def validate(self, states, actions, rewards, next_states, batch_size=100):

        num_steps = int(np.ceil(len(states) / batch_size))
        losses = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice],
                self.actions_pl: actions[batch_slice],
                self.rewards_pl: rewards[batch_slice],
                self.next_states_pl: next_states[batch_slice],
                self.is_training: False
            }

            l1, l2 = self.session.run([self.full_reward_loss_t, self.full_transition_loss_t], feed_dict=feed_dict)
            losses.append(np.transpose(np.array([l1, l2]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def predict(self, states, actions, batch_size=100):

        num_steps = int(np.ceil(len(states) / batch_size))
        rewards = []
        next_states = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice],
                self.actions_pl: actions[batch_slice],
                self.is_training: False
            }

            tmp_rewards, tmp_next_states = self.session.run(
                [self.reward_prediction_t, self.masked_transition_prediction_softmax_t], feed_dict=feed_dict
            )
            rewards.append(tmp_rewards)
            next_states.append(tmp_next_states)

        rewards = np.concatenate(rewards, axis=0)
        next_states = np.concatenate(next_states, axis=0)

        return rewards, next_states

    def build(self):

        self.build_placeholders_()
        self.build_model_()
        self.build_training_()

        self.saver = tf.train.Saver()

    def build_placeholders_(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, self.height, self.width), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None,), name="rewards_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, self.height, self.width), name="next_states_pl")
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.NUM_ACTIONS, dtype=tf.float32)
        self.is_training = tf.placeholder(tf.bool, name="is_training_pl")

    def build_model_(self):

        if self.fc_only:
            self.build_model_fc_()
        else:
            self.build_model_conv_()

    def build_model_fc_(self):

        with tf.variable_scope(self.MODEL_NAMESPACE):

            x = self.states_pl

            batch_size = tf.shape(x)[0]
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

                if self.dropout_prob is not None and self.dropout_prob > 0.0:
                    x = tf.layers.dropout(x, rate=self.dropout_prob, training=self.is_training)

            with tf.variable_scope("predict_reward"):
                self.reward_prediction_t = tf.layers.dense(
                    x, 1, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )[:, 0]

            with tf.variable_scope("predict_transition"):
                self.transition_prediction_t = tf.layers.dense(
                    x,  self.NUM_ACTIONS * self.height * self.width, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )
                self.transition_prediction_t = tf.reshape(
                    self.transition_prediction_t, (batch_size, self.NUM_ACTIONS, self.height, self.width)
                )
                self.masked_transition_prediction_t = tf.reduce_sum(
                    self.transition_prediction_t * self.actions_mask_t[:, :, tf.newaxis, tf.newaxis], axis=1
                )
                self.masked_transition_prediction_softmax_t = \
                    tf.reshape(tf.nn.softmax(tf.reshape(
                        self.masked_transition_prediction_t,
                        shape=[batch_size, -1]
                    ), axis=1), [batch_size, self.height, self.width])

    def build_model_conv_(self):
        # same network as for minatar, more or less

        with tf.variable_scope(self.MODEL_NAMESPACE):

            x = self.states_pl[:, :, :, tf.newaxis]
            batch_size = tf.shape(x)[0]

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

            if self.dropout_prob is not None and self.dropout_prob > 0.0:
                x = tf.layers.dropout(x, rate=self.dropout_prob, training=self.is_training)

            with tf.variable_scope("predict_reward"):
                self.reward_prediction_t = tf.layers.dense(
                    x, 1, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )[:, 0]

            with tf.variable_scope("predict_transition"):
                self.transition_prediction_t = tf.layers.dense(
                    x,  self.NUM_ACTIONS * self.height * self.width, activation=None,
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay),
                    kernel_initializer=agent_utils.get_mrsa_initializer()
                )
                self.transition_prediction_t = tf.reshape(
                    self.transition_prediction_t, (batch_size, self.NUM_ACTIONS, self.height, self.width)
                )
                self.masked_transition_prediction_t = tf.reduce_sum(
                    self.transition_prediction_t * self.actions_mask_t[:, :, tf.newaxis, tf.newaxis], axis=1
                )
                self.masked_transition_prediction_softmax_t = \
                    tf.reshape(tf.nn.softmax(tf.reshape(
                        self.masked_transition_prediction_t,
                        shape=[batch_size, -1]
                    ), axis=1), [batch_size, self.height, self.width])

    def build_training_(self):

        self.global_step = tf.train.get_or_create_global_step()

        self.build_reward_loss_()
        self.build_transition_loss_()
        self.build_weight_decay_loss_()

        self.loss_t = self.reward_loss_t + self.transition_loss_t + self.regularization_loss_t

        self.build_model_training_()

    def build_reward_loss_(self):

        self.full_reward_loss_t = (1 / 2) * tf.square(self.rewards_pl - self.reward_prediction_t)
        self.reward_loss_t = tf.reduce_mean(self.full_reward_loss_t)

    def build_transition_loss_(self):

        flat_pred = tf.reshape(
            self.masked_transition_prediction_t,
            (tf.shape(self.masked_transition_prediction_t)[0], self.height * self.width)
        )
        flat_gt = tf.reshape(
            self.next_states_pl, (tf.shape(self.next_states_pl)[0], self.height * self.width)
        )
        self.full_transition_loss_t = tf.losses.softmax_cross_entropy(
            onehot_labels=flat_gt, logits=flat_pred, reduction=tf.losses.Reduction.NONE
        )
        self.transition_loss_t = tf.reduce_mean(self.full_transition_loss_t)

    def build_weight_decay_loss_(self):

        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg) > 0:
            self.regularization_loss_t = tf.add_n(reg)
        else:
            self.regularization_loss_t = 0

    def build_model_training_(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.MODEL_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer, self.learning_rate)

        self.train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )
