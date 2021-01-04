import tensorflow as tf
from model.bisim.bisim import QPredictionModel
import agents.utils as agent_utils


class QPredictionModelMultiTask(QPredictionModel):

    def __init__(self, input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, weight_decay, optimizer_encoder,
                 max_steps, num_tasks, batch_norm=True, no_sample=False, softplus=False, only_one_q_value=False,
                 old_bn_settings=False, bn_momentum=0.99):

        super(QPredictionModelMultiTask, self).__init__(
            input_shape, num_blocks, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, weight_decay, optimizer_encoder,
            max_steps, batch_norm=batch_norm, no_sample=no_sample, softplus=softplus, only_one_q_value=only_one_q_value,
            old_bn_settings=old_bn_settings, bn_momentum=bn_momentum
        )
        self.num_tasks = num_tasks

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.q_values_pl = tf.placeholder(
            tf.float32, shape=(None, self.num_tasks, self.num_actions), name="q_values_pl"
        )
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

    def build_predictor(self):

        with tf.variable_scope(self.ENCODER_NAMESPACE):
            with tf.variable_scope("q_predictions"):

                x = self.state_sample_t

                self.q_prediction_t = tf.layers.dense(
                    x, self.num_tasks * self.num_actions,
                    kernel_initializer=agent_utils.get_xavier_initializer(),
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay)
                )
                self.q_prediction_t = tf.reshape(
                    self.q_prediction_t, shape=(tf.shape(self.q_prediction_t)[0], self.num_tasks, self.num_actions)
                )

    def build_q_loss(self):

        self.full_q_loss_t = (1 / 2) * tf.reduce_mean(tf.square(self.q_values_pl - self.q_prediction_t), axis=(1, 2))
        #self.full_q_loss_t = (1 / 2) * tf.reduce_mean(U.huber_loss(self.q_values_pl - self.q_prediction_t), axis=(1, 2))
        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)
