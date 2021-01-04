import tensorflow as tf
from model.bisim.bisim import QPredictionModelHMMPrior
import agents.utils as agent_utils


class QPredictionModelHMMPriorMultiTask(QPredictionModelHMMPrior):

    def __init__(self, input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
                 encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2,
                 weight_decay, optimizer_encoder, max_steps, num_tasks, batch_norm=True, no_sample=False, softplus=False,
                 only_one_q_value=False, beta3=0.0, alpha=0.1, cluster_predict_qs=False,
                 cluster_predict_qs_weight=0.1, old_bn_settings=False, fix_prior_training=False,
                 model_learning_rate=None):

        super(QPredictionModelHMMPriorMultiTask, self).__init__(
            input_shape, num_blocks, num_components, num_actions, encoder_filters, encoder_filter_sizes,
            encoder_strides, encoder_neurons, learning_rate_encoder, beta0, beta1, beta2,
            weight_decay, optimizer_encoder, max_steps, batch_norm=batch_norm, no_sample=no_sample, softplus=softplus,
            only_one_q_value=only_one_q_value, beta3=beta3, alpha=alpha, cluster_predict_qs=cluster_predict_qs,
            cluster_predict_qs_weight=cluster_predict_qs_weight, old_bn_settings=old_bn_settings,
            fix_prior_training=fix_prior_training, model_learning_rate=model_learning_rate
        )
        self.num_tasks = num_tasks

    def build_placeholders(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.q_values_pl = tf.placeholder(tf.float32, shape=(None, self.num_tasks, self.num_actions), name="q_values_pl")
        self.dones_pl = tf.placeholder(tf.bool, shape=(None,), name="dones_pl")
        self.float_dones_t = tf.cast(self.dones_pl, tf.float32)
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name="next_states_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        self.hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="hand_states_pl")
        self.next_hand_states_pl = tf.placeholder(tf.float32, shape=(None, 1), name="next_hand_states_pl")

    def build_predictor(self):

        with tf.variable_scope(self.ENCODER_NAMESPACE):
            with tf.variable_scope("q_predictions"):
                self.q_prediction_t = tf.layers.dense(
                    self.state_sample_t, self.num_tasks * self.num_actions,
                    kernel_initializer=agent_utils.get_xavier_initializer(),
                    kernel_regularizer=agent_utils.get_weight_regularizer(self.weight_decay)
                )
                self.q_prediction_t = tf.reshape(
                    self.q_prediction_t, shape=(tf.shape(self.q_prediction_t)[0], self.num_tasks, self.num_actions)
                )

    def build_q_loss(self):

        self.full_q_loss_t = (1 / 2) * tf.reduce_mean(tf.square(self.q_values_pl - self.q_prediction_t), axis=(1, 2))
        self.q_loss_t = tf.reduce_mean(self.full_q_loss_t, axis=0)
