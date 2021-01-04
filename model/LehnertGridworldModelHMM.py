import tensorflow as tf
import agents.utils as agent_utils
from model.LehnertGridworldModelGMM import LehnertGridworldModelGMM
import config_constants as cc


class LehnertGridworldModelHMM(LehnertGridworldModelGMM):

    ENCODER_NAMESPACE = "encoder"
    NUM_ACTIONS = 4

    def __init__(self, config):

        super(LehnertGridworldModelHMM, self).__init__(config)

        self.model_learning_rate = config[cc.MODEL_LEARNING_RATE]

    def build(self):

        self.build_placeholders_()
        self.build_model()
        self.build_predictors_()
        self.build_mixture_components_()
        self.build_hmm_variables_()
        self.build_training_()

        self.saver = tf.train.Saver()

    def build_placeholders_(self):

        self.states_pl = tf.placeholder(tf.float32, shape=(None, self.height, self.width), name="states_pl")
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, self.height, self.width), name="next_states_pl")
        self.actions_pl = tf.placeholder(tf.int32, shape=(None,), name="actions_pl")
        self.qs_pl = tf.placeholder(tf.float32, shape=(None, self.NUM_ACTIONS), name="rewards_pl")
        self.actions_mask_t = tf.one_hot(self.actions_pl, self.NUM_ACTIONS, dtype=tf.float32)

    def build_model(self):

        self.state_mu_t, self.state_var_t, self.state_sd_t, self.state_sample_t, self.state_noise_t = \
            self.build_encoder_(self.states_pl, self.ENCODER_NAMESPACE)

        self.next_state_mu_t, self.next_state_var_t, self.next_state_sd_t, self.next_state_sample_t, self.next_state_noise_t = \
            self.build_encoder_(self.next_states_pl, self.ENCODER_NAMESPACE, share_weights=True)

    def build_hmm_variables_(self):

        self.t_v = tf.get_variable(
            "transition_matrix", shape=(self.NUM_ACTIONS, self.num_components, self.num_components), dtype=tf.float32,
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

        self.gather_t_logsoftmax_t = tf.gather(self.t_logsoftmax_t, self.actions_pl)

    def build_training_(self):

        self.global_step = tf.train.get_or_create_global_step()

        self.build_q_loss_()
        self.build_prior_likelihood_()
        self.build_encoder_entropy_loss_()
        self.build_weight_decay_loss_()

        self.loss_t = self.beta0 * self.q_loss_t \
            - self.beta1 * self.encoder_entropy_t \
            - self.beta2 * self.prior_log_likelihood_t \
            + self.regularization_loss_t
        self.prior_loss_t = - self.prior_log_likelihood_t

        self.build_model_training_()
        self.build_prior_training_()

        self.train_step = tf.group(self.encoder_train_step, self.prior_train_step)

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

        self.z1_log_cond = self.z1_log_probs + self.prior_logsoftmax_t - self.z1_ll[:, tf.newaxis]
        self.z2_log_cond = self.z2_log_probs + self.z2_prior - self.z2_ll[:, tf.newaxis]

        prior_joint = self.z1_log_probs[:, :, tf.newaxis] + \
                                    self.prior_logsoftmax_t[tf.newaxis, :, tf.newaxis] + \
                                    (
                                            tf.stop_gradient(self.z2_log_probs[:, tf.newaxis, :]) +
                                            self.gather_t_logsoftmax_t
                                    )
        self.full_prior_log_likelihood_t = tf.reduce_logsumexp(prior_joint, axis=[1, 2])
        self.prior_log_likelihood_t = tf.reduce_mean(self.full_prior_log_likelihood_t, axis=0)

        self.next_cluster_prediction = tf.reduce_logsumexp(
            self.z1_log_cond[:, :, tf.newaxis] + self.gather_t_logsoftmax_t, axis=1
        )
        self.next_cluster_assignment = tf.argmax(self.next_cluster_prediction, axis=1, output_type=tf.int32)

    def build_model_training_(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer, self.learning_rate)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

    def build_prior_training_(self):

        self.mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v, self.t_v]
        mixture_optimizer = agent_utils.get_optimizer(self.optimizer, self.model_learning_rate)

        self.prior_train_step = mixture_optimizer.minimize(
            self.prior_loss_t, global_step=self.global_step, var_list=self.mixture_variables
        )

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
