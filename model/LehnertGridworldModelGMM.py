import numpy as np
import tensorflow as tf
import agents.utils as agent_utils
import config_constants as cc
from model.LehnertGridworldModelLatent import LehnertGridworldModelLatent


class LehnertGridworldModelGMM(LehnertGridworldModelLatent):

    ENCODER_NAMESPACE = "encoder"
    NUM_ACTIONS = 4

    def __init__(self, config):

        config[cc.BETA] = None

        super(LehnertGridworldModelGMM, self).__init__(config)

        self.num_components = config[cc.NUM_COMPONENTS]
        self.learning_rate = config[cc.LEARNING_RATE]
        self.beta0 = config[cc.BETA0]
        self.beta1 = config[cc.BETA1]
        self.beta2 = config[cc.BETA2]
        self.mixtures_mu_init_sd = config[cc.MIXTURES_MU_INIT_SD]
        self.mixtures_sd_init_mu = config[cc.MIXTURES_SD_INIT_MU]
        self.mixtures_sd_init_sd = config[cc.MIXTURES_SD_INIT_SD]

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

            l1, l2, l3 = self.session.run(
                [self.full_q_loss_t, self.full_encoder_entropy_t, self.full_prior_log_likelihood_t],
                feed_dict=feed_dict
            )
            losses.append(np.transpose(np.array([l1, l2, l3]), axes=(1, 0)))

        losses = np.concatenate(losses, axis=0)

        return losses

    def encode(self, states, batch_size=100):

        num_steps = int(np.ceil(len(states) / batch_size))
        encodings = []
        log_probs = []

        for i in range(num_steps):
            batch_slice = np.index_exp[i * batch_size:(i + 1) * batch_size]

            feed_dict = {
                self.states_pl: states[batch_slice]
            }

            tmp_encodings, tmp_log_probs = \
                self.session.run([self.state_sample_t, self.z1_log_cond], feed_dict=feed_dict)
            encodings.append(tmp_encodings)
            log_probs.append(tmp_log_probs)

        encodings = np.concatenate(encodings, axis=0)
        log_probs = np.concatenate(log_probs, axis=0)

        return encodings, log_probs

    def build(self):

        self.build_placeholders_()
        self.build_model()
        self.build_predictors_()
        self.build_mixture_components_()
        self.build_training_()

        self.saver = tf.train.Saver()

    def build_mixture_components_(self):

        self.mixtures_mu_v = tf.get_variable(
            "mixtures_mu", initializer=tf.random_normal_initializer(
                mean=0.0, stddev=self.mixtures_mu_init_sd, dtype=tf.float32
            ), shape=(self.num_components, self.num_blocks)
        )
        self.mixtures_logvar_v = tf.get_variable(
            "mixtures_var", initializer=tf.random_normal_initializer(
                mean=self.mixtures_sd_init_mu, stddev=self.mixtures_sd_init_sd, dtype=tf.float32
            ), shape=(self.num_components, self.num_blocks)
        )
        self.mixtures_var_t = tf.exp(self.mixtures_logvar_v)

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

    def build_prior_likelihood_(self):

        self.z1_log_probs = self.many_multivariate_normals_log_pdf(
            self.state_sample_t, self.mixtures_mu_v, self.mixtures_var_t, self.mixtures_logvar_v
        )
        self.z1_log_joint = self.z1_log_probs - np.log(self.num_components)

        self.full_prior_log_likelihood_t = tf.reduce_logsumexp(self.z1_log_joint, axis=1)
        self.prior_log_likelihood_t = tf.reduce_mean(self.full_prior_log_likelihood_t, axis=0)

        self.z1_log_cond = self.z1_log_joint - tf.reduce_logsumexp(self.z1_log_joint, axis=1)[:, tf.newaxis]

    def build_model_training_(self):

        encoder_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.ENCODER_NAMESPACE)
        encoder_optimizer = agent_utils.get_optimizer(self.optimizer, self.learning_rate)

        self.encoder_train_step = encoder_optimizer.minimize(
            self.loss_t, global_step=self.global_step, var_list=encoder_variables
        )

    def build_prior_training_(self):

        mixture_variables = [self.mixtures_mu_v, self.mixtures_logvar_v]
        mixture_optimizer = agent_utils.get_optimizer(self.optimizer, self.learning_rate)

        self.prior_train_step = mixture_optimizer.minimize(
            self.prior_loss_t, global_step=self.global_step, var_list=mixture_variables
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
