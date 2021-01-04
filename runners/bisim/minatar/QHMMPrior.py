import pickle
import numpy as np
import tensorflow as tf
import constants
from abstraction.quotient_mdp_full import QuotientMDPFull
from agents.quotient_mdp_nbisim import QuotientMDPNBisim
from model.bisim.bisim import QPredictionModelHMMPrior
from runners.bisim.puck_stack.QHMMPrior import QHMMPriorRunner
from utils.dataset import prepare_minatar_dataset
import vis_utils
import seaborn as sns
import matplotlib.colors as colors
from minatar import Environment
from agents.solver_minatar import SolverMinAtar


class QHMMPriorRunnerMinAtar(QHMMPriorRunner):

    def __init__(self, load_path, game, num_actions, logger, saver, num_blocks, num_components,
                 encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
                 disable_batch_norm=False, disable_softplus=False, no_sample=False, only_one_q_value=True,
                 gt_q_values=False, validation_fraction=0.2, validation_freq=1000, summaries=None, load_model_path=None,
                 batch_size=100, beta3=0.0, post_train_hmm=False, post_train_prior=False, post_train_t_and_prior=False,
                 save_gifs=False, zero_sd_after_training=False, hard_abstract_state=False,
                 freeze_hmm_no_entropy_at=None, cluster_predict_qs=False, cluster_predict_qs_weight=0.1,
                 discount=0.99, prune_abstraction=False, prune_threshold=0.01, prune_abstraction_new_means=False,
                 sample_abstract_state=False, softmax_policy=False, softmax_policy_temp=1.0,
                 fix_prior_training=False, model_learning_rate=None, q_scaling_factor=10.0,
                 eval_episodes=100):

        super(QHMMPriorRunnerMinAtar, self).__init__(
            load_path, 1, None, logger, saver, num_blocks, num_components, None,
            encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
            disable_batch_norm=disable_batch_norm, disable_softplus=disable_softplus, no_sample=no_sample,
            only_one_q_value=only_one_q_value, gt_q_values=gt_q_values, disable_resize=True,
            oversample=False, validation_fraction=validation_fraction, validation_freq=validation_freq,
            summaries=summaries, load_model_path=load_model_path, batch_size=batch_size, beta3=beta3,
            post_train_hmm=post_train_hmm, post_train_prior=post_train_prior,
            post_train_t_and_prior=post_train_t_and_prior, zero_sd_after_training=zero_sd_after_training,
            cluster_predict_qs=cluster_predict_qs, cluster_predict_qs_weight=cluster_predict_qs_weight,
            discount=discount, hard_abstract_state=hard_abstract_state, save_gifs=save_gifs,
            prune_abstraction=prune_abstraction, prune_threshold=prune_threshold,
            prune_abstraction_new_means=prune_abstraction_new_means, sample_abstract_state=sample_abstract_state,
            softmax_policy=softmax_policy, softmax_policy_temp=softmax_policy_temp,
            fix_prior_training=fix_prior_training, model_learning_rate=model_learning_rate
        )

        self.game = game
        self.num_actions = num_actions
        self.freeze_hmm_no_entropy_at = freeze_hmm_no_entropy_at
        self.q_scaling_factor = q_scaling_factor
        self.eval_episodes = eval_episodes

    def main_training_loop(self):

        self.prepare_losses_()

        if self.load_model_path is None:

            num_steps = self.num_steps
            if not self.disable_batch_norm:
                num_steps += 10000

            for train_step in range(num_steps):

                if train_step == num_steps - 10000 and not self.disable_batch_norm:
                    self.to_run[constants.TRAIN_STEP] = self.model.update_op

                if train_step % self.validation_freq == 0:

                    tmp_valid_losses = self.model.validate(
                        self.valid_dataset[constants.STATES], self.valid_dataset[constants.HAND_STATES],
                        self.valid_dataset[constants.ACTIONS], self.valid_dataset[constants.Q_VALUES],
                        self.valid_dataset[constants.NEXT_STATES], self.valid_dataset[constants.NEXT_HAND_STATES],
                        self.valid_dataset[constants.DONES]
                    )

                    self.valid_losses.append(np.mean(tmp_valid_losses, axis=0))

                epoch_step = train_step % self.epoch_size

                if train_step > 0 and epoch_step == 0:

                    self.logger.info("training step {:d}".format(train_step))

                    self.dataset.shuffle()

                    if len(self.tmp_epoch_losses) > 0:
                        self.add_epoch_losses_()

                if self.freeze_hmm_no_entropy_at is not None and train_step == self.freeze_hmm_no_entropy_at:
                    self.model.session.run(tf.assign(self.model.beta1_v, 0.0))
                    self.to_run[constants.TRAIN_STEP] = self.model.encoder_train_step

                feed_dict = self.get_feed_dict_(epoch_step)
                bundle = self.model.session.run(self.to_run, feed_dict=feed_dict)

                to_add = [
                    bundle[constants.TOTAL_LOSS], bundle[constants.Q_LOSS],
                    bundle[constants.PRIOR_LOG_LIKELIHOOD], bundle[constants.STATE_ENCODER_ENTROPY]
                ]

                self.add_step_losses_(to_add, train_step)

                if self.writer is not None:
                    self.writer.add_summary(bundle[constants.SUMMARY], global_step=train_step)

            if len(self.tmp_epoch_losses) > 0:
                self.add_epoch_losses_()

            self.postprocess_losses_()
            self.plot_losses_()

    def evaluate_and_visualize(self):

        self.inference_()
        self.setup_clusters_()
        self.evaluate_cluster_purities_()
        #self.evaluate_transitions_method_2_()
        #self.evaluate_transitions_method_3_()
        #self.plot_latent_space_()
        #self.plot_latent_space_with_centroids_()
        #self.plot_perplexities_()
        #self.plot_state_components_examples_()
        #self.predict_qs_from_clusters_hard_assignment_()
        #self.predict_next_clusters_hard_assignment_()

        #if self.prune_state_abstraction_new_means:
        #    self.prune_state_abstraction_new_means_()
        #    self.inference_()
        #    self.setup_clusters_()

        self.get_cluster_rewards_()
        self.plot_cluster_rewards_()
        self.get_cluster_rewards_no_actions_()
        self.plot_cluster_rewards_no_actions_()
        self.get_cluster_failures_()
        self.plot_cluster_failures_()
        self.get_cluster_failures_soft_()

        self.logger.info("running default abstract agent")
        self.setup_abstract_mdp_()
        self.plot_abstract_mdp_planning_()
        self.run_abstract_agent_()

        #self.logger.info("running abstract agent with new transitions")
        #self.setup_abstract_mdp_(new_transitions=True)
        #self.run_abstract_agent_(new_transitions=True)

        #self.logger.info("running abstract agent with failure rewards")
        #self.setup_abstract_mdp_(failure_rewards=True)
        #self.run_abstract_agent_(failure_rewards=True)

        #self.logger.info("running abstract agent with soft failure rewards")
        #self.setup_abstract_mdp_(soft_failure_rewards=True)
        #self.run_abstract_agent_(soft_failure_rewards=True)

        #self.logger.info("running abstract agent with no action rewards")
        #self.setup_abstract_mdp_(rewards_no_actions=True)
        #self.run_abstract_agent_(rewards_no_actions=True)

        #self.logger.info("running abstract agent with new transitions and no action rewards")
        #self.setup_abstract_mdp_(new_transitions=True, rewards_no_actions=True)
        #self.run_abstract_agent_(new_transitions=True, rewards_no_actions=True)

        self.get_cluster_q_values_()
        self.plot_cluster_q_values_()
        self.run_abstract_agent_(cluster_q_values=True)

        #if self.cluster_predict_qs:
        #    self.run_abstract_agent_(cluster_q_values=True, cluster_q_values_from_model=True)

        #self.learn_abstract_qs_()
        #self.run_abstract_agent_(learned_cluster_q_values=True)

        #self.learn_abstract_qs_(init_qs=self.cluster_q_values)
        #self.run_abstract_agent_(learned_cluster_q_values=True)

        #self.learn_abstract_qs_(init_qs=self.abstract_mdp.state_action_values)
        #self.run_abstract_agent_(learned_cluster_q_values=True)

    def get_cluster_failures_(self):

        self.cluster_failures = np.ones((self.num_components, self.num_actions), dtype=np.float32)

        for cluster_idx in range(self.num_components):
            for action_idx in range(self.num_actions):
                mask = np.bitwise_and(self.train_cluster_assignment == cluster_idx,
                                      self.dataset[constants.ACTIONS] == action_idx)
                if np.sum(mask) > 0:
                    self.cluster_failures[cluster_idx, action_idx] = \
                        np.sum(self.dataset[constants.DONES][mask].astype(np.float32)) / np.sum(mask)

        self.logger.info("component failure probs:")
        for i in range(self.num_components):
            self.logger.info("component {:d}".format(i) + str(self.cluster_failures[i]))

    def plot_cluster_failures_(self):

        vis_utils.plot_many_histograms(
            [self.cluster_failures.reshape(-1)],
            ["cluster_failure_probs"], xlabel="bins",
            num_bins=50, saver=self.saver
        )

    def get_cluster_failures_soft_(self):

        self.cluster_failures_soft = np.ones((self.num_components, self.num_actions), dtype=np.float32)

        for action_idx in range(self.num_actions):

            mask = self.dataset[constants.ACTIONS] == action_idx
            probs = np.exp(self.train_cluster_log_probs[mask])
            dones = self.dataset[constants.DONES].astype(np.float32)[mask][:, np.newaxis]

            self.cluster_failures_soft[:, action_idx] = np.sum(probs * dones, axis=0) / np.sum(probs, axis=0)

        self.logger.info("component failure soft probs:")
        for i in range(self.num_components):
            self.logger.info("component {:d}".format(i) + str(self.cluster_failures[i]))

    def setup_abstract_mdp_(self, new_transitions=False, rewards_no_actions=False, failure_rewards=False,
                            soft_failure_rewards=False):

        if new_transitions:
            t = self.new_T_add_one
        else:
            t = self.t_softmax

        t = np.transpose(t, axes=[1, 2, 0])

        if self.prune_abstraction:
            self.prune_state_abstraction_(t)

        if rewards_no_actions:
            r = self.cluster_rewards_no_actions
        elif failure_rewards:
            r = - self.cluster_failures
        elif soft_failure_rewards:
            r = - self.cluster_failures_soft
        else:
            r = self.cluster_rewards

        self.abstract_mdp = QuotientMDPFull(
            list(range(self.num_components)), list(range(self.num_actions)), t, r, discount=self.discount,
            absorbing_state_threshold=1e+6, state_rewards=rewards_no_actions
        )

        self.abstract_mdp.value_iteration()

    def run_abstract_agent_(self, cluster_q_values=False, cluster_q_values_from_model=False,
                            learned_cluster_q_values=False, new_transitions=False, rewards_no_actions=False,
                            failure_rewards=False, soft_failure_rewards=False):

        def classify(state):

            _, log_probs = self.model.encode(
                state[np.newaxis], hand_states=np.zeros((1, 1), dtype=np.int32)
            )
            log_probs = log_probs[0]

            if self.hard_abstract_state:
                idx = np.argmax(log_probs)
                output = np.zeros_like(log_probs)
                output[idx] = 1.0
            elif self.sample_abstract_state:
                idx = np.random.choice(list(range(len(log_probs))), p=np.exp(log_probs))
                output = np.zeros_like(log_probs)
                output[idx] = 1.0
            else:
                output = np.exp(log_probs)

            return output

        self.game_env = Environment(
            self.game, difficulty_ramping=False, sticky_action_prob=0.0
        )

        if cluster_q_values:
            if cluster_q_values_from_model:
                q_values = self.model.session.run(self.model.cluster_qs_v)
                rewards_name = "rewards_q_values_from_model"
                gifs_name = "gifs_q_values_from_model"
            else:
                q_values = self.cluster_q_values
                rewards_name = "rewards_q_values"
                gifs_name = "gifs_q_values"
        elif learned_cluster_q_values:
            q_values = self.learned_qs
            rewards_name = "rewards_learned_q_values"
            gifs_name = "gifs_learned_q_values"
        else:
            q_values = self.abstract_mdp.state_action_values
            rewards_name = "rewards"
            gifs_name = "gifs"

        if new_transitions:
            rewards_name += "_new_t"
            gifs_name += "_new_t"

        if rewards_no_actions:
            rewards_name += "_r_no_a"
            gifs_name += "_r_no_a"

        if failure_rewards:
            rewards_name += "_failure_r"
            gifs_name += "_failure_r"

        if soft_failure_rewards:
            rewards_name += "_soft_failure_r"
            gifs_name += "_soft_failure_r"

        self.abstract_agent = QuotientMDPNBisim(
            classify, self.game_env, q_values, minatar=True, softmax_policy=self.softmax_policy,
            softmax_policy_temp=self.softmax_policy_temp
        )

        if self.save_gifs:
            gifs_path = self.saver.get_new_dir(gifs_name)
        else:
            gifs_path = None

        solver = SolverMinAtar(
            self.game_env, self.abstract_agent, int(1e+7), int(1e+7), 0, max_episodes=self.eval_episodes, train=False,
            gif_save_path=gifs_path, rewards_file=self.saver.get_save_file(rewards_name, "dat")
        )
        solver.run()

    def plot_state_components_examples_(self):

        cmap = sns.color_palette("cubehelix", 4)
        cmap.insert(0, (0, 0, 0))
        cmap = colors.ListedColormap(cmap)
        bounds = [i for i in range(4 + 2)]
        norm = colors.BoundaryNorm(bounds, 4 + 1)

        hard_assignment = np.argmax(self.valid_cluster_log_probs, axis=1)

        for cluster_idx in range(self.num_components):

            mask = hard_assignment == cluster_idx

            if np.sum(mask) < 25:
                continue

            name = "state_cluster_{:d}".format(cluster_idx)
            images = self.valid_dataset[constants.STATES][mask][:25]
            new_images = np.empty((images.shape[0], images.shape[1], images.shape[2]), dtype=np.float32)

            for i in range(len(images)):
                new_images[i] = np.amax(images[i] * np.reshape(np.arange(4) + 1, (1, 1, -1)), 2) + 0.5

            vis_utils.plot_5x5_grid(new_images, name, self.saver, cmap=cmap, norm=norm)

    def prepare_dataset_(self):

        with open(self.load_path, "rb") as file:
            transitions = pickle.load(file)

        self.dataset = prepare_minatar_dataset(transitions, q_scaling_factor=self.q_scaling_factor)

        vis_utils.plot_many_histograms(
            [self.dataset[constants.Q_VALUES].reshape(-1), self.dataset[constants.NEXT_Q_VALUES].reshape(-1)],
            ["q_values_hist", "next_q_values_hist"], xlabel="q-values",
            num_bins=50, saver=self.saver
        )

        # TODO: dataset problem
        if len(self.dataset[constants.Q_VALUES].shape) == 3:
            self.dataset[constants.Q_VALUES] = self.dataset[constants.Q_VALUES][:, 0, :]

        if len(self.dataset[constants.NEXT_Q_VALUES].shape) == 3:
            self.dataset[constants.NEXT_Q_VALUES] = self.dataset[constants.NEXT_Q_VALUES][:, 0, :]

        self.input_shape = self.dataset[constants.STATES].shape[1:]

        self.dataset.shuffle()

        num_samples = self.dataset.size
        valid_samples = int(num_samples * self.validation_fraction)
        self.valid_dataset = self.dataset.split(valid_samples)

        self.epoch_size = self.dataset[constants.STATES].shape[0] // self.batch_size

        self.train_not_dones = np.logical_not(self.dataset[constants.DONES])
        self.valid_not_dones = np.logical_not(self.valid_dataset[constants.DONES])

        self.log_dataset_stats_()

    def log_dataset_stats_(self):

        for dataset, name in zip([self.dataset, self.valid_dataset], ["train", "valid"]):

            self.logger.info("number of blocks: {:d}".format(self.num_blocks))

            self.logger.info("actions: min {:d} max {:d}".format(
                np.min(dataset[constants.ACTIONS]), np.max(dataset[constants.ACTIONS]))
            )
            self.logger.info("rewards: min {:.0f} max {:.0f}".format(
                np.min(dataset[constants.REWARDS]), np.max(dataset[constants.REWARDS]))
            )
            self.logger.info("state labels: min {:d} max {:d}".format(
                np.min(dataset[constants.STATE_LABELS]), np.max(dataset[constants.STATE_LABELS]))
            )

    def prepare_model_(self):

        self.model = QPredictionModelHMMPrior(
            list(self.input_shape), self.num_blocks, self.num_components, self.num_actions,
            [32, 64], [3, 3], [1, 1], [128], self.encoder_learning_rate,
            self.beta0, self.beta1, self.beta2, self.weight_decay, self.encoder_optimizer, self.num_steps,
            batch_norm=not self.disable_batch_norm, softplus=not self.disable_softplus, no_sample=self.no_sample,
            only_one_q_value=self.only_one_q_value, beta3=self.beta3, cluster_predict_qs=self.cluster_predict_qs,
            cluster_predict_qs_weight=self.cluster_predict_qs_weight,
            fix_prior_training=self.fix_prior_training, model_learning_rate=self.model_learning_rate
        )
        self.model.build()

        if self.summaries:
            self.model.build_summaries()

        self.model.start_session()

        if self.load_model_path is not None:
            self.model.load(self.load_model_path)

        if self.summaries:
            run_dir = self.saver.get_new_dir("summary")

            self.tf_summaries = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(run_dir, graph=self.model.session.graph)

        self.to_run = {
            constants.TRAIN_STEP: self.model.train_step,
            constants.TOTAL_LOSS: self.model.loss_t,
            constants.Q_LOSS: self.model.q_loss_t,
            constants.PRIOR_LOG_LIKELIHOOD: self.model.prior_log_likelihood,
            constants.STATE_ENCODER_ENTROPY: self.model.encoder_entropy_t,
        }

        if self.summaries:
            self.to_run[constants.SUMMARY] = self.tf_summaries
