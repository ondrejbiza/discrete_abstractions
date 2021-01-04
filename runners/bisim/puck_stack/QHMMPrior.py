import copy as cp
from envs.continuous_puck_stack import ContinuousPuckStack
from runners.bisim.puck_stack.QGMMPrior import QGMMPriorRunner
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import tensorflow as tf
from model.bisim.bisim import QPredictionModelHMMPrior
import constants
from model.cluster import Cluster
import vis_utils
from agents.quotient_mdp_nbisim import QuotientMDPNBisim
from abstraction.quotient_mdp_full import QuotientMDPFull
from agents.solver import Solver


class QHMMPriorRunner(QGMMPriorRunner):

    def __init__(self, load_path, grid_size, num_pucks, logger, saver, num_blocks, num_components, hiddens,
                 encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
                 disable_batch_norm=False, disable_softplus=False, no_sample=False, only_one_q_value=True,
                 gt_q_values=False, disable_resize=False, oversample=False, validation_fraction=0.2,
                 validation_freq=1000, summaries=None, load_model_path=None, batch_size=100, beta3=0.0,
                 post_train_hmm=False, post_train_prior=False, post_train_t_and_prior=False,
                 zero_sd_after_training=False, cluster_predict_qs=False, cluster_predict_qs_weight=0.1,
                 hard_abstract_state=False, save_gifs=False, prune_abstraction=False,
                 prune_threshold=0.01, prune_abstraction_new_means=False, old_bn_settings=False,
                 sample_abstract_state=False, softmax_policy=False, softmax_policy_temp=1.0,
                 include_goal_states=False, discount=0.9, soft_picture_goals=False,
                 shift_q_values=False, show_qs=False, fix_prior_training=False, learning_rate_steps=None,
                 learning_rate_values=None, q_values_noise_sd=None, gt_q_values_all_actions=False,
                 new_dones=False, model_learning_rate=None):

        super(QHMMPriorRunner, self).__init__(
            load_path, grid_size, num_pucks, logger, saver, num_blocks, num_components, hiddens,
            encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
            disable_batch_norm=disable_batch_norm, disable_softplus=disable_softplus, no_sample=no_sample,
            only_one_q_value=only_one_q_value, gt_q_values=gt_q_values, disable_resize=disable_resize,
            oversample=oversample, validation_fraction=validation_fraction, validation_freq=validation_freq,
            summaries=summaries, load_model_path=load_model_path, batch_size=batch_size,
            zero_sd_after_training=zero_sd_after_training, hard_abstract_state=hard_abstract_state,
            save_gifs=save_gifs, include_goal_states=include_goal_states, discount=discount,
            shift_q_values=shift_q_values, show_qs=show_qs, q_values_noise_sd=q_values_noise_sd,
            gt_q_values_all_actions=gt_q_values_all_actions, new_dones=new_dones
        )

        self.beta3 = beta3
        self.post_train_hmm = post_train_hmm
        self.post_train_prior = post_train_prior
        self.post_train_t_and_prior = post_train_t_and_prior
        self.cluster_predict_qs = cluster_predict_qs
        self.cluster_predict_qs_weight = cluster_predict_qs_weight
        self.prune_abstraction = prune_abstraction
        self.prune_threshold = prune_threshold
        self.prune_state_abstraction_new_means = prune_abstraction_new_means
        self.old_bn_settings = old_bn_settings
        self.sample_abstract_state = sample_abstract_state
        self.softmax_policy = softmax_policy
        self.softmax_policy_temp = softmax_policy_temp
        self.soft_picture_goals = soft_picture_goals
        self.fix_prior_training = fix_prior_training
        self.learning_rate_steps = learning_rate_steps
        self.learning_rate_values = learning_rate_values
        self.model_learning_rate = model_learning_rate

        self.means = None
        self.stds = None
        self.new_means = None

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

                    # TODO: shuffle the dataset

                    if len(self.tmp_epoch_losses) > 0:
                        self.add_epoch_losses_()

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

    def post_hmm_training_loop(self, training_steps):

        self.post_train_prior_losses = []

        to_run = {constants.PRIOR_LOG_LIKELIHOOD: self.model.prior_log_likelihood}

        if self.post_train_hmm:
            to_run[constants.TRAIN_STEP] = self.model.hmm_train_step
        elif self.post_train_t_and_prior:
            to_run[constants.TRAIN_STEP] = self.model.T_and_prior_train_step
        else:
            to_run[constants.TRAIN_STEP] = self.model.prior_train_step

        for train_step in range(training_steps):

            epoch_step = train_step % self.epoch_size
            feed_dict = self.get_feed_dict_(epoch_step)
            bundle = self.model.session.run(to_run, feed_dict=feed_dict)

            self.post_train_prior_losses.append([bundle[constants.PRIOR_LOG_LIKELIHOOD]])

        self.post_train_prior_losses = np.array(self.post_train_prior_losses)
        self.plot_post_hmm_training_losses_()

    def evaluate_and_visualize(self):

        self.inference_()
        self.setup_clusters_()
        self.evaluate_cluster_purities_()
        self.evaluate_transitions_method_2_()
        #self.evaluate_transitions_method_3_()
        self.plot_latent_space_()
        self.plot_latent_space_with_centroids_()
        self.plot_perplexities_()
        #self.plot_state_components_examples_()
        #self.predict_qs_from_clusters_hard_assignment_()
        #self.predict_next_clusters_hard_assignment_()

        if self.prune_state_abstraction_new_means:
            self.prune_state_abstraction_new_means_()
            self.inference_()
            self.setup_clusters_()

        self.get_cluster_rewards_()
        self.plot_cluster_rewards_()
        self.setup_abstract_mdp_(self.t_softmax, self.cluster_rewards, state_rewards=False)
        self.plot_abstract_mdp_planning_()
        self.run_abstract_agent_(self.abstract_mdp.state_action_values)

        self.get_cluster_q_values_()
        self.plot_cluster_q_values_()
        self.run_abstract_agent_(self.cluster_q_values, append_name="q_values")

        #self.evaluate_all_positions_()

        #self.setup_abstract_mdp_(new_transitions=True)
        #self.plot_abstract_mdp_planning_(new_transitions=True)
        #self.run_abstract_agent_(new_transitions=True)

    def inference_(self):

        # get HMM parameters
        self.mixtures_mu, self.mixtures_logvar, self.t_softmax = self.model.session.run(
            [self.model.mixtures_mu_v, self.model.mixtures_logvar_v, self.model.t_softmax_t]
        )
        self.saver.save_array(self.t_softmax, "transition_matrix")

        # get current and next state embeddings and clusters
        self.train_embeddings, self.train_cluster_log_probs = self.model.encode(
                self.dataset[constants.STATES], self.dataset[constants.HAND_STATES][:, np.newaxis],
                zero_sd=self.zero_sd_after_training, new_means=self.new_means
            )
        self.train_cluster_assignment = np.argmax(self.train_cluster_log_probs, axis=1)

        self.valid_embeddings, self.valid_cluster_log_probs = self.model.encode(
                self.valid_dataset[constants.STATES], self.valid_dataset[constants.HAND_STATES][:, np.newaxis],
                zero_sd=self.zero_sd_after_training, new_means=self.new_means
            )
        self.valid_cluster_assignment = np.argmax(self.valid_cluster_log_probs, axis=1)

        self.train_next_embeddings, self.train_next_cluster_log_probs =  self.model.encode(
                self.dataset[constants.NEXT_STATES], self.dataset[constants.NEXT_HAND_STATES][:, np.newaxis],
                zero_sd=self.zero_sd_after_training, new_means=self.new_means
            )
        self.train_next_cluster_assignment = np.argmax(self.train_next_cluster_log_probs, axis=1)

        self.valid_next_embeddings, self.valid_next_cluster_log_probs = self.model.encode(
                self.valid_dataset[constants.NEXT_STATES],
                self.valid_dataset[constants.NEXT_HAND_STATES][:, np.newaxis],
                zero_sd=self.zero_sd_after_training, new_means=self.new_means
            )
        self.valid_next_cluster_assignment = np.argmax(self.valid_next_cluster_log_probs, axis=1)

        _, self.train_predicted_cluster_log_probs, _, _ = self.model.predict_next_states(
            self.dataset[constants.STATES], self.dataset[constants.HAND_STATES],
            self.dataset[constants.ACTIONS], zero_sd=self.zero_sd_after_training, new_means=self.new_means
        )

        self.train_predicted_cluster_assignment = np.argmax(self.train_predicted_cluster_log_probs, axis=1)

        _, self.valid_predicted_cluster_log_probs, _, _ = self.model.predict_next_states(
            self.valid_dataset[constants.STATES], self.valid_dataset[constants.HAND_STATES],
            self.valid_dataset[constants.ACTIONS], zero_sd=self.zero_sd_after_training, new_means=self.new_means
        )
        self.valid_predicted_cluster_assignment = np.argmax(self.valid_predicted_cluster_log_probs, axis=1)

    def evaluate_transitions_method_2_(self):

        to_save = []

        for tmp_train_assigned_labels, tmp_valid_assigned_labels, tmp_name in zip(
                [self.train_assigned_labels, self.train_assigned_labels_soft],
                [self.valid_assigned_labels, self.valid_assigned_labels_soft],
                ["transition accuracy, method 2", "soft transition accuracy, method2"]):

            train_next_state_predicted_labels = Cluster.translate_cluster_labels_to_assigned_labels(
                self.train_next_cluster_assignment, tmp_train_assigned_labels
            )
            valid_next_state_predicted_labels = Cluster.translate_cluster_labels_to_assigned_labels(
                self.valid_next_cluster_assignment, tmp_valid_assigned_labels
            )
            valid_next_state_predicted_labels_train_assignment = Cluster.translate_cluster_labels_to_assigned_labels(
                self.valid_next_cluster_assignment, tmp_train_assigned_labels
            )

            train_transition_accuracy_method_2 = np.mean(
                (train_next_state_predicted_labels == self.dataset[constants.NEXT_STATE_LABELS])[self.train_not_dones]
            )
            valid_transition_accuracy_method_2 = np.mean(
                (valid_next_state_predicted_labels == self.valid_dataset[constants.NEXT_STATE_LABELS])[self.valid_not_dones]
            )
            valid_transition_accuracy_method_2_train_assignment = np.mean(
                (valid_next_state_predicted_labels_train_assignment == self.valid_dataset[constants.NEXT_STATE_LABELS])[
                    self.valid_not_dones]
            )

            self.logger.info("train {:s}: {:.2f}%".format(tmp_name, train_transition_accuracy_method_2 * 100))
            self.logger.info("valid {:s}: {:.2f}%".format(tmp_name, valid_transition_accuracy_method_2 * 100))
            self.logger.info(
                "valid {:s}, train assignment: {:.2f}%".format(
                    tmp_name, valid_transition_accuracy_method_2_train_assignment * 100
                )
            )

            to_save.append(train_transition_accuracy_method_2)
            to_save.append(valid_transition_accuracy_method_2)
            to_save.append(valid_transition_accuracy_method_2_train_assignment)

        self.saver.save_array_as_txt(to_save, "transition_accuracies_method_2")

    def evaluate_transitions_method_3_(self):

        new_T = np.zeros((self.num_actions, self.num_components, self.num_components), dtype=np.float32)
        new_T_add_one = np.ones((self.num_actions, self.num_components, self.num_components), dtype=np.float32)

        for a_idx in range(self.num_actions):
            for k in range(self.num_components):
                for l in range(self.num_components):
                    mask = np.logical_and(
                        np.logical_and(
                            self.dataset[constants.ACTIONS] == a_idx, self.train_cluster_assignment == k),
                        self.train_next_cluster_assignment == l
                    )

                    new_T[a_idx, k, l] = np.sum(mask)
                    new_T_add_one[a_idx, k, l] += np.sum(mask)

        new_T_sum = np.sum(new_T, axis=2)
        new_T[new_T_sum == 0.0] = 1.0

        self.new_T = new_T / np.sum(new_T, axis=2)[:, :, np.newaxis]
        self.new_T_add_one = new_T_add_one / np.sum(new_T_add_one, axis=2)[:, :, np.newaxis]

        self.saver.save_array(self.new_T, "new_transition_matrix")

        gather_valid_new_T = self.new_T[self.valid_dataset[constants.ACTIONS]]
        gather_valid_new_T_add_one = self.new_T_add_one[self.valid_dataset[constants.ACTIONS]]

        valid_sample_probs_norm = self.valid_cluster_log_probs - logsumexp(self.valid_cluster_log_probs, axis=1)[:, np.newaxis]

        valid_next_state_cluster_prediction = \
            np.sum(np.exp(valid_sample_probs_norm)[:, :, np.newaxis] * gather_valid_new_T, axis=1)
        valid_next_state_cluster_prediction_add_one = \
            np.sum(np.exp(valid_sample_probs_norm)[:, :, np.newaxis] * gather_valid_new_T_add_one, axis=1)

        valid_next_state_cluster_assignment = np.argmax(valid_next_state_cluster_prediction, axis=1)
        valid_next_state_cluster_assignment_add_one = np.argmax(valid_next_state_cluster_prediction_add_one, axis=1)

        valid_next_state_label_assignment = Cluster.translate_cluster_labels_to_assigned_labels(
            valid_next_state_cluster_assignment, self.valid_assigned_labels
        )
        valid_next_state_label_assignment_add_one = Cluster.translate_cluster_labels_to_assigned_labels(
            valid_next_state_cluster_assignment_add_one, self.valid_assigned_labels
        )

        valid_transition_accuracy_method_3 = np.mean(
            (valid_next_state_label_assignment == self.valid_dataset[constants.NEXT_STATE_LABELS])[self.valid_not_dones]
        )
        valid_transition_accuracy_method_3_add_one = np.mean(
            (valid_next_state_label_assignment_add_one == self.valid_dataset[constants.NEXT_STATE_LABELS])[self.valid_not_dones]
        )

        self.logger.info("valid transition accuracy, method 3: {:.2f}%".format(valid_transition_accuracy_method_3 * 100))
        self.logger.info("valid transition accuracy, method 3, add one: {:.2f}%".format(
            valid_transition_accuracy_method_3_add_one * 100)
        )
        self.saver.save_array_as_txt(
            [valid_transition_accuracy_method_3, valid_transition_accuracy_method_3_add_one],
            "transition_accuracy_method_3"
        )

    def prune_state_abstraction_(self, transitions):

        state_cluster_mass = np.sum(np.exp(self.train_cluster_log_probs), axis=0)
        total_mass = np.sum(state_cluster_mass)

        fractions = state_cluster_mass / total_mass

        to_delete = np.where(fractions < self.prune_threshold)

        self.logger.info("state pruning to delete:" + str(to_delete))

        for cluster_idx in to_delete:
            transitions[cluster_idx, :, :] = 0.0
            transitions[:, cluster_idx, :] = 0.0

    def prune_state_abstraction_new_means_(self):

        state_cluster_mass = np.sum(np.exp(self.train_cluster_log_probs), axis=0)
        total_mass = np.sum(state_cluster_mass)

        fractions = state_cluster_mass / total_mass

        to_delete = np.where(fractions < self.prune_threshold)

        self.logger.info("state pruning to delete:" + str(to_delete))

        self.new_means = cp.deepcopy(self.mixtures_mu)
        self.new_means[to_delete] = 1e+6

    def learn_abstract_qs_(self, alpha=0.5, num_steps=1000, max_diff_threshold=0.001, init_qs=None):

        if init_qs is None:
            self.learned_qs = np.zeros((self.num_components, self.num_actions), dtype=np.float32)
        else:
            self.learned_qs = cp.deepcopy(init_qs)

        for step_idx in range(num_steps):

            new_learned_qs = cp.deepcopy(self.learned_qs)
            max_qs = np.max(self.learned_qs, axis=1)

            for comp_idx in range(self.num_components):
                for a_idx in range(self.num_actions):

                    mask = np.bitwise_and(
                        self.train_cluster_assignment == comp_idx,
                        self.dataset[constants.ACTIONS] == a_idx
                    )

                    if np.sum(mask) == 0:
                        continue

                    rewards = self.dataset[constants.REWARDS][mask]
                    next_comps = self.train_next_cluster_assignment[mask]

                    select_max_qs = max_qs[next_comps]
                    new_learned_qs[comp_idx, a_idx] = new_learned_qs[comp_idx, a_idx] + \
                        np.mean(alpha * (rewards + self.discount * select_max_qs - new_learned_qs[comp_idx, a_idx]))

            max_diff = np.max(np.abs(self.learned_qs - new_learned_qs))
            self.learned_qs = new_learned_qs

            print(step_idx, max_diff)

            if max_diff < max_diff_threshold:
                break

    def get_cluster_rewards_(self):

        self.cluster_rewards = np.zeros((self.num_components, self.num_actions), dtype=np.float32)
        self.cluster_rewards_stds = np.zeros((self.num_components, self.num_actions), dtype=np.float32)

        for cluster_idx in range(self.num_components):
            for action_idx in range(self.num_actions):
                # we execute action 'action_idx' in abstract state 'cluster idx'
                mask = np.bitwise_and(self.train_cluster_assignment == cluster_idx, self.dataset[constants.ACTIONS] == action_idx)
                # average rewards if such abstract-state-action pair exists
                if np.sum(mask) > 0:
                    self.cluster_rewards[cluster_idx, action_idx] = np.mean(self.dataset[constants.REWARDS][mask])
                    self.cluster_rewards_stds[cluster_idx, action_idx] = np.std(self.dataset[constants.REWARDS][mask])

    def plot_cluster_rewards_(self):

        vis_utils.plot_many_histograms(
            [self.cluster_rewards.reshape(-1), self.cluster_rewards_stds.reshape(-1)],
            ["cluster_rewards", "cluster_reward_stds"], xlabel="bins",
            num_bins=50, saver=self.saver
        )

    def get_cluster_rewards_no_actions_(self):

        self.cluster_rewards_no_actions = np.zeros(self.num_components, dtype=np.float32)
        self.cluster_rewards_no_actions_stds = np.zeros(self.num_components, dtype=np.float32)

        for cluster_idx in range(self.num_components):
            # we execute action 'action_idx' in abstract state 'cluster idx'
            mask = self.train_cluster_assignment == cluster_idx
            # average rewards if such abstract-state-action pair exists
            if np.sum(mask) > 0:
                self.cluster_rewards_no_actions[cluster_idx] = np.mean(self.dataset[constants.REWARDS][mask])
                self.cluster_rewards_no_actions_stds[cluster_idx] = np.std(self.dataset[constants.REWARDS][mask])

        self.logger.info("component rewards:")
        for i in range(self.num_components):
            self.logger.info("component {:d}: {:.4f}".format(i, self.cluster_rewards_no_actions[i]))

    def plot_cluster_rewards_no_actions_(self):

        vis_utils.plot_many_histograms(
            [self.cluster_rewards_no_actions, self.cluster_rewards_no_actions_stds],
            ["cluster_rewards_no_actions", "cluster_reward_stds_no_actions"], xlabel="bins",
            num_bins=50, saver=self.saver
        )

    def setup_abstract_mdp_(self, t, r, state_rewards=False):

        t = np.transpose(t, axes=[1, 2, 0])

        if self.prune_abstraction or self.prune_state_abstraction_new_means:
            self.prune_state_abstraction_(t)

        self.abstract_mdp = QuotientMDPFull(
            list(range(self.num_components)), list(range(self.num_actions)), t, r, discount=self.discount,
            state_rewards=state_rewards
        )

        self.abstract_mdp.value_iteration()

    def plot_abstract_mdp_planning_(self, new_transitions=False):

        name = "q_values_value_iteration"
        if new_transitions:
            name += "new_t"

        q_values = self.abstract_mdp.state_action_values

        vis_utils.plot_many_histograms(
            [q_values.reshape(-1)], [name], xlabel="q_values", num_bins=50, saver=self.saver
        )

    def run_abstract_agent_(self, q_values, append_name=None, check_x_y=None):

        def classify(state):

            depth = state[0][:, :, 0]
            if not self.disable_resize:
                depth = self.model.session.run(self.resized_t, feed_dict={
                    self.images_pl: depth[np.newaxis]
                })[0]
            depth = (depth - self.means) / self.stds

            _, log_probs = self.model.encode(np.array([depth]), hand_states=np.array([[state[1]]]))
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

        self.game_env = ContinuousPuckStack(
            self.grid_size, self.num_pucks, self.num_pucks, self.grid_size, render_always=True, height_noise_std=0.05
        )
        if check_x_y is not None:
            self.game_env.check_x_y = self.game_env.action_grid[check_x_y]

        rewards_name = "rewards"
        gifs_name = "gifs"

        if append_name is not None:
            rewards_name += "_" + append_name
            gifs_name += "_" + append_name

        self.abstract_agent = QuotientMDPNBisim(
            classify, self.game_env, q_values, minatar=False, softmax_policy=self.softmax_policy,
            softmax_policy_temp=self.softmax_policy_temp
        )

        if self.save_gifs:
            gifs_path = self.saver.get_new_dir(gifs_name)
        else:
            gifs_path = None

        solver = Solver(
            self.game_env, self.abstract_agent, int(1e+7), int(1e+7), 0, max_episodes=1000, train=False,
            gif_save_path=gifs_path, rewards_file=self.saver.get_save_file(rewards_name, "dat"),
            verbose=False
        )
        solver.run()

        self.logger.info("{:s}: {:.2f}".format(rewards_name, np.mean(solver.episode_rewards)))

    def predict_qs_from_clusters_hard_assignment_(self):

        qs = []

        for i in range(self.num_components):

            mask = self.train_cluster_assignment == i
            q = np.mean(self.dataset[constants.Q_VALUES][mask], axis=0)
            qs.append(q)

        qs = np.stack(qs, axis=0)

        assert qs.shape == (self.num_components, self.num_actions)

        predicted_qs = qs[self.valid_cluster_assignment]

        loss = np.mean(np.sum((predicted_qs - self.valid_dataset[constants.Q_VALUES]) ** 2, axis=1), axis=0)

        self.logger.info("cluster q-values prediction loss: {:.6f}".format(loss))

    def predict_next_clusters_hard_assignment_(self):

        accuracy = np.mean(self.valid_next_cluster_assignment == self.valid_predicted_cluster_assignment)
        self.logger.info("cluster transition accuracy: {:.2f}%".format(accuracy * 100))

    def plot_post_hmm_training_losses_(self):

        if len(self.post_train_prior_losses) > 0:
            fig = plt.figure()
            ax = fig.gca()

            ax.plot(
                list(range(1, len(self.post_train_prior_losses) + 1)), self.post_train_prior_losses[:, 0],
                label="prior log likelihood"
            )

            ax.legend()
            ax.set_xlabel("epoch")

            self.saver.save_figure(fig, "post_train_prior_losses")

    def evaluate_all_positions_(self):

        for x in range(self.grid_size):
            for y in range(self.grid_size):

                goal_depths, goal_hand_states = self.generate_target_states_(x, y, 25)
                state_log_cluster_probs = self.encode_target_states_(goal_depths, goal_hand_states)
                target_rewards = self.rewards_for_target_states_(state_log_cluster_probs)

                self.plot_examples_of_target_states_(x, y, goal_depths)
                self.plot_encoded_target_states_(x, y, state_log_cluster_probs)
                self.log_rewards_for_target_states_(target_rewards)

                self.setup_abstract_mdp_(self.t_softmax, target_rewards, state_rewards=True)
                self.run_abstract_agent_(
                    self.abstract_mdp.state_action_values, append_name="{:d}_{:d}_goal".format(x, y),
                    check_x_y=(x, y)
                )

    def generate_target_states_(self, x, y, num):

        assert self.num_pucks in [2, 3]

        goal_depths = []
        goal_hand_states = []

        env = ContinuousPuckStack(
            self.grid_size, self.num_pucks, self.num_pucks, self.grid_size, render_always=True, height_noise_std=0.05
        )

        for i in range(num):

            x_env, y_env = env.action_grid[x, y]

            env.reset()
            pucks = env.layers[0]

            for puck in pucks:
                puck.x = x_env
                puck.y = y_env

            if self.num_pucks == 2:
                env.layers[1].append(pucks[1])
                del env.layers[0][1]
            else:
                env.layers[1].append(pucks[1])
                env.layers[2].append(pucks[2])
                del env.layers[0][2]
                del env.layers[0][1]

            env.render()

            goal_depths.append(cp.deepcopy(env.depth))
            goal_hand_states.append(int(env.hand is None))

        goal_depths = np.array(goal_depths)
        goal_hand_states = np.array(goal_hand_states)

        return goal_depths, goal_hand_states

    def plot_examples_of_target_states_(self, x, y, goal_depths, num_examples=25):

        images = goal_depths[:num_examples]
        name = "goal_examples_{:d}_{:d}".format(x, y)
        vis_utils.plot_5x5_grid(images, name, self.saver)

    def encode_target_states_(self, goal_depths, goal_hand_states):

        if not self.disable_resize:
            goal_depths = self.model.session.run(self.resized_t, feed_dict={
                self.images_pl: goal_depths
            })

        _, state_log_cluster_probs = self.model.encode(
            (goal_depths - self.means) / self.stds, goal_hand_states[:, np.newaxis],
            zero_sd=self.zero_sd_after_training
        )

        return state_log_cluster_probs

    def plot_encoded_target_states_(self, x, y, state_log_cluster_probs):

        state_cluster_probs = np.exp(state_log_cluster_probs)
        name = "goal_cluster_probs_{:d}_{:d}".format(x, y)
        vis_utils.plot_soft_histogram(state_cluster_probs, name, self.saver, xlabel="cluster index")

    def rewards_for_target_states_(self, goal_log_cluster_probs):

        if self.soft_picture_goals:
            return np.mean(np.exp(goal_log_cluster_probs), axis=0)
        else:
            sums = np.sum(goal_log_cluster_probs, axis=0)
            argmax = np.argmax(sums, axis=0)
            return np.eye(sums.shape[0])[argmax]

    def log_rewards_for_target_states_(self, rewards):

        self.logger.info("goal rewards:" + str(rewards))

    def get_feed_dict_(self, epoch_step):

        b = np.index_exp[epoch_step * self.batch_size:(epoch_step + 1) * self.batch_size]

        feed_dict = {
            self.model.states_pl: self.dataset[constants.STATES][b],
            self.model.next_states_pl: self.dataset[constants.NEXT_STATES][b],
            self.model.hand_states_pl: self.dataset[constants.HAND_STATES][b][:, np.newaxis],
            self.model.next_hand_states_pl: self.dataset[constants.NEXT_HAND_STATES][b][:, np.newaxis],
            self.model.actions_pl: self.dataset[constants.ACTIONS][b],
            self.model.q_values_pl: self.dataset[constants.Q_VALUES][b],
            self.model.dones_pl: self.dataset[constants.DONES][b],
            self.model.is_training_pl: True
        }

        return feed_dict

    def prepare_model_(self):

        self.model = QPredictionModelHMMPrior(
            [self.depth_size, self.depth_size], self.num_blocks, self.num_components, self.num_actions,
            [32, 64, 128, 256], [4, 4, 4, 4], [2, 2, 2, 2], self.hiddens, self.encoder_learning_rate,
            self.beta0, self.beta1, self.beta2, self.weight_decay, self.encoder_optimizer, self.num_steps,
            batch_norm=not self.disable_batch_norm, softplus=not self.disable_softplus, no_sample=self.no_sample,
            only_one_q_value=self.only_one_q_value, beta3=self.beta3, cluster_predict_qs=self.cluster_predict_qs,
            cluster_predict_qs_weight=self.cluster_predict_qs_weight, fix_prior_training=self.fix_prior_training,
            learning_rate_steps=self.learning_rate_steps, learning_rate_values=self.learning_rate_values,
            model_learning_rate=self.model_learning_rate
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
