import pickle
import numpy as np
import tensorflow as tf
from model.bisim.QPredictionModelHMMPriorMultiTask import QPredictionModelHMMPriorMultiTask
from runners.bisim.puck_stack.QHMMPrior import QHMMPriorRunner
from utils.dataset import prepare_bisim_dataset_with_q_values
from envs.continuous_multi_task import ContinuousMultiTask, CheckGoalFactory
import constants
import vis_utils
from agents.quotient_mdp_nbisim import QuotientMDPNBisim
from agents.solver import Solver


class QHMMPriorMultiTask(QHMMPriorRunner):

    def __init__(self, load_path, grid_size, num_pucks, logger, saver, num_blocks, num_components, hiddens,
                 encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
                 active_task_indices, evaluation_task_indices, dones_index,
                 disable_batch_norm=False, disable_softplus=False, no_sample=False, only_one_q_value=True,
                 gt_q_values=False, disable_resize=False, oversample=False, validation_fraction=0.2,
                 validation_freq=1000, summaries=None, load_model_path=None, batch_size=100, beta3=0.0,
                 post_train_hmm=False, post_train_prior=False, post_train_t_and_prior=False,
                 zero_sd_after_training=False, cluster_predict_qs=False, cluster_predict_qs_weight=0.1,
                 hard_abstract_state=False, save_gifs=False, prune_abstraction=False,
                 prune_threshold=0.01, prune_abstraction_new_means=False, old_bn_settings=False,
                 sample_abstract_state=False, softmax_policy=False, softmax_policy_temp=1.0,
                 include_goal_states=False, discount=0.9, soft_picture_goals=False,
                 shift_q_values=False, goal_rewards_threshold=None, q_values_indices=None,
                 fast_eval=False, show_qs=False, fix_prior_training=False, model_learning_rate=None,
                 random_shape=False):

        super(QHMMPriorMultiTask, self).__init__(
            load_path, grid_size, num_pucks, logger, saver, num_blocks, num_components, hiddens,
            encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
            disable_batch_norm=disable_batch_norm, disable_softplus=disable_softplus, no_sample=no_sample,
            only_one_q_value=only_one_q_value, gt_q_values=gt_q_values, disable_resize=disable_resize,
            oversample=oversample, validation_fraction=validation_fraction, validation_freq=validation_freq,
            summaries=summaries, load_model_path=load_model_path, batch_size=batch_size, beta3=beta3,
            post_train_hmm=post_train_hmm, post_train_prior=post_train_prior,
            post_train_t_and_prior=post_train_t_and_prior, zero_sd_after_training=zero_sd_after_training,
            cluster_predict_qs=cluster_predict_qs, cluster_predict_qs_weight=cluster_predict_qs_weight,
            hard_abstract_state=hard_abstract_state, save_gifs=save_gifs, prune_abstraction=prune_abstraction,
            prune_threshold=prune_threshold, prune_abstraction_new_means=prune_abstraction_new_means,
            old_bn_settings=old_bn_settings, sample_abstract_state=sample_abstract_state, softmax_policy=softmax_policy,
            softmax_policy_temp=softmax_policy_temp, include_goal_states=include_goal_states, discount=discount,
            soft_picture_goals=soft_picture_goals, shift_q_values=shift_q_values, show_qs=show_qs,
            fix_prior_training=fix_prior_training, model_learning_rate=model_learning_rate
        )

        self.active_task_indices = active_task_indices
        self.evaluation_task_indices = evaluation_task_indices
        self.dones_index = dones_index
        self.goal_rewards_threshold = goal_rewards_threshold
        self.q_values_indices = q_values_indices
        self.fast_eval = fast_eval
        self.random_shape = random_shape

        self.num_tasks = len(self.active_task_indices)

    def main_training_loop(self):

        self.prepare_losses_()

        if self.load_model_path is None:

            num_steps = self.num_steps + 10000

            for train_step in range(num_steps):

                if train_step == num_steps - 10000:
                    self.to_run[constants.TRAIN_STEP] = self.model.update_op

                if train_step % self.validation_freq == 0:

                    tmp_valid_losses = self.model.validate(
                        self.valid_dataset[constants.STATES], self.valid_dataset[constants.HAND_STATES],
                        self.valid_dataset[constants.ACTIONS], self.valid_dataset[constants.Q_VALUES],
                        self.valid_dataset[constants.NEXT_STATES], self.valid_dataset[constants.NEXT_HAND_STATES],
                        self.valid_dataset[constants.NEW_DONES]
                    )

                    self.valid_losses.append(np.mean(tmp_valid_losses, axis=0))

                epoch_step = train_step % self.epoch_size

                if train_step > 0 and epoch_step == 0:

                    self.logger.info("training step {:d}".format(train_step))

                    self.dataset.shuffle()

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

        to_run = { constants.PRIOR_LOG_LIKELIHOOD: self.model.prior_log_likelihood}

        if self.post_train_hmm:
            to_run[constants.TRAIN_STEP] = self.model.hmm_train_step
        elif self.post_train_t_and_prior:
            to_run[constants.TRAIN_STEP] = self.model.T_and_prior_train_step
        else:
            to_run[constants.TRAIN_STEP] = self.model.prior_train_step

        for train_step in range(training_steps):

            epoch_step = train_step % self.epoch_size

            if epoch_step == 0:
                self.dataset.shuffle()
                self.logger.info("HMM prior training step {:d}".format(train_step))

            feed_dict = self.get_feed_dict_(epoch_step)
            bundle = self.model.session.run(to_run, feed_dict=feed_dict)

            self.post_train_prior_losses.append([bundle[constants.PRIOR_LOG_LIKELIHOOD]])

        self.post_train_prior_losses = np.array(self.post_train_prior_losses)
        self.plot_post_hmm_training_losses_()

    def evaluate_and_visualize(self):

        self.inference_()
        self.setup_clusters_()

        if not self.fast_eval:
            self.evaluate_cluster_purities_()
            #self.evaluate_transitions_method_2_()
            #self.evaluate_transitions_method_3_()
            self.plot_latent_space_()
            self.plot_latent_space_with_centroids_()
            self.plot_perplexities_()
            #self.plot_state_components_examples_()

        self.evaluate_tasks_()

    def evaluate_tasks_(self):

        for task_idx in self.active_task_indices:

            cluster_rewards = self.get_cluster_rewards_for_task_(task_idx)
            self.plot_cluster_rewards_(cluster_rewards, name="cluster_rewards_task_{:d}_active".format(task_idx))

            self.setup_abstract_mdp_(self.t_softmax, cluster_rewards, state_rewards=False)
            self.run_abstract_agent_(
                self.abstract_mdp.state_action_values, task_idx, append_name="task_{:d}_active".format(task_idx)
            )

        for task_idx in self.evaluation_task_indices:

            cluster_rewards = self.get_cluster_rewards_for_task_(task_idx)
            self.plot_cluster_rewards_(cluster_rewards, name="cluster_rewards_task_{:d}_inactive".format(task_idx))

            self.setup_abstract_mdp_(self.t_softmax, cluster_rewards, state_rewards=False)
            self.run_abstract_agent_(
                self.abstract_mdp.state_action_values, task_idx, append_name="task_{:d}_inactive".format(task_idx)
            )

    def get_cluster_rewards_for_task_(self, task_idx):

        cluster_rewards = np.zeros((self.num_components, self.num_actions), dtype=np.float32)

        for cluster_idx in range(self.num_components):
            for action_idx in range(self.num_actions):
                # we execute action 'action_idx' in abstract state 'cluster idx'
                mask = np.bitwise_and(self.train_cluster_assignment == cluster_idx,
                                      self.dataset[constants.ACTIONS] == action_idx)
                # average rewards if such abstract-state-action pair exists
                if np.sum(mask) > 0:
                    cluster_rewards[cluster_idx, action_idx] = np.mean(
                        self.dataset[constants.REWARDS][mask][:, task_idx]
                    )

        if self.goal_rewards_threshold is not None:
            cluster_rewards[cluster_rewards < self.goal_rewards_threshold] = 0.0

        return cluster_rewards

    def plot_cluster_rewards_(self, cluster_rewards, name):

        vis_utils.plot_many_histograms(
            [cluster_rewards.reshape(-1)], [name], xlabel="bins", num_bins=50, saver=self.saver
        )

    def run_abstract_agent_(self, q_values, goal_idx, append_name=None):

        def classify(state):

            depth = state[0][:, :, 0]
            if not self.disable_resize:
                depth = self.model.session.run(self.resized_t, feed_dict={
                    self.images_pl: depth[np.newaxis]
                })[0]
            depth = (depth - self.means) / self.stds

            _, log_probs = self.model.encode(
                np.array([depth]), hand_states=np.array([[state[1]]]), zero_sd=self.zero_sd_after_training
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

        fcs = [
            CheckGoalFactory.check_goal_puck_stacking(2), CheckGoalFactory.check_goal_puck_stacking(3),
            CheckGoalFactory.check_goal_row(2), CheckGoalFactory.check_goal_row(3),
            CheckGoalFactory.check_goal_two_stacks(2), CheckGoalFactory.check_goal_stairs(3),
            CheckGoalFactory.check_goal_row_diag(2), CheckGoalFactory.check_goal_row_diag(3)
        ]
        fcs = [fcs[goal_idx]]

        self.game_env = ContinuousMultiTask(
            self.grid_size, self.num_pucks, fcs, self.grid_size, height_noise_std=0.05, render_always=True,
            only_one_goal=True, random_shape=self.random_shape
        )

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

    def show_dataset_examples_with_qs_(self):

        for i in range(100):

            state = self.dataset[constants.STATES][i]
            hand_state = self.dataset[constants.HAND_STATES][i]
            action = self.dataset[constants.ACTIONS][i]
            reward = self.dataset[constants.REWARDS][i]
            q_values = self.dataset[constants.Q_VALUES][i]

            if self.dones_index is not None:
                done = self.dataset[constants.NEW_DONES][i]
            else:
                done = self.dataset[constants.DONES][i]

            import matplotlib.pyplot as plt

            print("hand state: {:d}".format(hand_state))
            print("q values: " + str(q_values))
            print("action: {:d}".format(action))
            print("reward: " + str(reward))
            print("done: " + str(done))

            plt.subplot(1, 2, 1)
            plt.imshow(state)

            plt.subplot(1, 2, 2)
            plt.imshow(state * self.stds + self.means)

            plt.show()

    def prepare_dataset_(self):

        with open(self.load_path, "rb") as file:
            transitions = pickle.load(file)

        self.dataset = prepare_bisim_dataset_with_q_values(
            transitions, self.grid_size, self.num_pucks, gt_q_values=self.gt_q_values,
            only_one_q_value=self.only_one_q_value, include_goal_state=self.include_goal_states,
            shift_qs_by=self.discount if self.shift_q_values else None
        )

        self.dataset[constants.NEW_DONES] = self.dataset[constants.REWARDS][:, self.dones_index] > 0.0

        if self.q_values_indices is not None:
            self.dataset[constants.Q_VALUES] = self.dataset[constants.Q_VALUES][:, self.q_values_indices, :]
            self.dataset[constants.NEXT_Q_VALUES] = self.dataset[constants.NEXT_Q_VALUES][:, self.q_values_indices, :]

        vis_utils.plot_many_histograms(
            [self.dataset[constants.Q_VALUES].reshape(-1), self.dataset[constants.NEXT_Q_VALUES].reshape(-1)],
            ["q_values_hist", "next_q_values_hist"], xlabel="q-values",
            num_bins=50, saver=self.saver
        )

        vis_utils.plot_many_histograms(
            [self.dataset[constants.STATE_LABELS], self.dataset[constants.NEXT_STATE_LABELS]],
            ["state_labels_hist", "next_state_labels_hist"], xlabel="label index",
            num_bins=10, saver=self.saver
        )

        original_size = self.dataset[constants.STATES][0].shape[0]

        assert len(self.dataset[constants.STATES][0].shape) == 2
        assert original_size == self.dataset[constants.STATES][0].shape[1]

        if original_size == self.RESIZE_SIZE:
            self.disable_resize = True

        self.depth_size = self.RESIZE_SIZE
        if not self.disable_resize:
            self.images_pl, self.resized_t = \
                self.dataset.resize_tf(original_size, self.RESIZE_SIZE, [constants.STATES, constants.NEXT_STATES])

        self.logger.info("original depth size: {:d}, new size: {:d}".format(original_size, self.depth_size))

        if self.oversample:
            self.dataset.oversample((self.dataset[constants.REWARDS] > 0))

        self.means, self.stds = self.dataset.normalize_together(
            [constants.STATES, constants.NEXT_STATES], std_threshold=0.001
        )

        self.dataset.shuffle()

        num_samples = self.dataset.size
        valid_samples = int(num_samples * self.validation_fraction)
        self.valid_dataset = self.dataset.split(valid_samples)

        self.epoch_size = self.dataset[constants.STATES].shape[0] // self.batch_size

        # TODO: this doesn't get shuffled with the dataset--very bad
        self.train_not_dones = np.logical_not(self.dataset[constants.NEW_DONES])
        self.valid_not_dones = np.logical_not(self.valid_dataset[constants.NEW_DONES])

        self.log_dataset_stats_()

        if self.show_qs:
            self.show_dataset_examples_with_qs_()

    def get_feed_dict_(self, epoch_step):

        b = np.index_exp[epoch_step * self.batch_size:(epoch_step + 1) * self.batch_size]

        feed_dict = {
            self.model.states_pl: self.dataset[constants.STATES][b],
            self.model.next_states_pl: self.dataset[constants.NEXT_STATES][b],
            self.model.hand_states_pl: self.dataset[constants.HAND_STATES][b][:, np.newaxis],
            self.model.next_hand_states_pl: self.dataset[constants.NEXT_HAND_STATES][b][:, np.newaxis],
            self.model.actions_pl: self.dataset[constants.ACTIONS][b],
            self.model.q_values_pl: self.dataset[constants.Q_VALUES][b],
            self.model.dones_pl: self.dataset[constants.NEW_DONES][b],
            self.model.is_training_pl: True
        }

        return feed_dict

    def prepare_model_(self):

        self.model = QPredictionModelHMMPriorMultiTask(
            [self.depth_size, self.depth_size], self.num_blocks, self.num_components, self.num_actions,
            [32, 64, 128, 256], [4, 4, 4, 4], [2, 2, 2, 2], self.hiddens, self.encoder_learning_rate,
            self.beta0, self.beta1, self.beta2, self.weight_decay, self.encoder_optimizer, self.num_steps, self.num_tasks,
            batch_norm=not self.disable_batch_norm, softplus=not self.disable_softplus, no_sample=self.no_sample,
            only_one_q_value=self.only_one_q_value, beta3=self.beta3, cluster_predict_qs=self.cluster_predict_qs,
            cluster_predict_qs_weight=self.cluster_predict_qs_weight, fix_prior_training=self.fix_prior_training,
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
