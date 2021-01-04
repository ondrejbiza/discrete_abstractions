import evaluate
import vis_utils
from runners.bisim.puck_stack.Q import QRunner
import numpy as np
import tensorflow as tf
from model.bisim.bisim import QPredictionModelGMMPrior
import model.utils as model_utils
import constants
from model.cluster import Cluster
from agents.solver import Solver
from agents.quotient_mdp_nbisim import QuotientMDPNBisim
from envs.continuous_puck_stack import ContinuousPuckStack


class QGMMPriorRunner(QRunner):

    def __init__(self, load_path, grid_size, num_pucks, logger, saver, num_blocks, num_components, hiddens,
                 encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
                 disable_batch_norm=False, disable_softplus=False, no_sample=False, only_one_q_value=True,
                 gt_q_values=False, disable_resize=False, oversample=False, validation_fraction=0.2,
                 validation_freq=1000, summaries=None, load_model_path=None, batch_size=100,
                 zero_sd_after_training=False, hard_abstract_state=False, save_gifs=False,
                 include_goal_states=False, discount=0.9, shift_q_values=False, show_qs=False,
                 q_values_noise_sd=None, gt_q_values_all_actions=False, new_dones=False):

        super(QGMMPriorRunner, self).__init__(
            load_path, grid_size, num_pucks, logger, saver, num_blocks, hiddens, encoder_learning_rate,
            weight_decay, encoder_optimizer, num_steps, disable_batch_norm=disable_batch_norm,
            disable_softplus=disable_softplus, no_sample=no_sample, only_one_q_value=only_one_q_value,
            gt_q_values=gt_q_values, disable_resize=disable_resize, oversample=oversample,
            validation_fraction=validation_fraction, validation_freq=validation_freq, summaries=summaries,
            load_model_path=load_model_path, batch_size=batch_size, zero_sd_after_training=zero_sd_after_training,
            include_goal_states=include_goal_states, discount=discount, shift_q_values=shift_q_values,
            show_qs=show_qs, q_values_noise_sd=q_values_noise_sd, gt_q_values_all_actions=gt_q_values_all_actions,
            new_dones=new_dones
        )

        self.num_components = num_components
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.hard_abstract_state = hard_abstract_state
        self.save_gifs = save_gifs

        self.means = None
        self.stds = None

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
                        self.valid_dataset[constants.ACTIONS], self.valid_dataset[constants.Q_VALUES]
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

    def evaluate_and_visualize(self):

        self.inference_()
        self.setup_clusters_()
        self.evaluate_cluster_purities_()
        self.plot_latent_space_()
        self.plot_latent_space_with_centroids_()
        self.plot_perplexities_()
        #self.plot_state_components_examples_()
        #self.predict_qs_from_clusters_hard_assignment_()

        self.get_cluster_q_values_()
        self.plot_cluster_q_values_()
        self.run_abstract_agent_()

    def inference_(self):

        self.mixtures_mu, self.mixtures_logvar = self.model.session.run(
            [self.model.mixtures_mu_v, self.model.mixtures_logvar_v]
        )

        # get current and next state embeddings and clusters
        self.train_embeddings, self.train_cluster_log_probs = self.model.encode(
            self.dataset[constants.STATES], self.dataset[constants.HAND_STATES][:, np.newaxis],
            zero_sd=self.zero_sd_after_training
        )

        self.valid_embeddings, self.valid_cluster_log_probs = self.model.encode(
            self.valid_dataset[constants.STATES], self.valid_dataset[constants.HAND_STATES][:, np.newaxis],
            zero_sd=self.zero_sd_after_training
        )
        self.train_cluster_assignment = np.argmax(self.train_cluster_log_probs, axis=1)
        self.valid_cluster_assignment = np.argmax(self.valid_cluster_log_probs, axis=1)

    def plot_losses_(self):

        train_indices = [1]
        valid_indices = [0]
        train_labels = ["train q loss"]
        valid_labels = ["valid q loss"]

        if len(self.per_epoch_losses) > 0:
            vis_utils.plot_epoch_losses(
                self.per_epoch_losses, self.valid_losses, train_indices, valid_indices, train_labels,
                valid_labels, self.epoch_size, self.validation_freq, saver=self.saver, save_name="epoch_losses"
            )

        if len(self.all_losses) > 0:
            vis_utils.plot_all_losses(
                self.all_losses, self.valid_losses, train_indices, valid_indices, train_labels,
                valid_labels, self.validation_freq, saver=self.saver, save_name="losses"
            )

            vis_utils.plot_all_losses(
                self.all_losses, self.valid_losses, [2, 3], [1, 2],
                ["train prior log likelihood", "train encoder entropy"],
                ["valid prior log likelihood", "valid encoder entropy"],
                self.validation_freq, saver=self.saver, save_name="likelihood_and_entropy", log_scale=False
            )

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

    def plot_latent_space_with_centroids_(self):

        tmp_embeddings = np.concatenate([self.valid_embeddings, self.mixtures_mu], axis=0)
        tmp_labels = np.concatenate(
            [self.valid_dataset[constants.STATE_LABELS],
             np.zeros(len(self.mixtures_mu), dtype=np.int32) +
             np.max(self.valid_dataset[constants.STATE_LABELS]) + 1]
        )

        fig = model_utils.transform_and_plot_embeddings(tmp_embeddings, tmp_labels)
        self.saver.save_figure(fig, "embeddings_and_centroids")

    def plot_perplexities_(self):

        train_perplexities = evaluate.get_perplexities(self.train_cluster_log_probs)
        valid_perplexities = evaluate.get_perplexities(self.valid_cluster_log_probs)

        vis_utils.plot_many_histograms(
            [train_perplexities, valid_perplexities],
            ["train_state_perplexities", "valid_state_perplexities"],
            xlabel="perplexity", num_bins=50, saver=self.saver
        )

    def plot_state_components_examples_(self):

        hard_assignment = np.argmax(self.valid_cluster_log_probs, axis=1)

        for cluster_idx in range(self.num_components):

            mask = hard_assignment == cluster_idx

            if np.sum(mask) < 25:
                continue

            name = "state_cluster_{:d}".format(cluster_idx)
            images = self.valid_dataset[constants.STATES][mask][:25]
            if self.means is not None and self.stds is not None:
                images = images * self.stds + self.means

            vis_utils.plot_5x5_grid(images, name, self.saver)

    def setup_clusters_(self):

        self.cluster = Cluster(self.dataset, self.valid_dataset)
        self.cluster.num_clusters = self.num_components

        self.train_assigned_labels = self.cluster.assign_label_to_each_cluster(
            self.train_cluster_assignment, validation=False
        )
        self.train_assigned_labels_soft = self.cluster.assign_label_to_each_cluster_soft(
            np.exp(self.train_cluster_log_probs), validation=False
        )
        self.valid_assigned_labels = self.cluster.assign_label_to_each_cluster(
            self.valid_cluster_assignment, validation=True
        )
        self.valid_assigned_labels_soft = self.cluster.assign_label_to_each_cluster_soft(
            np.exp(self.valid_cluster_log_probs), validation=True
        )

    def evaluate_cluster_purities_(self):

        cluster_purities, cluster_sizes, mean_purity = self.cluster.evaluate_purity_soft(
            np.exp(self.valid_cluster_log_probs)
        )
        self.logger.info("state mixture soft purity: {:.2f}%".format(mean_purity * 100))

        cluster_mean_inverse_purity, state_cluster_inverse_purities, state_cluster_totals = \
            self.cluster.evaluate_inverse_purity_soft(
                np.exp(self.valid_cluster_log_probs), self.train_assigned_labels_soft
            )
        self.logger.info("state mixture inverse soft purity: {:.2f}%".format(cluster_mean_inverse_purity * 100))
        self.logger.info("cluster sizes:" + str(cluster_sizes))

        self.saver.save_array_as_txt([mean_purity, cluster_mean_inverse_purity], "cluster_purity")

    def get_cluster_q_values_(self):

        self.cluster_q_values = []

        for i in range(self.num_components):

            mask = self.train_cluster_assignment == i

            if np.sum(mask) > 0:
                q_values = np.mean(self.dataset[constants.Q_VALUES][mask], axis=0)
            else:
                q_values = np.zeros(self.num_actions, dtype=np.float32)

            self.cluster_q_values.append(q_values)

        self.cluster_q_values = np.array(self.cluster_q_values)

    def plot_cluster_q_values_(self):

        print(self.cluster_q_values)

        vis_utils.plot_many_histograms(
            [self.cluster_q_values.reshape(-1)],
            ["cluster_q_values"], xlabel="bins",
            num_bins=50, saver=self.saver
        )

    def run_abstract_agent_(self):

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
            else:
                output = np.exp(log_probs)

            return output

        self.game_env = ContinuousPuckStack(
            self.grid_size, self.num_pucks, self.num_pucks, self.grid_size, render_always=True, height_noise_std=0.05
        )

        q_values = self.cluster_q_values
        rewards_name = "rewards_q_values"
        gifs_name = "gifs_q_values"

        self.abstract_agent = QuotientMDPNBisim(
            classify, self.game_env, q_values, minatar=False
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

    def prepare_model_(self):

        self.model = QPredictionModelGMMPrior(
            [self.depth_size, self.depth_size], self.num_blocks, self.num_components, self.num_actions,
            [32, 64, 128, 256], [4, 4, 4, 4], [2, 2, 2, 2], self.hiddens, self.encoder_learning_rate,
            self.beta0, self.beta1, self.beta2, self.weight_decay, self.encoder_optimizer, self.num_steps,
            batch_norm=not self.disable_batch_norm, softplus=not self.disable_softplus, no_sample=self.no_sample,
            only_one_q_value=self.only_one_q_value,
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
