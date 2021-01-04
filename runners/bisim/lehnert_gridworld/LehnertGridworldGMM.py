import numpy as np
import constants
import vis_utils
from model.LehnertGridworldModelGMM import LehnertGridworldModelGMM
import config_constants as cc
from runners.bisim.lehnert_gridworld.LehnertGridworldLatent import LehnertGridworldLatentRunner
from model.cluster import Cluster
from envs.lehnert_gridworld import LehnertGridworld
from utils.dataset import ArrayDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class LehnertGridworldGMMRunner(LehnertGridworldLatentRunner):

    def __init__(self, runner_config, model_config):

        super(LehnertGridworldGMMRunner, self).__init__(runner_config, model_config)

        self.num_components = model_config[cc.NUM_COMPONENTS]

    def evaluate_and_visualize(self):

        self.inference_()
        self.setup_clusters_()

        self.plot_latent_space_()
        self.plot_latent_space_with_centroids_()
        self.evaluate_cluster_purities_()
        self.evaluate_cluster_purities_all_states_()

    def inference_(self):

        self.mixtures_mu, self.mixtures_var = self.model.session.run(
            [self.model.mixtures_mu_v, self.model.mixtures_var_t]
        )

        self.train_embeddings, self.train_cluster_log_probs = self.model.encode(self.dataset[constants.STATES])
        self.valid_embeddings, self.valid_cluster_log_probs = self.model.encode(self.valid_dataset[constants.STATES])

        self.train_cluster_assignment = np.argmax(self.train_cluster_log_probs, axis=1)
        self.valid_cluster_assignment = np.argmax(self.valid_cluster_log_probs, axis=1)

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

    def evaluate_cluster_purities_all_states_(self):

        env = LehnertGridworld()
        all_states, all_state_labels = env.get_all_states()

        all_states = np.array(all_states)
        all_state_labels = np.array(all_state_labels)

        _, log_probs = self.model.encode(all_states)
        probs = np.exp(log_probs)

        dataset = ArrayDataset({
            constants.STATE_LABELS: all_state_labels
        })

        cluster = Cluster(None, dataset)
        cluster.num_clusters = self.num_components

        cluster_map = cluster.assign_label_to_each_cluster_soft(probs)

        purities, sizes, mean_purity = cluster.evaluate_purity_soft(probs)
        mean_i_purity, i_scores, i_totals = cluster.evaluate_inverse_purity_soft(probs, cluster_map)

        self.logger.info("all states mixture soft purity: {:.2f}%".format(mean_purity * 100))
        self.logger.info("all states mixture inverse soft purity: {:.2f}%".format(mean_i_purity * 100))

        self.saver.save_array_as_txt([mean_purity, mean_i_purity], "cluster_purity_all_states")

    def plot_latent_space_with_centroids_(self, step_idx=None, ext="pdf"):

        tmp_embeddings = np.concatenate([self.valid_embeddings, self.mixtures_mu], axis=0)

        pca = PCA(n_components=2)
        tmp_embeddings = pca.fit_transform(tmp_embeddings)
        explained = pca.explained_variance_ratio_

        proj_embeds = tmp_embeddings[:len(self.valid_embeddings)]
        proj_centroids = tmp_embeddings[len(self.valid_embeddings):]

        plt.style.use("seaborn-colorblind")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for l in np.unique(self.valid_dataset[constants.STATE_LABELS]):

            mask = self.valid_dataset[constants.STATE_LABELS] == l
            ax.scatter(proj_embeds[:, 0][mask], proj_embeds[:, 1][mask], label="ground-truth block {:d}".format(l + 1))

        ax.scatter(proj_centroids[:, 0], proj_centroids[:, 1], c="red", marker="x", label="cluster")

        plt.xlabel("PCA1: {:.1f}% variance explained".format(explained[0] * 100))
        plt.ylabel("PCA2: {:.1f}% variance explained".format(explained[1] * 100))
        plt.legend()

        name = "embeddings_and_centroids"
        if step_idx is not None:
            name += "_{:d}".format(step_idx)

        self.saver.save_figure(fig, name, ext=ext)

    def main_training_loop(self):

        self.prepare_losses_()

        if self.load_model_path is None:

            for train_step in range(self.num_steps):

                if train_step % self.validation_freq == 0:
                    tmp_valid_losses = self.model.validate(
                        self.valid_dataset[constants.STATES], self.valid_dataset[constants.ACTIONS],
                        self.valid_dataset[constants.Q_VALUES]
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

                to_add = [bundle[constants.TOTAL_LOSS], bundle[constants.Q_LOSS],
                          bundle[constants.ENTROPY_LOSS], bundle[constants.PRIOR_LOG_LIKELIHOOD]]

                self.add_step_losses_(to_add, train_step)

            if len(self.tmp_epoch_losses) > 0:
                self.add_epoch_losses_()

            self.postprocess_losses_()
            self.plot_losses_()

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
                ["train encoder entropy", "train prior log likelihood"],
                ["valid encoder entropy", "valid prior log likelihood"],
                self.validation_freq, saver=self.saver, save_name="prior_and_entropy", log_scale=False
            )

    def prepare_model_(self):

        self.model = LehnertGridworldModelGMM(self.model_config)
        self.model.build()
        self.model.start_session()

        if self.load_model_path is not None:
            self.model.load(self.load_model_path)

        self.to_run = {
            constants.TRAIN_STEP: self.model.train_step,
            constants.TOTAL_LOSS: self.model.loss_t,
            constants.Q_LOSS: self.model.q_loss_t,
            constants.ENTROPY_LOSS: self.model.encoder_entropy_t,
            constants.PRIOR_LOG_LIKELIHOOD: self.model.prior_log_likelihood_t
        }
