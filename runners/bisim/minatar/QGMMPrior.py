import pickle
import numpy as np
import tensorflow as tf
import constants
from model.bisim.bisim import QPredictionModelGMMPrior
from runners.bisim.puck_stack.QGMMPrior import QGMMPriorRunner
from utils.dataset import prepare_minatar_dataset
import vis_utils
import seaborn as sns
import matplotlib.colors as colors
from minatar import Environment
from agents.solver_minatar import SolverMinAtar
from agents.quotient_mdp_nbisim import QuotientMDPNBisim


class QGMMPriorRunnerMinAtar(QGMMPriorRunner):

    def __init__(self, load_path, game, num_actions, logger, saver, num_blocks, num_components,
                 encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
                 disable_batch_norm=False, disable_softplus=False, no_sample=False, only_one_q_value=True,
                 gt_q_values=False, validation_fraction=0.2, validation_freq=1000, summaries=None,
                 load_model_path=None, batch_size=100, zero_sd_after_training=False, hard_abstract_state=False,
                 save_gifs=False):

        super(QGMMPriorRunnerMinAtar, self).__init__(
            load_path, 1, None, logger, saver, num_blocks, num_components, None,
            encoder_learning_rate, beta0, beta1, beta2, weight_decay, encoder_optimizer, num_steps,
            disable_batch_norm=disable_batch_norm, disable_softplus=disable_softplus, no_sample=no_sample,
            only_one_q_value=only_one_q_value, gt_q_values=gt_q_values, disable_resize=True,
            oversample=False, validation_fraction=validation_fraction, validation_freq=validation_freq,
            summaries=summaries, load_model_path=load_model_path, batch_size=batch_size,
            zero_sd_after_training=zero_sd_after_training, hard_abstract_state=hard_abstract_state,
            save_gifs=save_gifs
        )

        self.game = game
        self.num_actions = num_actions

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

        self.dataset = prepare_minatar_dataset(transitions)

        vis_utils.plot_many_histograms(
            [self.dataset[constants.Q_VALUES].reshape(-1), self.dataset[constants.NEXT_Q_VALUES].reshape(-1)],
            ["q_values_hist", "next_q_values_hist"], xlabel="q-values",
            num_bins=50, saver=self.saver
        )

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

    def run_abstract_agent_(self):

        def classify(state):

            _, log_probs = self.model.encode(
                state[np.newaxis], hand_states=np.zeros((1, 1), dtype=np.int32)
            )
            log_probs = log_probs[0]

            if self.hard_abstract_state:
                idx = np.argmax(log_probs)
                output = np.zeros_like(log_probs)
                output[idx] = 1.0
            else:
                output = np.exp(log_probs)

            return output

        self.game_env = Environment(
            self.game, difficulty_ramping=False, sticky_action_prob=0.0
        )

        q_values = self.cluster_q_values
        rewards_name = "rewards_q_values"
        gifs_name = "gifs_q_values"

        self.abstract_agent = QuotientMDPNBisim(
            classify, self.game_env, q_values, minatar=True
        )

        if self.save_gifs:
            gifs_path = self.saver.get_new_dir(gifs_name)
        else:
            gifs_path = None

        solver = SolverMinAtar(
            self.game_env, self.abstract_agent, int(1e+7), int(1e+7), 0, max_episodes=100, train=False,
            gif_save_path=gifs_path, rewards_file=self.saver.get_save_file(rewards_name, "dat")
        )
        solver.run()

    def prepare_model_(self):

        self.model = QPredictionModelGMMPrior(
            [10, 10, 4], self.num_blocks, self.num_components, self.num_actions, [32, 64], [3, 3], [1, 1], [128],
            self.encoder_learning_rate, self.beta0, self.beta1, self.beta2, self.weight_decay, self.encoder_optimizer,
            self.num_steps, batch_norm=not self.disable_batch_norm, softplus=not self.disable_softplus,
            no_sample=self.no_sample, only_one_q_value=self.only_one_q_value,
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
