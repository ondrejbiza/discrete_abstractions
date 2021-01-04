import pickle
import numpy as np
import tensorflow as tf
import constants
from runners.bisim.puck_stack.Q import QRunner
from model.bisim.bisim import QPredictionModel
from utils.dataset import prepare_minatar_dataset
import vis_utils


class QRunnerMinAtar(QRunner):

    def __init__(self, load_path, num_actions, logger, saver, num_blocks, encoder_learning_rate,
                 weight_decay, encoder_optimizer, num_steps, disable_batch_norm=False, disable_softplus=False,
                 no_sample=False, only_one_q_value=True, gt_q_values=False, validation_fraction=0.2,
                 validation_freq=1000, summaries=None, load_model_path=None, batch_size=100,
                 zero_sd_after_training=False):

        super(QRunnerMinAtar, self).__init__(
            load_path, 1, None, logger, saver, num_blocks, None, encoder_learning_rate,
            weight_decay, encoder_optimizer, num_steps, disable_batch_norm=disable_batch_norm,
            disable_softplus=disable_softplus, no_sample=no_sample, only_one_q_value=only_one_q_value,
            gt_q_values=gt_q_values, disable_resize=True, oversample=False, validation_fraction=validation_fraction,
            validation_freq=validation_freq, summaries=summaries, load_model_path=load_model_path,
            batch_size=batch_size, zero_sd_after_training=zero_sd_after_training
        )

        self.num_actions = num_actions

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

    def prepare_model_(self):

        self.model = QPredictionModel(
            [10, 10, 4], self.num_blocks, self.num_actions, [32, 64], [3, 3],
            [1, 1], [128], self.encoder_learning_rate, self.weight_decay, self.encoder_optimizer,
            self.num_steps, batch_norm=not self.disable_batch_norm, softplus=not self.disable_softplus,
            no_sample=self.no_sample, only_one_q_value=self.only_one_q_value
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
            constants.Q_LOSS: self.model.q_loss_t
        }

        if self.summaries:
            self.to_run[constants.SUMMARY] = self.tf_summaries
