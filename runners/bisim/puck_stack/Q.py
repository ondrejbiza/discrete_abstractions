import pickle
import constants
import numpy as np
import tensorflow as tf
from model.bisim.bisim import QPredictionModel
from runners.runner import Runner
from utils.dataset import prepare_bisim_dataset_with_q_values
import model.utils as model_utils
import vis_utils


class QRunner(Runner):

    RESIZE_SIZE = 64

    def __init__(self, load_path, grid_size, num_pucks, logger, saver, num_blocks, hiddens, encoder_learning_rate,
                 weight_decay, encoder_optimizer, num_steps, disable_batch_norm=False, disable_softplus=False,
                 no_sample=False, only_one_q_value=True, gt_q_values=False, disable_resize=False, oversample=False,
                 validation_fraction=0.2, validation_freq=1000, summaries=None, load_model_path=None, batch_size=100,
                 zero_sd_after_training=False, include_goal_states=False, discount=0.9, shift_q_values=False,
                 show_qs=False, q_values_noise_sd=None, gt_q_values_all_actions=False, new_dones=False):

        super(QRunner, self).__init__()

        self.load_path = load_path
        self.grid_size = grid_size
        self.num_pucks = num_pucks
        self.logger = logger
        self.saver = saver
        self.num_blocks = num_blocks
        self.hiddens = hiddens
        self.encoder_learning_rate = encoder_learning_rate
        self.weight_decay = weight_decay
        self.encoder_optimizer = encoder_optimizer
        self.num_steps = num_steps
        self.disable_batch_norm = disable_batch_norm
        self.disable_softplus = disable_softplus
        self.no_sample = no_sample
        self.only_one_q_value = only_one_q_value
        self.gt_q_values = gt_q_values
        self.disable_resize = disable_resize
        self.oversample = oversample
        self.validation_fraction = validation_fraction
        self.validation_freq = validation_freq
        self.summaries = summaries
        self.load_model_path = load_model_path
        self.batch_size = batch_size
        self.zero_sd_after_training = zero_sd_after_training
        self.include_goal_states = include_goal_states
        self.discount = discount
        self.shift_q_values = shift_q_values
        self.show_qs = show_qs
        self.q_values_noise_sd = q_values_noise_sd
        self.gt_q_values_all_actions = gt_q_values_all_actions
        self.new_dones = new_dones

        self.num_actions = grid_size ** 2
        self.tf_summaries = None
        self.writer = None
        self.images_pl = None
        self.resized_t = None

    def setup(self):

        self.prepare_dataset_()
        self.prepare_model_()

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

                to_add = [bundle[constants.TOTAL_LOSS], bundle[constants.Q_LOSS]]

                self.add_step_losses_(to_add, train_step)

                if self.writer is not None:
                    self.writer.add_summary(bundle[constants.SUMMARY], global_step=train_step)

            if len(self.tmp_epoch_losses) > 0:
                self.add_epoch_losses_()

            self.postprocess_losses_()
            self.plot_losses_()

    def evaluate_and_visualize(self):

        self.inference_()
        self.plot_latent_space_()

    def inference_(self):

        # get current and next state embeddings and clusters
        self.train_embeddings = self.model.encode(
            self.dataset[constants.STATES], self.dataset[constants.HAND_STATES][:, np.newaxis],
            zero_sd=self.zero_sd_after_training
        )

        self.valid_embeddings = self.model.encode(
            self.valid_dataset[constants.STATES], self.valid_dataset[constants.HAND_STATES][:, np.newaxis],
            zero_sd=self.zero_sd_after_training
        )

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

    def plot_latent_space_(self):

        fig = model_utils.transform_and_plot_embeddings(
            self.valid_embeddings, self.valid_dataset[constants.STATE_LABELS]
        )
        self.saver.save_figure(fig, "embeddings")

    def get_feed_dict_(self, epoch_step):

        b = np.index_exp[epoch_step * self.batch_size:(epoch_step + 1) * self.batch_size]

        feed_dict = {
            self.model.states_pl: self.dataset[constants.STATES][b],
            self.model.hand_states_pl: self.dataset[constants.HAND_STATES][b][:, np.newaxis],
            self.model.actions_pl: self.dataset[constants.ACTIONS][b],
            self.model.q_values_pl: self.dataset[constants.Q_VALUES][b],
            self.model.is_training_pl: True
        }

        return feed_dict

    def prepare_dataset_(self):

        with open(self.load_path, "rb") as file:
            transitions = pickle.load(file)

        self.dataset = prepare_bisim_dataset_with_q_values(
            transitions, self.grid_size, self.num_pucks, gt_q_values=self.gt_q_values,
            gt_q_values_all_actions=self.gt_q_values_all_actions,
            only_one_q_value=self.only_one_q_value, include_goal_state=self.include_goal_states,
            shift_qs_by=self.discount if self.shift_q_values else None
        )

        if self.q_values_noise_sd is not None:
            self.dataset[constants.Q_VALUES] += np.random.normal(
                loc=0, scale=self.q_values_noise_sd, size=self.dataset[constants.Q_VALUES].shape
            )
            self.dataset[constants.NEXT_Q_VALUES] += np.random.normal(
                loc=0, scale=self.q_values_noise_sd, size=self.dataset[constants.NEXT_Q_VALUES].shape
            )

        if self.new_dones:
            self.dataset[constants.DONES] = self.dataset[constants.REWARDS] > 0.0

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
        self.train_not_dones = np.logical_not(self.dataset[constants.DONES])
        self.valid_not_dones = np.logical_not(self.valid_dataset[constants.DONES])

        self.log_dataset_stats_()

        del transitions

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

        self.logger.info("state mean {:.2f} std {:.2f}".format(np.mean(self.means), np.mean(self.stds)))

    def prepare_model_(self):

        self.model = QPredictionModel(
            [self.depth_size, self.depth_size], self.num_blocks, self.num_actions, [32, 64, 128, 256], [4, 4, 4, 4],
            [2, 2, 2, 2], self.hiddens, self.encoder_learning_rate, self.weight_decay, self.encoder_optimizer,
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
