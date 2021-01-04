import pickle
import numpy as np
import tensorflow as tf
from utils.dataset import prepare_bisim_dataset_with_q_values
from model.bisim.QPredictionModelMultiTask import QPredictionModelMultiTask
from runners.bisim.puck_stack.Q import QRunner
import constants
import vis_utils


class QMultiTaskRunner(QRunner):

    def __init__(self, load_path, grid_size, num_pucks, logger, saver, num_blocks, hiddens, encoder_learning_rate,
                 weight_decay, encoder_optimizer, num_steps, active_task_indices, disable_batch_norm=False, disable_softplus=False,
                 no_sample=False, only_one_q_value=True, gt_q_values=False, disable_resize=False, oversample=False,
                 validation_fraction=0.2, validation_freq=1000, summaries=None, load_model_path=None, batch_size=100,
                 zero_sd_after_training=False, include_goal_states=False, discount=0.9, shift_q_values=False,
                 bn_momentum=0.99):

        super(QMultiTaskRunner, self).__init__(
            load_path, grid_size, num_pucks, logger, saver, num_blocks, hiddens, encoder_learning_rate,
            weight_decay, encoder_optimizer, num_steps, disable_batch_norm=disable_batch_norm,
            disable_softplus=disable_softplus, no_sample=no_sample, only_one_q_value=only_one_q_value,
            gt_q_values=gt_q_values, disable_resize=disable_resize, oversample=oversample,
            validation_fraction=validation_fraction, validation_freq=validation_freq, summaries=summaries,
            load_model_path=load_model_path, batch_size=batch_size, zero_sd_after_training=zero_sd_after_training,
            include_goal_states=include_goal_states, discount=discount, shift_q_values=shift_q_values
        )

        self.active_task_indices = active_task_indices
        self.bn_momentum = bn_momentum

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

    def prepare_dataset_(self):

        with open(self.load_path, "rb") as file:
            transitions = pickle.load(file)

        self.dataset = prepare_bisim_dataset_with_q_values(
            transitions, self.grid_size, self.num_pucks, gt_q_values=self.gt_q_values,
            only_one_q_value=self.only_one_q_value, include_goal_state=self.include_goal_states,
            shift_qs_by=self.discount if self.shift_q_values else None
        )

        self.dataset[constants.NEW_DONES] = self.dataset[constants.REWARDS][:, self.active_task_indices] > 0.0

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

        self.train_not_dones = np.logical_not(self.dataset[constants.NEW_DONES])
        self.valid_not_dones = np.logical_not(self.valid_dataset[constants.NEW_DONES])

        self.log_dataset_stats_()

    def prepare_model_(self):

        self.model = QPredictionModelMultiTask(
            [self.depth_size, self.depth_size], self.num_blocks, self.num_actions, [32, 64, 128, 256], [4, 4, 4, 4],
            [2, 2, 2, 2], self.hiddens, self.encoder_learning_rate, self.weight_decay, self.encoder_optimizer,
            self.num_steps, self.num_tasks, batch_norm=not self.disable_batch_norm, softplus=not self.disable_softplus,
            no_sample=self.no_sample, only_one_q_value=self.only_one_q_value, bn_momentum=self.bn_momentum
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
