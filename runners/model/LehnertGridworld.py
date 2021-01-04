import numpy as np
import pickle
from utils.dataset import ListDataset
import constants
import vis_utils
from model.LehnertGridworldModel import LehnertGridworldModel
import config_constants as cc
from runners.runner import Runner


class LehnertGridworldRunner(Runner):

    def __init__(self, runner_config, model_config):

        super(LehnertGridworldRunner, self).__init__()

        self.model_config = model_config

        self.load_path = runner_config[cc.LOAD_PATH]
        self.saver = runner_config[cc.SAVER]
        self.logger = runner_config[cc.LOGGER]
        self.oversample = runner_config[cc.OVERSAMPLE]
        self.validation_fraction = runner_config[cc.VALIDATION_FRACTION]
        self.validation_freq = runner_config[cc.VALIDATION_FREQ]
        self.batch_size = runner_config[cc.BATCH_SIZE]
        self.load_model_path = runner_config[cc.LOAD_MODEL_PATH]
        self.num_steps = runner_config[cc.NUM_STEPS]
        self.data_limit = runner_config[cc.DATA_LIMIT]

    def setup(self):

        self.prepare_dataset_()
        self.prepare_model_()

    def evaluate_and_visualize(self):
        pass

    def main_training_loop(self):

        self.prepare_losses_()

        if self.load_model_path is None:

            for train_step in range(self.num_steps):

                if train_step % self.validation_freq == 0:
                    tmp_valid_losses = self.model.validate(
                        self.valid_dataset[constants.STATES], self.valid_dataset[constants.ACTIONS],
                        self.valid_dataset[constants.REWARDS], self.valid_dataset[constants.NEXT_STATES]
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

                to_add = [bundle[constants.TOTAL_LOSS], bundle[constants.REWARD_LOSS],
                          bundle[constants.TRANSITION_LOSS]]

                self.add_step_losses_(to_add, train_step)

            if len(self.tmp_epoch_losses) > 0:
                self.add_epoch_losses_()

            self.postprocess_losses_()
            self.plot_losses_()

    def plot_losses_(self):

        train_indices = [1, 2]
        valid_indices = [0, 1]
        train_labels = ["train reward loss", "train transition loss"]
        valid_labels = ["valid reward loss", "valid transition loss"]

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

    def get_feed_dict_(self, epoch_step):

        b = np.index_exp[epoch_step * self.batch_size:(epoch_step + 1) * self.batch_size]

        feed_dict = {
            self.model.states_pl: self.dataset[constants.STATES][b],
            self.model.actions_pl: self.dataset[constants.ACTIONS][b],
            self.model.rewards_pl: self.dataset[constants.REWARDS][b],
            self.model.next_states_pl: self.dataset[constants.NEXT_STATES][b],
            self.model.is_training: True
        }

        return feed_dict

    def prepare_dataset_(self):

        with open(self.load_path, "rb") as file:
            transitions = pickle.load(file)

        if self.data_limit is not None:
            transitions = transitions[:self.data_limit]

        self.dataset = self.transitions_to_dataset_(transitions)

        vis_utils.plot_many_histograms(
            [self.dataset[constants.REWARDS], self.dataset[constants.STATE_LABELS],
             self.dataset[constants.NEXT_STATE_LABELS]],
            ["rewards_hist", "state_labels_hist", "next_state_labels_hist"], xlabel="items",
            num_bins=10, saver=self.saver
        )

        if self.oversample:
            self.dataset.oversample((self.dataset[constants.REWARDS] > 0))

        self.dataset.shuffle()

        self.num_samples = self.dataset.size
        self.valid_samples = int(self.num_samples * self.validation_fraction)
        self.valid_dataset = self.dataset.split(self.valid_samples)

        self.epoch_size = self.dataset[constants.STATES].shape[0] // self.batch_size

        self.log_dataset_stats_()

    def log_dataset_stats_(self):

        for dataset, name in zip([self.dataset, self.valid_dataset], ["train", "valid"]):

            self.logger.info("actions: min {:d} max {:d}".format(
                np.min(dataset[constants.ACTIONS]), np.max(dataset[constants.ACTIONS]))
            )
            self.logger.info("rewards: min {:.0f} max {:.0f}".format(
                np.min(dataset[constants.REWARDS]), np.max(dataset[constants.REWARDS]))
            )
            self.logger.info("state labels: min {:d} max {:d}".format(
                np.min(dataset[constants.STATE_LABELS]), np.max(dataset[constants.STATE_LABELS]))
            )

    @staticmethod
    def transitions_to_dataset_(transitions):

        dataset = ListDataset()

        for t in transitions:
            dataset.add(constants.STATES, t.state)
            dataset.add(constants.ACTIONS, t.action)
            dataset.add(constants.REWARDS, t.reward)
            dataset.add(constants.NEXT_STATES, t.next_state)

            dataset.add(constants.STATE_LABELS, t.state_block)
            dataset.add(constants.NEXT_STATE_LABELS, t.next_state_block)

        types_dict = {
            constants.STATES: np.float32, constants.ACTIONS: np.int32, constants.REWARDS: np.float32,
            constants.NEXT_STATES: np.float32, constants.STATE_LABELS: np.int32, constants.NEXT_STATE_LABELS: np.int32
        }

        return dataset.to_array_dataset(types_dict)

    def prepare_model_(self):

        self.model = LehnertGridworldModel(self.model_config)
        self.model.build()
        self.model.start_session()

        if self.load_model_path is not None:
            self.model.load(self.load_model_path)

        self.to_run = {
            constants.TRAIN_STEP: self.model.train_step,
            constants.TOTAL_LOSS: self.model.loss_t,
            constants.REWARD_LOSS: self.model.reward_loss_t,
            constants.TRANSITION_LOSS: self.model.transition_loss_t
        }
