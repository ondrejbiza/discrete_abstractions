from abc import ABC, abstractmethod
import numpy as np


class Runner(ABC):

    def __init__(self):

        self.model = NotImplemented
        self.saver = NotImplemented
        self.train_embeddings = NotImplemented
        self.valid_embeddings = NotImplemented

        self.all_losses, self.per_epoch_losses, self.tmp_epoch_losses, self.valid_losses = [None] * 4

    @abstractmethod
    def setup(self):
        return

    @abstractmethod
    def main_training_loop(self):
        return

    @abstractmethod
    def evaluate_and_visualize(self):
        return

    def save_model(self):
        self.model.save(self.saver.get_save_file("model", ext=""))

    def close_model_session(self):
        self.model.stop_session()

    def prepare_losses_(self):

        self.all_losses = []
        self.per_epoch_losses = []
        self.tmp_epoch_losses = []
        self.valid_losses = []

    def add_step_losses_(self, losses, step):

        losses = np.array([np.mean(loss) for loss in losses])

        if np.any(np.isnan(losses)):
            raise ValueError("NAN in losses at step {:d}".format(step))

        self.tmp_epoch_losses.append(losses)
        self.all_losses.append(losses)

    def add_epoch_losses_(self):

        tmp_epoch_losses = np.mean(np.stack(self.tmp_epoch_losses, axis=0), axis=0)
        self.per_epoch_losses.append(tmp_epoch_losses)
        self.tmp_epoch_losses = []

    def postprocess_losses_(self):

        self.valid_losses = np.stack(self.valid_losses, axis=0)
        self.all_losses = np.stack(self.all_losses, axis=0)
        self.per_epoch_losses = np.stack(self.per_epoch_losses, axis=0)

    def save_embeddings(self):

        self.saver.save_array(self.train_embeddings, "train_embeddings")
        self.saver.save_array(self.valid_embeddings, "valid_embeddings")
