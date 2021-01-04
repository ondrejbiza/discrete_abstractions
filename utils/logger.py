import os
import logging
import sys
import numpy as np
import constants


class Logger:

    def __init__(self, save_file=None, print_logs=True):

        self.save_file = save_file
        self.print_logs = print_logs

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers = []

        if self.save_file is not None:

            file_dir = os.path.dirname(self.save_file)

            if len(file_dir) > 0 and not os.path.isdir(file_dir):
                os.makedirs(file_dir)

            file_handler = logging.FileHandler(self.save_file)
            file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(file_handler)

        if self.print_logs:

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def close(self):

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    @staticmethod
    def silence_tensorflow():

        import tensorflow as tf

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        version = tf.__version__
        major, minor = version.split(".")[:2]
        major, minor = int(major), int(minor)

        if major == 1 and minor >= 14:
            tf.get_logger().setLevel(logging.ERROR)


class LearnExpectationContinuousLogger(Logger):

    def __init__(self, save_file=None, print_logs=True):

        super(LearnExpectationContinuousLogger, self).__init__(save_file=save_file, print_logs=print_logs)

    def log_cluster_accuracy_and_purity(self, cluster_accuracy, mean_purity, cluster_purities, cluster_sizes):

        self.info("cluster accuracy: {:.2f}%".format(cluster_accuracy * 100))

        for idx in range(len(cluster_purities)):
            self.info("cluster {:d}: {:.2f}% purity, {:d} points".format(
                idx, cluster_purities[idx] * 100, cluster_sizes[idx])
            )

        self.info("mean purity: {:.2f}%".format(mean_purity * 100))

    def log_dataset_statistics(self, dataset):

        self.info("actions: min {:d} max {:d}".format(
            np.min(dataset[constants.ACTIONS]), np.max(dataset[constants.ACTIONS]))
        )
        self.info("rewards: min {:.0f} max {:.0f}".format(
            np.min(dataset[constants.REWARDS]), np.max(dataset[constants.REWARDS]))
        )
        self.info("state labels: min {:d} max {:d}".format(
            np.min(dataset[constants.STATE_LABELS]), np.max(dataset[constants.STATE_LABELS]))
        )

        if constants.ACTION_LABELS in dataset:
            self.info("action labels: min {:d} max {:d}".format(
                np.min(dataset[constants.ACTION_LABELS]), np.max(dataset[constants.ACTION_LABELS]))
            )
