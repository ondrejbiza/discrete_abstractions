from abc import ABC
import os
import tensorflow as tf


class Model(ABC):

    def __init__(self):

        self.session = NotImplemented
        self.saver = NotImplemented

    def start_session(self, gpu_memory=None):

        gpu_options = None
        if gpu_memory is not None:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)

        tf_config = tf.ConfigProto(gpu_options=gpu_options)

        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()

    def load(self, path):

        self.saver.restore(self.session, path)

    def save(self, path):

        path_dir = os.path.dirname(path)

        if len(path_dir) > 0 and not os.path.isdir(path_dir):
            os.makedirs(path_dir)

        self.saver.save(self.session, path)

    @staticmethod
    def summarize(var, name):

        tf.summary.scalar(name + "_mean", tf.reduce_mean(var))
        tf.summary.scalar(name + "_min", tf.reduce_min(var))
        tf.summary.scalar(name + "_max", tf.reduce_max(var))

        tf.summary.histogram(name + "_hist", var)
