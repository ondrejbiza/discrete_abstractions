import os
import numpy as np


class Loader:

    NUMPY_ARRAY_EXT = "npy"

    def __init__(self, load_dir):

        self.load_dir = load_dir

    def load_array(self, file_name):

        if self.load_dir is not None:

            load_path = self.get_load_file(file_name, self.NUMPY_ARRAY_EXT)
            return np.load(load_path)

        return None

    def get_load_file(self, file_name, ext):

        return os.path.join(self.load_dir, "{}.{}".format(file_name, ext))
