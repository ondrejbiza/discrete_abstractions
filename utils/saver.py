import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Saver:

    NUMPY_ARRAY_EXT = "npy"
    PICKLE_EXT = "pickle"

    def __init__(self, save_dir):

        self.save_dir = save_dir

    def create_dir_name(self, template, variables, switches, args):

        self.save_dir = template.format(*variables)

        for switch in switches:
            if getattr(args, switch):
                self.save_dir += "_{:s}".format(switch)

    def append_dir_name(self, dir_name):

        self.save_dir = os.path.join(self.save_dir, dir_name)

    def create_dir(self, add_run_subdir=False):

        if self.save_dir is not None and not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        if add_run_subdir:
            i = 1
            while True:
                run_name = "run{:d}".format(i)
                run_dir = os.path.join(self.save_dir, run_name)

                if not os.path.isdir(run_dir):
                    os.makedirs(run_dir)
                    self.save_dir = run_dir
                    return

                i += 1

    def save_array_as_txt(self, data, file_name, ext="dat"):

        if self.save_dir is not None:

            save_path = self.get_save_file(file_name, ext)
            np.savetxt(save_path, data)

    def save_array(self, data, file_name):

        if self.save_dir is not None:

            save_path = self.get_save_file(file_name, self.NUMPY_ARRAY_EXT)
            np.save(save_path, data)

    def save_pickle(self,  data, file_name):

        if self.save_dir is not None:

            save_path = self.get_save_file(file_name, self.PICKLE_EXT)
            with open(save_path, "wb") as file:
                pickle.dump(data, file)

    def save_figure(self, fig, file_name, ext="pdf", show=False, close=True):

        if self.save_dir is not None:

            save_path = self.get_save_file(file_name, ext)
            fig.savefig(save_path)

        if show:
            fig.show()

        if close:
            plt.close(fig)

    def save_by_print(self, data, file_name, ext="txt"):

        if self.save_dir is not None:

            save_path = self.get_save_file(file_name, ext)

            with open(save_path, "w") as file:
                print(data, file=file)

    def get_save_file(self, file_name, ext):

        if self.save_dir is None:
            return None
        else:
            if ext is None:
                return os.path.join(self.save_dir, file_name)
            else:
                return os.path.join(self.save_dir, "{}.{}".format(file_name, ext))

    def get_new_dir(self, dir_name):

        dir_path = os.path.join(self.save_dir, dir_name)

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        return dir_path

    def has_dir(self):

        return self.save_dir is not None
