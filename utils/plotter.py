import matplotlib.pyplot as plt
import seaborn as sns
import constants


class Plotter:

    def __init__(self, saver):

        self.saver = saver


class OneStepModelPlotter(Plotter):

    def __init__(self, saver, means, sds):

        super(OneStepModelPlotter, self).__init__(saver)
        self.means = means
        self.sds = sds

    def show_data_with_losses(self, valid_dataset, losses, loss_names, to_show=50):

        for i in range(to_show):

            print("sample {:d}".format(i + 1))
            print("action {:d}, reward {:.2f}".format(
                valid_dataset[constants.ACTIONS][i], valid_dataset[constants.REWARDS][i])
            )

            if constants.Q_VALUES in valid_dataset:
                print("q values:", valid_dataset[constants.Q_VALUES][i])

            for j in range(len(loss_names)):
                print("{:s}: {:.4f}".format(loss_names[j], losses[i, j]))

            plt.subplot(1, 2, 1)
            plt.imshow(valid_dataset[constants.STATES][i] * self.sds + self.means)

            plt.subplot(1, 2, 2)
            plt.imshow(valid_dataset[constants.NEXT_STATES][i] * self.sds + self.means)
            plt.show()

    def plot_loss_distribution(self, losses, loss_name):

        sns.distplot(losses, hist_kws={"edgecolor": "black"})
        plt.title(loss_name)
        plt.show()
