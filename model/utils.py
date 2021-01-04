import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D         # don't delete this, necessary for 3d projection
import seaborn as sns
import vis_utils


def save_dists(embeddings, save_dir, step, ext="pdf", assignments=None, num_classes=None):

    orig_dir = os.path.join(save_dir, "original")
    ord_dir = os.path.join(save_dir, "ordered")

    if not os.path.isdir(orig_dir):
        os.makedirs(orig_dir)
    if not os.path.isdir(ord_dir):
        os.makedirs(ord_dir)

    if assignments is None:
        assignments = np.argmax(embeddings, axis=1).astype(np.int32)

    if num_classes is None:
        num_classes = embeddings.shape[1]

    counts = np.zeros(num_classes, dtype=np.int32)
    tmp_vals, tmp_counts = np.unique(assignments, return_counts=True)
    counts[tmp_vals] = tmp_counts

    df = pd.DataFrame({
        "counts": counts
    })

    plt.clf()
    sns.barplot(x=df["counts"].index, y=df["counts"], color="cornflowerblue")
    plt.xticks([])
    plt.xlabel("blocks (original order)")
    plt.savefig(os.path.join(orig_dir, "{}.{}".format(step, ext)))

    plt.clf()
    sns.barplot(x=df["counts"].index, y=df["counts"].sort_values(ascending=False), color="cornflowerblue")
    plt.xticks([])
    plt.xlabel("blocks (ordered)")
    plt.savefig(os.path.join(ord_dir, "{}.{}".format(step, ext)))


def save_dists_report(embeddings, states, i, save_dir):

    assignments = np.argmax(embeddings, axis=1).astype(np.int32)
    blocks = np.unique(assignments)

    save_subdir = os.path.join(save_dir, "dists_report", str(i))
    if not os.path.isdir(save_subdir):
        os.makedirs(save_subdir)

    for block in sorted(blocks):

        all_indices = np.where(assignments == block)[0]

        if len(all_indices) < 16:
            replace = True
        else:
            replace = False

        indices = np.random.choice(all_indices, size=16, replace=replace)

        save_file = os.path.join(save_subdir, "block_{}.png".format(block))

        plt.clf()
        vis_utils.image_grid(states[indices], 4, 4, 0, 2)
        plt.savefig(save_file)


def transform_and_plot_embeddings(embeddings, state_labels=None, num_components=3, use_colorbar=True):

    if embeddings.shape[1] > 3:
        # project embeddings with dimensionality higher than 3
        pca = PCA(n_components=num_components)
        embeddings = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_

    fig = plt.figure()

    if embeddings.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    if state_labels is not None:
        if embeddings.shape[1] == 3:
            sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=state_labels)
        else:
            sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=state_labels)

        if use_colorbar:
            fig.colorbar(sc)
    else:
        if embeddings.shape[1] == 3:
            sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2])
        else:
            sc = ax.scatter(embeddings[:, 0], embeddings[:, 1])

    plt.xlabel("PCA1: {:.1f}% variance explained".format(explained[0] * 100))
    plt.ylabel("PCA2: {:.1f}% variance explained".format(explained[1] * 100))

    if embeddings.shape[1] == 3:
        plt.xlabel("PCA3: {:.1f}% variance explained".format(explained[2] * 100))

    return fig


def states_cutoff(assignments, r_hat, t_hat, num_blocks, cutoff_fraction):

    r_hat = np.copy(r_hat)
    t_hat = np.copy(t_hat)

    all_counts = np.zeros((num_blocks,), dtype=np.int32)
    blocks, counts = np.unique(assignments, return_counts=True)
    all_counts[blocks] = counts
    cutoff_num = len(assignments) * cutoff_fraction
    cutoff = all_counts < cutoff_num

    if np.sum(cutoff) > 0:

        # all states under the cutoff receive zero reward
        r_hat[:, cutoff] = 0.0

        # all states under the cutoff are absorbing
        for idx in range(num_blocks):
            if cutoff[idx]:
                t_hat[:, idx, :] = 0.0
                t_hat[:, idx, idx] = 1.0

    return r_hat, t_hat


def transform_embeddings(embeddings, t_hat, actions):

    per_action_t_hat = t_hat[actions, :, :]
    return np.matmul(embeddings[:, np.newaxis, :], per_action_t_hat)[:, 0, :]


def binary_reward_accuracies(assignments, r_hat, actions, r):

    per_action_r_hat = r_hat[actions, :]
    predicted_rewards = per_action_r_hat[list(range(len(assignments))), assignments]
    predicted_rewards[predicted_rewards < 0.5] = 0.0
    predicted_rewards[predicted_rewards >= 0.5] = 1.0

    correct = predicted_rewards == r

    if np.sum(r == 1.0) > 0:
        accuracy_one = np.sum(correct[r == 1.0]) / np.sum(r == 1.0)
    else:
        accuracy_one = 1.0

    if np.sum(r == 0.0) > 0:
        accuracy_zero = np.sum(correct[r == 0.0]) / np.sum(r == 0.0)
    else:
        accuracy_zero = 1.0

    return accuracy_one, accuracy_zero


def binary_reward_accuracies_continuous(embeddings, r_hat, actions, r):

    per_action_r_hat = r_hat[actions, :]
    predicted_rewards = np.sum(per_action_r_hat * embeddings, axis=1)
    predicted_rewards[predicted_rewards < 0.5] = 0.0
    predicted_rewards[predicted_rewards >= 0.5] = 1.0

    correct = predicted_rewards == r

    if np.sum(r == 1.0) > 0:
        accuracy_one = np.sum(correct[r == 1.0]) / np.sum(r == 1.0)
    else:
        accuracy_one = 1.0

    if np.sum(r == 0.0) > 0:
        accuracy_zero = np.sum(correct[r == 0.0]) / np.sum(r == 0.0)
    else:
        accuracy_zero = 1.0

    return accuracy_one, accuracy_zero


class LinearDecay:

    def __init__(self, start_value, end_value, start_decay, end_decay):

        self.start_value = start_value
        self.end_value = end_value
        self.start_decay = start_decay
        self.end_decay = end_decay

    def value(self, current_step):

        if current_step < self.start_decay:
            return self.start_value
        else:
            fraction = min((current_step - self.start_decay) / (self.end_decay - self.start_decay), 1)
            return self.start_value + fraction * (self.end_value - self.start_value)
