import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import constants


def evaluate_next_state_1nn(dataset, next_state_predictions):

    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(dataset[constants.EMBEDDINGS], dataset[constants.STATE_LABELS])
    nearest_labels = nn.predict(next_state_predictions)

    cls_accuracies = []
    for cls in np.unique(dataset[constants.NEXT_STATE_LABELS]):

        if cls == -1:
            continue

        mask = np.logical_and(
            dataset[constants.NEXT_STATE_LABELS] == cls, np.logical_not(dataset[constants.DONES])
        )
        tmp_accuracy = np.mean(nearest_labels[mask] == dataset[constants.NEXT_STATE_LABELS][mask])
        cls_accuracies.append(tmp_accuracy)

    balanced_t_accuracy = np.mean(cls_accuracies)

    return cls_accuracies, balanced_t_accuracy


def get_perplexities(log_distribution):
    log2 = log_distribution / np.log(2)
    return 2 ** (- np.sum(np.exp(log_distribution) * log2, axis=1))
