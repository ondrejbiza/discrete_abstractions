import copy as cp
import collections
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import balancing.utils as balancing_utils
import constants


class ArrayDataset:

    def __init__(self, arrays_dict):

        self.arrays = arrays_dict
        self.size = None

        self.compute_size()

    def compute_size(self):
        self.size = len(list(self.arrays.values())[0])

    def oversample(self, classes):

        indices = np.array(list(range(self.size)), dtype=np.int32)
        y = classes.astype(np.int32)
        x, _ = balancing_utils.oversample(indices, y)

        for key in self.arrays.keys():

            self.arrays[key] = self.arrays[key][x]

    def normalize_together(self, column_keys, std_threshold=0.001):

        columns = [self.arrays[key] for key in column_keys]
        columns = np.concatenate(columns, axis=0)

        means = np.mean(columns, axis=0)
        stds = np.std(columns, axis=0)
        stds[stds < std_threshold] = 1

        for key in column_keys:

            self.arrays[key] = (self.arrays[key] - means) / stds

        return means, stds

    def shuffle(self):

        keys = []
        arrays_list = []

        for key, array in self.arrays.items():

            keys.append(key)
            arrays_list.append(array)

        arrays_list = shuffle(*arrays_list)

        for key, array in zip(keys, arrays_list):

            self.arrays[key] = array

    def split(self, other_size):

        other_dataset = {}

        for key, array in self.arrays.items():

            other_dataset[key] = array[:other_size]
            self.arrays[key] = array[other_size:]

        return ArrayDataset(other_dataset)

    def resize_tf(self, original_size, target_size, keys):

        images_pl = tf.placeholder(tf.float32, shape=(None, original_size, original_size), name="images")
        resized_t = tf.image.resize_bilinear(
            images_pl[:, :, :, tf.newaxis], size=(target_size, target_size)
        )[:, :, :, 0]

        with tf.Session() as session:

            for key in keys:

                self.arrays[key] = session.run(resized_t, feed_dict={
                    images_pl: self.arrays[key]
                })

        return images_pl, resized_t

    def concatenate(self, keys, arrays):

        assert len(keys) == len(arrays)

        for key, array in zip(keys, arrays):
            self.arrays[key] = np.concatenate([self.arrays[key], array], axis=0)

        self.compute_size()

    def __getitem__(self, key):

        return self.arrays[key]

    def __setitem__(self, key, value):

        self.arrays[key] = value

    def __contains__(self, key):

        return key in self.arrays


class ListDataset:

    def __init__(self):

        self.columns = collections.defaultdict(list)

    def add(self, key, value):

        self.columns[key].append(value)

    def test_equal_sizes(self):

        ref_size = len(list(self.columns.values())[0])

        for val in self.columns.values():
            assert len(val) == ref_size

    def to_array_dataset(self, dtypes):

        assert dtypes.keys() == self.columns.keys()
        self.test_equal_sizes()

        arrays_dict = {}

        for key in dtypes.keys():

            arrays_dict[key] = np.array(self.columns[key], dtypes[key])

        return ArrayDataset(arrays_dict)


def prepare_bisim_dataset_with_q_values(transitions, grid_size, num_pucks, gt_q_values=False, only_one_q_value=False,
                                        add_action_labels=False, add_state_action_labels=False,
                                        gt_q_values_all_actions=False, include_goal_state=False, reward_scale=10.0,
                                        shift_qs_by=None):

    dataset = ListDataset()

    for idx in range(len(transitions)):

        transition = transitions[idx]

        dataset.add(constants.STATES, transition.state[0][:, :, 0])

        dataset.add(constants.HAND_STATES, transition.state[1])
        dataset.add(constants.ACTIONS, transition.action % (grid_size ** 2))

        dataset.add(constants.REWARDS, transition.reward / reward_scale)
        dataset.add(constants.NEXT_STATES, transition.next_state[0][:, :, 0])
        dataset.add(constants.NEXT_HAND_STATES, transition.next_state[1])
        dataset.add(constants.DONES, transition.is_end)

        if transition.state_block is None:
            dataset.add(constants.STATE_LABELS, 0)
        else:
            dataset.add(constants.STATE_LABELS, transition.state_block)

        if gt_q_values:
            #assert only_one_q_value

            if gt_q_values_all_actions:
                q_values = manual_q_values_homo_all_actions(
                    transition.state_block, transition.action_block, num_pucks, 0.9
                )
            else:
                q_values = manual_q_values(
                    transition.next_state[0], transition.next_state[1], transition.reward / reward_scale, num_pucks, 0.9
                )
                q_values = np.array([q_values] * len(transition.q_values))
        else:
            q_values = transition.q_values / reward_scale

        if shift_qs_by is not None:
            q_values = q_values * shift_qs_by

        dataset.add(constants.Q_VALUES, q_values)

        if transition.next_q_values is None:
            dataset.add(constants.NEXT_Q_VALUES, 1.0)
        else:
            dataset.add(constants.NEXT_Q_VALUES, transition.next_q_values / reward_scale)

        # TODO: hack for multiple tasks
        if np.any(transition.is_end):
            dataset.add(constants.NEXT_STATE_LABELS, -1)
        else:
            if transition.next_state_block is None:
                dataset.add(constants.NEXT_STATE_LABELS, 0)
            else:
                dataset.add(constants.NEXT_STATE_LABELS, transition.next_state_block)

        if add_action_labels:
            dataset.add(constants.ACTION_LABELS, transition.action_block)

            if add_state_action_labels:
                if transition.reward > 0:
                    if num_pucks == 2:
                        tmp_sa_block = 2
                    elif num_pucks == 3:
                        tmp_sa_block = 4
                    else:
                        raise NotImplementedError("More than 3 pucks not supported.")
                else:
                    tmp_sa_block = transition.next_state_block
                dataset.add(constants.STATE_ACTION_LABELS, tmp_sa_block)

    types_dict = {
        constants.STATES: np.float32, constants.HAND_STATES: np.int32, constants.ACTIONS: np.int32,
        constants.REWARDS: np.float32, constants.NEXT_STATES: np.float32, constants.NEXT_HAND_STATES: np.int32,
        constants.DONES: np.bool, constants.STATE_LABELS: np.int32, constants.NEXT_STATE_LABELS: np.int32,
        constants.Q_VALUES: np.float32, constants.NEXT_Q_VALUES: np.float32
    }

    if add_action_labels:
        types_dict[constants.ACTION_LABELS] = np.int32
    if add_state_action_labels:
        types_dict[constants.STATE_ACTION_LABELS] = np.int32

    dataset = dataset.to_array_dataset(types_dict)

    if include_goal_state:

        max_state_label = np.max(dataset[constants.STATE_LABELS])

        mask = dataset[constants.REWARDS] > 0.0
        dataset[constants.DONES][mask] = False
        dataset[constants.NEXT_STATE_LABELS][mask] = max_state_label + 1

        goal_states = cp.deepcopy(dataset[constants.NEXT_STATES][mask])
        goal_hand_states = np.zeros_like(dataset[constants.HAND_STATES][mask])
        goal_actions = np.zeros_like(dataset[constants.ACTIONS][mask])
        goal_rewards = np.zeros_like(dataset[constants.REWARDS][mask])
        goal_dones = np.ones_like(dataset[constants.DONES][mask])
        goal_state_labels = np.zeros_like(dataset[constants.STATE_LABELS][mask]) + (max_state_label + 1)
        goal_next_state_labels = np.zeros_like(dataset[constants.NEXT_STATE_LABELS][mask])
        goal_q_values = np.ones_like(dataset[constants.Q_VALUES][mask])
        goal_next_q_values = np.ones_like(dataset[constants.Q_VALUES][mask])

        if add_action_labels:
            goal_action_labels = np.zeros_like(dataset[constants.ACTION_LABELS][mask])

        keys = [
            constants.STATES, constants.HAND_STATES, constants.ACTIONS, constants.REWARDS, constants.NEXT_STATES,
            constants.NEXT_HAND_STATES, constants.DONES, constants.STATE_LABELS, constants.NEXT_STATE_LABELS,
            constants.Q_VALUES, constants.NEXT_Q_VALUES
        ]

        arrays = [
            goal_states, goal_hand_states, goal_actions, goal_rewards, goal_states, goal_hand_states, goal_dones,
            goal_state_labels, goal_next_state_labels, goal_q_values, goal_next_q_values
        ]

        if add_action_labels:
            keys.append(constants.ACTION_LABELS)
            arrays.append(goal_action_labels)

        dataset.concatenate(keys, arrays)

    return dataset


def prepare_minatar_dataset(transitions, q_scaling_factor=10.0):

    dataset = ListDataset()

    for idx in range(len(transitions)):

        transition = transitions[idx]

        dataset.add(constants.STATES, transition.state)
        dataset.add(constants.ACTIONS, transition.action)
        dataset.add(constants.REWARDS, transition.reward)
        dataset.add(constants.NEXT_STATES, transition.next_state)
        dataset.add(constants.DONES, transition.is_end)

        dataset.add(constants.Q_VALUES, transition.q_values / q_scaling_factor)

        if transition.next_q_values is None:
            dataset.add(constants.NEXT_Q_VALUES, 1.0)
        else:
            dataset.add(constants.NEXT_Q_VALUES, transition.next_q_values / q_scaling_factor)

        # compatibility
        dataset.add(constants.HAND_STATES, 0)
        dataset.add(constants.NEXT_HAND_STATES, 0)
        dataset.add(constants.STATE_LABELS, 0)
        dataset.add(constants.NEXT_STATE_LABELS, 0)

    types_dict = {
        constants.STATES: np.float32, constants.HAND_STATES: np.int32, constants.ACTIONS: np.int32,
        constants.REWARDS: np.float32, constants.NEXT_STATES: np.float32, constants.NEXT_HAND_STATES: np.int32,
        constants.DONES: np.bool, constants.STATE_LABELS: np.int32, constants.NEXT_STATE_LABELS: np.int32,
        constants.Q_VALUES: np.float32, constants.NEXT_Q_VALUES: np.float32
    }

    dataset = dataset.to_array_dataset(types_dict)

    return dataset


def prepare_bisim_dataset_with_values(transitions, grid_size, num_pucks, gt_values=False, add_action_labels=False,
                                      add_state_action_labels=False, include_goal_state=False,
                                      correct_goal_next_values=True):

    dataset = ListDataset()

    for idx in range(len(transitions)):

        transition = transitions[idx]

        dataset.add(constants.STATES, transition.state[0][:, :, 0])

        dataset.add(constants.HAND_STATES, transition.state[1])
        dataset.add(constants.ACTIONS, transition.action % (grid_size ** 2))

        dataset.add(constants.REWARDS, transition.reward / 10)
        dataset.add(constants.NEXT_STATES, transition.next_state[0][:, :, 0])
        dataset.add(constants.NEXT_HAND_STATES, transition.next_state[1])
        dataset.add(constants.DONES, transition.is_end)

        if transition.state_block is None:
            dataset.add(constants.STATE_LABELS, 0)
        else:
            dataset.add(constants.STATE_LABELS, transition.state_block)

        if gt_values:
            raise NotImplementedError("TODO")
        else:
            values = transition.value / 10
            next_values = transition.next_value / 10

        dataset.add(constants.VALUES, values)

        if transition.reward > 0 and correct_goal_next_values:
            dataset.add(constants.NEXT_VALUES, 1)
        else:
            dataset.add(constants.NEXT_VALUES, next_values)

        if transition.is_end:
            dataset.add(constants.NEXT_STATE_LABELS, -1)
        else:
            if transition.next_state_block is None:
                dataset.add(constants.NEXT_STATE_LABELS, 0)
            else:
                dataset.add(constants.NEXT_STATE_LABELS, transition.next_state_block)

        if add_action_labels:
            dataset.add(constants.ACTION_LABELS, transition.action_block)

            if add_state_action_labels:
                if transition.reward > 0:
                    if num_pucks == 2:
                        tmp_sa_block = 2
                    elif num_pucks == 3:
                        tmp_sa_block = 4
                    else:
                        raise NotImplementedError("More than 3 pucks not supported.")
                else:
                    tmp_sa_block = transition.next_state_block
                dataset.add(constants.STATE_ACTION_LABELS, tmp_sa_block)

    types_dict = {
        constants.STATES: np.float32, constants.HAND_STATES: np.int32, constants.ACTIONS: np.int32,
        constants.REWARDS: np.float32, constants.NEXT_STATES: np.float32, constants.NEXT_HAND_STATES: np.int32,
        constants.DONES: np.bool, constants.STATE_LABELS: np.int32, constants.NEXT_STATE_LABELS: np.int32,
        constants.VALUES: np.float32, constants.NEXT_VALUES: np.float32
    }

    if add_action_labels:
        types_dict[constants.ACTION_LABELS] = np.int32
    if add_state_action_labels:
        types_dict[constants.STATE_ACTION_LABELS] = np.int32

    dataset = dataset.to_array_dataset(types_dict)

    if include_goal_state:

        max_state_label = np.max(dataset[constants.STATE_LABELS])

        mask = dataset[constants.REWARDS] > 0.0
        dataset[constants.DONES][mask] = False
        dataset[constants.NEXT_STATE_LABELS][mask] = max_state_label + 1

        goal_states = cp.deepcopy(dataset[constants.NEXT_STATES][mask])
        goal_hand_states = np.zeros_like(dataset[constants.HAND_STATES][mask])
        goal_actions = np.zeros_like(dataset[constants.ACTIONS][mask])
        goal_rewards = np.zeros_like(dataset[constants.REWARDS][mask])
        goal_dones = np.ones_like(dataset[constants.DONES][mask])
        goal_state_labels = np.zeros_like(dataset[constants.STATE_LABELS][mask]) + (max_state_label + 1)
        goal_next_state_labels = np.zeros_like(dataset[constants.NEXT_STATE_LABELS][mask])
        goal_values = np.ones_like(dataset[constants.VALUES][mask])
        goal_next_values = np.ones_like(dataset[constants.VALUES][mask])

        if add_action_labels:
            goal_action_labels = np.zeros_like(dataset[constants.ACTION_LABELS][mask])

        keys = [
            constants.STATES, constants.HAND_STATES, constants.ACTIONS, constants.REWARDS, constants.NEXT_STATES,
            constants.NEXT_HAND_STATES, constants.DONES, constants.STATE_LABELS, constants.NEXT_STATE_LABELS,
            constants.VALUES, constants.NEXT_VALUES
        ]

        arrays = [
            goal_states, goal_hand_states, goal_actions, goal_rewards, goal_states, goal_hand_states, goal_dones,
            goal_state_labels, goal_next_state_labels, goal_values, goal_next_values
        ]

        if add_action_labels:
            keys.append(constants.ACTION_LABELS)
            arrays.append(goal_action_labels)

        dataset.concatenate(keys, arrays)

    return dataset


def manual_q_values(next_state, next_hand_state, reward, goal, discount):

    if reward == 1.0:
        return 1.0

    height = round(np.max(next_state))
    steps_to_goal = goal - height

    assert steps_to_goal > 0

    if next_hand_state == 0:
        return discount ** (steps_to_goal * 2)
    else:
        return discount ** (steps_to_goal * 2 - 1)


def manual_q_values_homo_all_actions(homo_state, homo_actions, goal, discount):

    homo_actions = np.array(homo_actions)
    q_values = np.zeros_like(homo_actions, dtype=np.float32)

    if goal == 2:
        if homo_state == 0:
            q_values[homo_actions == 0] = discount ** 2
            q_values[homo_actions == 1] = discount
        else:
            q_values[homo_actions == 0] = discount ** 2
            q_values[homo_actions == 1] = 1.0
    elif goal == 3:
        if homo_state == 0:
            q_values[homo_actions == 0] = discount ** 4
            q_values[homo_actions == 1] = discount ** 3
        elif homo_state == 1:
            # hand full
            q_values[homo_actions == 0] = discount ** 4
            q_values[homo_actions == 1] = discount ** 2
        elif homo_state == 2:
            q_values[homo_actions == 0] = discount ** 3
            q_values[homo_actions == 1] = discount ** 2
            q_values[homo_actions == 2] = discount
        else:
            # hand full
            q_values[homo_actions == 0] = discount ** 2
            q_values[homo_actions == 1] = 1.0
    else:
        raise NotImplementedError("Only 2 and 3 pucks supported.")

    return q_values


def prepare_bisim_dataset_with_q_values_from_values(transitions, grid_size, num_pucks, add_action_labels=False,
                                                    add_state_action_labels=False, include_goal_state=False,
                                                    discount=0.9):

    dataset = ListDataset()

    for idx in range(len(transitions)):

        transition = transitions[idx]

        dataset.add(constants.STATES, transition.state[0][:, :, 0])

        dataset.add(constants.HAND_STATES, transition.state[1])
        dataset.add(constants.ACTIONS, transition.action % (grid_size ** 2))

        dataset.add(constants.REWARDS, transition.reward / 10)
        dataset.add(constants.NEXT_STATES, transition.next_state[0][:, :, 0])
        dataset.add(constants.NEXT_HAND_STATES, transition.next_state[1])
        dataset.add(constants.DONES, transition.is_end)

        if transition.state_block is None:
            dataset.add(constants.STATE_LABELS, 0)
        else:
            dataset.add(constants.STATE_LABELS, transition.state_block)

        if transition.is_end:
            q_value = transition.reward / 10
        else:
            q_value = (transition.reward + discount * transition.next_value) / 10.0

        dataset.add(constants.Q_VALUES, [q_value] * (grid_size ** 2))
        dataset.add(constants.NEXT_Q_VALUES, 0)

        if transition.is_end:
            dataset.add(constants.NEXT_STATE_LABELS, -1)
        else:
            if transition.next_state_block is None:
                dataset.add(constants.NEXT_STATE_LABELS, 0)
            else:
                dataset.add(constants.NEXT_STATE_LABELS, transition.next_state_block)

        if add_action_labels:
            dataset.add(constants.ACTION_LABELS, transition.action_block)

            if add_state_action_labels:
                if transition.reward > 0:
                    if num_pucks == 2:
                        tmp_sa_block = 2
                    elif num_pucks == 3:
                        tmp_sa_block = 4
                    else:
                        raise NotImplementedError("More than 3 pucks not supported.")
                else:
                    tmp_sa_block = transition.next_state_block
                dataset.add(constants.STATE_ACTION_LABELS, tmp_sa_block)

    types_dict = {
        constants.STATES: np.float32, constants.HAND_STATES: np.int32, constants.ACTIONS: np.int32,
        constants.REWARDS: np.float32, constants.NEXT_STATES: np.float32, constants.NEXT_HAND_STATES: np.int32,
        constants.DONES: np.bool, constants.STATE_LABELS: np.int32, constants.NEXT_STATE_LABELS: np.int32,
        constants.Q_VALUES: np.float32, constants.NEXT_Q_VALUES: np.float32
    }

    if add_action_labels:
        types_dict[constants.ACTION_LABELS] = np.int32
    if add_state_action_labels:
        types_dict[constants.STATE_ACTION_LABELS] = np.int32

    dataset = dataset.to_array_dataset(types_dict)

    if include_goal_state:

        max_state_label = np.max(dataset[constants.STATE_LABELS])

        mask = dataset[constants.REWARDS] > 0.0
        dataset[constants.DONES][mask] = False
        dataset[constants.NEXT_STATE_LABELS][mask] = max_state_label + 1

        goal_states = cp.deepcopy(dataset[constants.NEXT_STATES][mask])
        goal_hand_states = np.zeros_like(dataset[constants.HAND_STATES][mask])
        goal_actions = np.zeros_like(dataset[constants.ACTIONS][mask])
        goal_rewards = np.zeros_like(dataset[constants.REWARDS][mask])
        goal_dones = np.ones_like(dataset[constants.DONES][mask])
        goal_state_labels = np.zeros_like(dataset[constants.STATE_LABELS][mask]) + (max_state_label + 1)
        goal_next_state_labels = np.zeros_like(dataset[constants.NEXT_STATE_LABELS][mask])
        goal_q_values = np.ones_like(dataset[constants.Q_VALUES][mask])
        goal_next_q_values = np.ones_like(dataset[constants.Q_VALUES][mask])

        if add_action_labels:
            goal_action_labels = np.zeros_like(dataset[constants.ACTION_LABELS][mask])

        keys = [
            constants.STATES, constants.HAND_STATES, constants.ACTIONS, constants.REWARDS, constants.NEXT_STATES,
            constants.NEXT_HAND_STATES, constants.DONES, constants.STATE_LABELS, constants.NEXT_STATE_LABELS,
            constants.Q_VALUES, constants.NEXT_Q_VALUES
        ]

        arrays = [
            goal_states, goal_hand_states, goal_actions, goal_rewards, goal_states, goal_hand_states, goal_dones,
            goal_state_labels, goal_next_state_labels, goal_q_values, goal_next_q_values
        ]

        if add_action_labels:
            keys.append(constants.ACTION_LABELS)
            arrays.append(goal_action_labels)

        dataset.concatenate(keys, arrays)

    return dataset
