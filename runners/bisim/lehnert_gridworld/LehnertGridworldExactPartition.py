import copy as cp
import numpy as  np
from envs.lehnert_gridworld import LehnertGridworld
from model.LehnertGridworldModel import LehnertGridworldModel
from runners.runner import Runner
import config_constants as cc
import vis_utils
from model.cluster import Cluster
from utils.dataset import ArrayDataset
import constants


class LehnertGridworldExactPartitionRunner(Runner):

    def __init__(self, runner_config, model_config):

        super(LehnertGridworldExactPartitionRunner, self).__init__()

        self.model_config = model_config

        self.saver = runner_config[cc.SAVER]
        self.logger = runner_config[cc.LOGGER]
        self.load_model_path = runner_config[cc.LOAD_MODEL_PATH]
        self.round_to = runner_config[cc.ROUND_TO]
        self.hard_t = runner_config[cc.HARD_T]

    def setup(self):

        # prepare_dataset_ uses prepare_model_
        self.prepare_model_()
        self.prepare_dataset_()

    def main_training_loop(self):

        self.partition_iteration_()

    def evaluate_and_visualize(self):

        #if self.saver.has_dir():
        #    self.visualize_partition_()

        self.evaluate_purities_all_states_()

    def visualize_partition_(self):

        for idx, block in enumerate(self.partition):

            block_list = list(block)

            block_states = self.all_states[block_list]

            vis_utils.plot_5x5_grid(block_states, "block_{:d}".format(idx), self.saver)

    def evaluate_purities_all_states_(self):

        dataset = ArrayDataset({
            constants.STATE_LABELS: self.all_state_labels
        })

        cluster = Cluster(None, dataset)
        cluster.num_clusters = len(self.partition)

        probs = np.zeros((len(self.all_states), len(self.partition)))

        for block_idx, block in enumerate(self.partition):
            for state_idx in block:
                probs[state_idx, block_idx] = 1.0

        cluster_map = cluster.assign_label_to_each_cluster_soft(probs)

        purities, sizes, mean_purity = cluster.evaluate_purity_soft(probs)
        mean_i_purity, i_scores, i_totals = cluster.evaluate_inverse_purity_soft(probs, cluster_map)

        self.logger.info("state mixture soft purity: {:.2f}%".format(mean_purity * 100))
        self.logger.info("state mixture inverse soft purity: {:.2f}%".format(mean_i_purity * 100))

        self.saver.save_array_as_txt([mean_purity, mean_i_purity, len(self.partition)], "cluster_purity")

    def partition_iteration_(self):

        self.partition = {frozenset(self.all_states_indices)}
        new_partition = self.partition_improvement_()

        while new_partition != self.partition:

            self.logger.info("new partition: {:d} blocks".format(len(new_partition)))
            self.partition = new_partition
            new_partition = self.partition_improvement_()

        self.logger.info("final partition: {:d} blocks".format(len(new_partition)))

    def partition_improvement_(self):

        new_partition = cp.deepcopy(self.partition)

        for ref_block in self.partition:

            flag = True

            while flag:

                flag = False

                for block in new_partition:

                    tmp_new_partition = self.split_(block, ref_block, new_partition)

                    if tmp_new_partition != new_partition:
                        new_partition = tmp_new_partition
                        flag = True
                        break

        return new_partition

    def split_(self, block, ref_block, partition):

        partition = cp.deepcopy(partition)
        partition.remove(block)

        new_blocks = {}

        for state in block:

            reward = self.all_states_rewards[state]
            next_state = self.all_states_transitions[state]

            block_states = self.all_states[list(ref_block)]
            mask = np.sum(block_states, axis=0)

            next_state = next_state * mask[np.newaxis, :, :]
            probs = np.sum(next_state, axis=(1, 2))

            key = tuple(np.concatenate([reward, probs], axis=0))

            if key not in new_blocks.keys():
                new_blocks[key] = []

            new_blocks[key].append(state)

        for new_block in new_blocks.values():
            partition.add(frozenset(new_block))

        return partition

    def prepare_dataset_(self):

        env = LehnertGridworld()
        self.all_states, self.all_state_labels = env.get_all_states()

        self.all_states = np.array(self.all_states)
        self.all_state_labels = np.array(self.all_state_labels)

        self.all_states_indices = list(range(len(self.all_states)))

        self.all_states_rewards = []
        self.all_states_transitions = []

        for action in [LehnertGridworld.A_UP, LehnertGridworld.A_DOWN, LehnertGridworld.A_LEFT,
                       LehnertGridworld.A_RIGHT]:

            reward, next_state = self.model.predict(self.all_states, np.array([action] * len(self.all_states)))

            if self.round_to is not None:
                reward = np.round(reward, decimals=self.round_to)

            if self.hard_t:
                next_state = next_state.reshape((next_state.shape[0], next_state.shape[1] * next_state.shape[2]))
                pos = np.argmax(next_state, axis=1)
                next_state = np.zeros((next_state.shape[0], next_state.shape[1]), dtype=np.float32)
                next_state[range(len(pos)), pos] = 1.0
                next_state = next_state.reshape((next_state.shape[0], 30, 3))

            self.all_states_rewards.append(reward)
            self.all_states_transitions.append(next_state)

        # final shape: |S|x|A| and |S|x|A|x30x3
        self.all_states_rewards = np.array(self.all_states_rewards).transpose((1, 0))
        self.all_states_transitions = np.array(self.all_states_transitions).transpose((1, 0, 2, 3))

    def prepare_model_(self):

        self.model = LehnertGridworldModel(self.model_config)
        self.model.build()
        self.model.start_session()
        self.model.load(self.load_model_path)
