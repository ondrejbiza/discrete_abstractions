import copy as cp
import numpy as  np
from runners.bisim.lehnert_gridworld.LehnertGridworldExactPartition import LehnertGridworldExactPartitionRunner
import config_constants as cc


class LehnertGridworldApproxPartitionRunner(LehnertGridworldExactPartitionRunner):

    NUM_ACTIONS = 4

    def __init__(self, runner_config, model_config):

        super(LehnertGridworldApproxPartitionRunner, self).__init__(runner_config, model_config)

        self.epsilon = runner_config[cc.EPSILON]

    def partition_iteration_(self):

        self.partition = {frozenset(self.all_states_indices)}

        # split rewards
        for action in range(self.NUM_ACTIONS):

            flag = True

            while flag:

                flag = False

                for block in self.partition:

                    new_partition = self.split_rewards_(block, action, self.partition)

                    if new_partition != self.partition:
                        self.partition = new_partition
                        flag = True
                        break

        self.logger.info("{:d} blocks after reward splitting".format(len(self.partition)))

        # split transitions
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

                    for action in range(self.NUM_ACTIONS):

                        tmp_new_partition = self.split_transitions_(block, action, ref_block, new_partition)

                        if tmp_new_partition != new_partition:
                            new_partition = tmp_new_partition
                            flag = True
                            break

                    if flag:
                        break

        return new_partition

    def split_rewards_(self, block, action, partition):

        partition = cp.deepcopy(partition)
        partition.remove(block)

        states = list(block)
        state_rewards = self.all_states_rewards[:, action][states]
        assert len(states) == len(state_rewards) == len(block)

        sorted_states = np.argsort(state_rewards)
        sorted_state_rewards = state_rewards[sorted_states]
        assert len(sorted_states) == len(sorted_state_rewards) == len(block)

        start_idx = 0
        new_blocks = [[]]

        for idx in range(len(sorted_states)):

            start_reward = sorted_state_rewards[start_idx]
            current_reward = sorted_state_rewards[idx]

            if np.abs(start_reward - current_reward) > self.epsilon:
                new_blocks.append([])
                start_idx = idx

            new_blocks[-1].append(states[sorted_states[idx]])

        for new_block in new_blocks:
            partition.add(frozenset(new_block))

        return partition

    def split_transitions_(self, block, action, ref_block, partition):

        partition = cp.deepcopy(partition)
        partition.remove(block)

        block_states = self.all_states[list(ref_block)]
        mask = np.sum(block_states, axis=0)

        states = list(block)
        next_states = self.all_states_transitions[:, action][states]
        next_states = next_states * mask[np.newaxis, :, :]
        probs = np.sum(next_states, axis=(1, 2))

        sorted_states = np.argsort(probs)
        sorted_state_probs = probs[sorted_states]

        start_idx = 0
        new_blocks = [[]]

        for idx in range(len(sorted_states)):

            start_reward = sorted_state_probs[start_idx]
            current_reward = sorted_state_probs[idx]

            if np.abs(start_reward - current_reward) > self.epsilon:
                new_blocks.append([])
                start_idx = idx

            new_blocks[-1].append(states[sorted_states[idx]])

        for new_block in new_blocks:
            partition.add(frozenset(new_block))

        return partition
