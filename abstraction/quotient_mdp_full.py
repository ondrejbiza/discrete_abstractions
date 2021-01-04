from io import BytesIO
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import constants
from agents import utils
from utils.dataset import ArrayDataset


class QuotientMDPFull:

    # quotient MDP plot
    NODE_FILL_COLOR = "#E1D5E7"
    NODE_BORDER_COLOR = "#9674A6"
    STANDARD_EDGE_COLOR = "#666666"
    REWARD_EDGE_COLOR = "#8CB972"
    RESOLUTION = 500  # dpi

    # value iteration
    DIFF_THRESHOLD = 0.001
    MAX_STEPS = 1000

    def __init__(self, states, actions, transitions, rewards, discount=0.9, state_rewards=False,
                 absorbing_state_threshold=0.8):

        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount = discount
        self.state_rewards = state_rewards
        self.absorbing_state_threshold = absorbing_state_threshold

        self.reset()

    def reset(self):

        self.state_action_values = np.zeros((len(self.states), len(self.actions)), dtype=np.float32)
        self.policy = np.zeros(len(self.states), dtype=np.int32)

    def value_iteration(self):

        max_diff = self.DIFF_THRESHOLD + 1
        step = 0

        while max_diff > self.DIFF_THRESHOLD and step < self.MAX_STEPS:

            new_state_action_values = np.zeros_like(self.state_action_values)

            for state_idx in range(len(self.states)):

                for action_idx in range(len(self.actions)):

                    transition = self.transitions[state_idx, :, action_idx]

                    if self.state_rewards:
                        reward = self.rewards[state_idx]
                    else:
                        reward = self.rewards[state_idx, action_idx]

                    if self.absorbing_state_threshold is not None and reward > self.absorbing_state_threshold:
                        new_state_action_values[state_idx, action_idx] = reward
                    else:
                        new_state_action_values[state_idx, action_idx] = \
                            reward + self.discount * \
                            np.sum(transition * np.max(self.state_action_values, axis=1))

            max_diff = np.max(np.abs(self.state_action_values - new_state_action_values))

            self.state_action_values = new_state_action_values
            step += 1

        return step

    def clip_matrices(self):

        self.transitions[self.transitions < 0] = 0
        #self.transitions[self.transitions > 1] = 1

    def normalize_transitions(self, softmax=False):

        if softmax:
            self.transitions = self.softmax(self.transitions, 1)
        else:
            sums = np.sum(self.transitions, axis=1)
            sums[sums == 0] = 1
            self.transitions = self.transitions / sums[:, np.newaxis, :]

    def plot_mdp(self, save_path=None, return_figure=False):

        import pydot
        graph = pydot.Dot(graph_type="digraph")

        for state_idx in range(len(self.states)):

            state_name = str(state_idx)

            node = pydot.Node(state_name, fillcolor=self.NODE_FILL_COLOR, color=self.NODE_BORDER_COLOR,
                              style="filled")
            graph.add_node(node)

            for action_idx in range(len(self.actions)):

                next_state = np.argmax(self.transitions[state_idx, :, action_idx])
                next_state_name = str(next_state)

                if self.state_rewards:
                    reward = self.rewards[state_idx]
                else:
                    reward = self.rewards[state_idx, action_idx]

                if reward > self.absorbing_state_threshold:
                    color = self.REWARD_EDGE_COLOR
                    next_state_name = "goal"
                else:
                    color = self.STANDARD_EDGE_COLOR

                edge = pydot.Edge(
                    state_name, next_state_name, label="{:d}: {:.2f}".format(
                        action_idx, self.state_action_values[state_idx, action_idx]
                    ), color=color
                )
                graph.add_edge(edge)

        if save_path is not None:
            assert save_path.split(".")[-1] == "pdf"
            graph.write_pdf(save_path)

        if return_figure:
            # return figure so that we can save more figures into the same pdf file
            png = graph.create_png(prog=["dot", "-Gdpi={:d}".format(self.RESOLUTION)])

            bio = BytesIO()
            bio.write(png)
            bio.seek(0)
            img = mpimg.imread(bio)

            fig, ax = plt.subplots()
            ax.imshow(img, aspect="equal")
            ax.axis("off")

            return fig

    def learn_model(self, state_probs, next_state_probs, actions, rewards, dones, rlr=0.01, tlr=0.5, num_steps=500,
                    ad_hoc_loss=False, batch_size=1000, opt=constants.OPT_ADAM):

        dataset = ArrayDataset({
            constants.STATE_PROBABILITIES: state_probs, constants.NEXT_STATE_PROBABILITIES: next_state_probs,
            constants.ACTIONS: actions, constants.REWARDS: rewards, constants.DONES: dones
        })

        with tf.variable_scope("learn_model"):

            R = tf.get_variable(
                "reward_matrix", shape=(len(self.actions), len(self.states)), dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
            )

            T = tf.get_variable(
                "transition_matrix", shape=(len(self.actions), len(self.states), len(self.states)), dtype=tf.float32,
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1, dtype=tf.float32)
            )
            T_softmax = tf.nn.softmax(T, axis=2)
            T_logsoftmax = tf.nn.log_softmax(T, axis=2)

            state_probs_pl = tf.placeholder(dtype=tf.float32, name="state_probs_pl")
            next_state_probs_pl = tf.placeholder(dtype=tf.float32, name="next_state_probs_pl")
            actions_pl = tf.placeholder(dtype=tf.int32, name="actions_pl")
            rewards_pl = tf.placeholder(dtype=tf.float32, name="rewards_pl")
            dones_pl = tf.placeholder(dtype=tf.bool, name="dones_pl")

            R_gather = tf.gather(R, actions_pl)
            T_softmax_gather = tf.gather(T_softmax, actions_pl)
            T_logsoftmax_gather = tf.gather(T_logsoftmax, actions_pl)

            if ad_hoc_loss:
                reward_loss = (1 / 2) * tf.reduce_mean(
                    tf.square(rewards_pl - tf.reduce_sum(R_gather * state_probs_pl, axis=1)), axis=0
                )

                transition_loss = (1 / 2) * tf.reduce_mean(tf.reduce_sum(
                    tf.square(
                        next_state_probs_pl - tf.matmul(tf.transpose(T_softmax_gather, perm=[0, 2, 1]),
                                                       state_probs_pl[:, :, tf.newaxis])[:, :, 0]
                    ), axis=1
                ), axis=0)
            else:
                reward_loss = (1 / 2) * tf.reduce_mean(
                    tf.reduce_sum(tf.square(rewards_pl[:, tf.newaxis] - R_gather) * state_probs_pl, axis=1), axis=0
                )

                transition_loss = - tf.reduce_mean(tf.reduce_sum(
                    state_probs_pl[:, :, tf.newaxis] * next_state_probs_pl[:, tf.newaxis, :] * T_logsoftmax_gather,
                    axis=[1, 2]
                ) * (1 - tf.cast(dones_pl, tf.float32)), axis=0)

            R_step = utils.get_optimizer(opt, rlr).minimize(reward_loss)
            T_step = utils.get_optimizer(opt, tlr).minimize(transition_loss)

            losses = []
            epoch_size = max(len(state_probs) // batch_size, 1)

            with tf.Session() as sess:

                sess.run(tf.variables_initializer(
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="learn_model"))
                )

                for i in range(num_steps):

                    epoch_step = i % epoch_size

                    if epoch_step == 0:
                        dataset.shuffle()

                    b = np.index_exp[epoch_step * batch_size: (epoch_step + 1) * batch_size]

                    _, _, tmp_reward_loss, tmp_transition_loss = sess.run(
                        [R_step, T_step, reward_loss, transition_loss], feed_dict={
                            state_probs_pl: dataset[constants.STATE_PROBABILITIES][b],
                            next_state_probs_pl: dataset[constants.NEXT_STATE_PROBABILITIES][b],
                            actions_pl: dataset[constants.ACTIONS][b],
                            rewards_pl: dataset[constants.REWARDS][b],
                            dones_pl: dataset[constants.DONES][b]
                        }
                    )

                    losses.append([tmp_reward_loss, tmp_transition_loss])

                self.rewards = np.transpose(sess.run(R), axes=[1, 0])
                self.transitions = np.transpose(sess.run(T_softmax), axes=[1, 2, 0])

            losses = np.stack(losses, axis=0)

        return losses

    @staticmethod
    def softmax(x, axis):
        e_x = np.exp(x - np.max(x, axis=axis))
        return e_x / e_x.sum(axis=axis)
