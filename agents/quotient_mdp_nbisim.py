import copy as cp
import numpy as np
from scipy.special import softmax


class QuotientMDPNBisim:

    def __init__(self, classify, env, q_values, minatar=False, softmax_policy=False, softmax_policy_temp=1.0):

        self.classify = classify
        self.env = env
        self.q_values = q_values
        self.exploration_schedule = None
        self.minatar = minatar
        self.softmax_policy = softmax_policy
        self.softmax_policy_temp = softmax_policy_temp

    def act(self, state, timestep):

        state = cp.deepcopy(state)
        state_distribution = self.classify(state)

        action_values = np.sum(self.q_values * state_distribution[:, np.newaxis], axis=0)

        if self.softmax_policy:
            action_values = softmax(action_values / self.softmax_policy_temp, axis=0)
            action = int(np.random.choice(list(range(len(action_values))), p=action_values / np.sum(action_values)))
        else:
            action = int(np.argmax(action_values))

        if not self.minatar and state[1] == 1:
            action += self.q_values.shape[1]

        if self.minatar:
            reward, done = self.env.act(action)
            new_state = self.env.state()
        else:
            new_state, reward, done, _ = self.env.step(action)
            new_state = cp.deepcopy(new_state)

        return state, state_distribution, None, action, new_state, None, None, None, reward, done
