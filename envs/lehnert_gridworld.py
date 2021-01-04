import numpy as np


class LehnertGridworld:

    A_UP = 0
    A_DOWN = 1
    A_LEFT = 2
    A_RIGHT = 3
    ACTIONS = [A_UP, A_DOWN, A_LEFT, A_RIGHT]

    POSITIVE_REWARD = 1.0
    NEGATIVE_REWARD = 0.0

    def __init__(self, height=30, width=3, episode_length=None):

        self.height = height
        self.width = width
        self.episode_length = episode_length

        self.agent_x = None
        self.agent_y = None
        self.num_steps = None

        self.reset()

    def reset(self):

        self.agent_x = np.random.randint(0, self.height)
        self.agent_y = np.random.randint(0, self.width)
        self.num_steps = 0

        return self.get_obs()

    def step(self, action):

        assert action in self.ACTIONS
        self.num_steps += 1

        if self.agent_y == self.width - 1:
            reward = self.POSITIVE_REWARD
        else:
            reward = self.NEGATIVE_REWARD

        if action == self.A_UP:
            # up
            if self.agent_x > 0:
                self.agent_x -= 1
        elif action == self.A_DOWN:
            # down
            if self.agent_x < self.height - 1:
                self.agent_x += 1
        elif action == self.A_LEFT:
            # left
            if self.agent_y > 0:
                self.agent_y -= 1
        else:
            # right
            if self.agent_y < self.width - 1:
                self.agent_y += 1

        done = False
        if self.episode_length is not None:
            done = self.num_steps >= self.episode_length

        return self.get_obs(), reward, done, {}

    def get_obs(self):

        state = np.zeros((self.height, self.width), dtype=np.float32)
        state[self.agent_x, self.agent_y] = 1.0

        return state

    def get_bisim_state(self):
        return self.agent_y

    def get_all_states(self):

        states = []
        labels = []

        for i in range(self.height):
            for j in range(self.width):
                self.agent_x = i
                self.agent_y = j
                states.append(self.get_obs())
                labels.append(j)

        return states, labels
