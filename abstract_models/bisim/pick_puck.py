import numpy as np


class BisimPickPuckModel:

    @classmethod
    def get_state_block(cls, hand, layers, action_grid):

        num_positions = action_grid.shape[0] * action_grid.shape[1]
        action_grid = np.reshape(action_grid, (-1, 2))

        if hand is not None:

            return num_positions

        else:

            puck = layers[0][0]
            index = cls.get_coords(action_grid, puck.x, puck.y)

            return index

    @staticmethod
    def get_coords(action_grid, x, y):

        return np.where(np.sum(action_grid == [x, y], axis=1) == 2)[0][0]
