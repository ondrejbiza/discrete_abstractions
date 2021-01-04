import numpy as np


class BisimPuckStack:

    @classmethod
    def get_state_block(cls, hand, layers, action_grid):

        num_positions = action_grid.shape[0] * action_grid.shape[1]
        b1 = (num_positions * (num_positions - 1)) // 2
        b2 = num_positions

        action_grid = np.reshape(action_grid, (-1, 2))

        if hand is None:

            if len(layers[1]) > 0:
                # bisim blocks type 3
                top_puck = layers[1][0]
                index = cls.get_coords(action_grid, top_puck.x, top_puck.y)

                return b1 + b2 + index

            else:
                # bisim blocks type 1
                puck_1 = layers[0][0]
                index_1 = cls.get_coords(action_grid, puck_1.x, puck_1.y)

                puck_2 = layers[0][1]
                index_2 = cls.get_coords(action_grid, puck_2.x, puck_2.y)

                index_1, index_2 = sorted([index_1, index_2])
                #index = index_1 * (num_positions - 1) + (index_2 - index_1 - 1)
                if index_1 == 0:
                    index = index_2 - 1
                else:
                    index = np.sum([num_positions - 1 - i for i in range(index_1)]) + (index_2 - index_1 - 1)

                return index

        else:
            # bisim blocks type 2
            puck = layers[0][0]
            index = cls.get_coords(action_grid, puck.x, puck.y)

            return b1 + index

    @staticmethod
    def get_coords(action_grid, x, y):
        return np.where(np.sum(action_grid == [x, y], axis=1) == 2)[0][0]
