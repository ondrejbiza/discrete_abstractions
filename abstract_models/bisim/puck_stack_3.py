import numpy as np


class BisimPuckStack:

    @classmethod
    def get_state_block(cls, hand, layers, action_grid):

        num_positions = action_grid.shape[0] * action_grid.shape[1]

        # three pucks in the grid world
        b1 = cls.get_num_combinations(num_positions, 3)
        # two pucks in the grid world
        b2 = cls.get_num_combinations(num_positions, 2)
        # one stack of two pucks and one singular puck in the grid world
        b3 = num_positions * (num_positions - 1)
        # one stack of two pucks in the grid world
        b4 = num_positions

        action_grid = np.reshape(action_grid, (-1, 2))

        if hand is None:

            if len(layers[2]) > 0:
                # bisim blocks type 5
                top_puck = layers[2][0]
                index = cls.get_coords(action_grid, top_puck.x, top_puck.y)

                return b1 + b2 + b3 + b4 + index

            elif len(layers[1]) > 0:

                # bisim blocks type 3
                top_puck = layers[1][0]
                top_puck_index = cls.get_coords(action_grid, top_puck.x, top_puck.y)

                other_puck_1 = layers[0][0]
                other_puck_index_1 = cls.get_coords(action_grid, other_puck_1.x, other_puck_1.y)

                other_puck_2 = layers[0][1]
                other_puck_index_2 = cls.get_coords(action_grid, other_puck_2.x, other_puck_2.y)

                if other_puck_index_1 == top_puck_index:
                    other_puck_index = other_puck_index_2
                else:
                    other_puck_index = other_puck_index_1

                if top_puck_index == 0:
                    index = other_puck_index - 1
                else:
                    if other_puck_index > top_puck_index:
                        index = top_puck_index * (num_positions - 1) + other_puck_index - 1
                    else:
                        index = top_puck_index * (num_positions - 1) + other_puck_index

                return b1 + b2 + index

            else:

                # bisim blocks type 1
                puck_1 = layers[0][0]
                index_1 = cls.get_coords(action_grid, puck_1.x, puck_1.y)

                puck_2 = layers[0][1]
                index_2 = cls.get_coords(action_grid, puck_2.x, puck_2.y)

                puck_3 = layers[0][2]
                index_3 = cls.get_coords(action_grid, puck_3.x, puck_3.y)

                index_1, index_2, index_3 = sorted([index_1, index_2, index_3])

                index = 0
                for i in range(index_1):
                    j = num_positions - (i + 1)
                    index += int(j * (j - 1) / 2)

                for i in range(index_1 + 1, index_2):
                    index += num_positions - (i + 1)

                index += index_3 - index_2 - 1

                return index

        else:

            if len(layers[1]) > 0:

                # bisim blocks type 4
                top_puck = layers[1][0]
                top_puck_index = cls.get_coords(action_grid, top_puck.x, top_puck.y)

                return b1 + b2 + b3 + top_puck_index

            else:

                # bisim blocks type 2
                puck_1 = layers[0][0]
                index_1 = cls.get_coords(action_grid, puck_1.x, puck_1.y)

                puck_2 = layers[0][1]
                index_2 = cls.get_coords(action_grid, puck_2.x, puck_2.y)

                index_1, index_2 = sorted([index_1, index_2])

                if index_1 == 0:
                    index = index_2 - 1
                else:
                    index = np.sum([num_positions - 1 - i for i in range(index_1)]) + (index_2 - index_1 - 1)

                return b1 + index

    @staticmethod
    def get_num_combinations(num_positions, num_pucks):
        t1 = int(np.product([num_positions - i for i in range(num_pucks)]))
        t2 = np.math.factorial(num_pucks)
        return int(t1 / t2)

    @staticmethod
    def get_coords(action_grid, x, y):
        return np.where(np.sum(action_grid == [x, y], axis=1) == 2)[0][0]
