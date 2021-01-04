import numpy as np
from envs.continuous_puck_stack import ContinuousPuckStack


class CheckGoalFactory:

    @staticmethod
    def check_goal_puck_stacking(stack_height):

        def check_goal(env):

            good_stack = False

            if len(env.layers) >= stack_height:
                good_stack = len(env.layers[stack_height - 1]) > 0

            good_position = True
            if good_stack and env.check_x_y is not None:
                good_position = env.layers[stack_height - 1][0].x == env.check_x_y[0] and \
                                env.layers[stack_height - 1][0].y == env.check_x_y[1]

            return good_stack and good_position

        return check_goal

    @staticmethod
    def check_goal_two_stacks(stack_height):

        def check_goal(env):

            # check if there are two stacks of the target height
            if len(env.layers) < stack_height or len(env.layers[stack_height - 1]) != 2:
                return False

            return True

        return check_goal

    @staticmethod
    def check_goal_stairs(num_pucks):

        def check_goal(env):

            if num_pucks == 3:

                # check if there's a stack of 2
                if len(env.layers[1]) != 1:
                    return False

                top_puck = env.layers[1][0]

                # check if the third puck is close to the stack of 2
                if len(env.layers[0]) != 2:
                    return False

                if env.layers[0][0].top is None:
                    third_puck = env.layers[0][0]
                elif env.layers[0][1].top is None:
                    third_puck = env.layers[0][1]
                else:
                    raise ValueError("The third puck is obstructed; shouldn't happen.")

                dist = env.euclid(top_puck, [third_puck.x, third_puck.y])

                if dist > top_puck.radius + third_puck.radius + 14:
                    return False

                return True

            else:

                # check if there's a stack of 3
                if len(env.layers[2]) != 1:
                    return False

                # check if there's a stack of 2
                if len(env.layers[1]) != 2:
                    return False

                # check if the sixth puck is on the ground
                if len(env.layers[0]) != 3:
                    return False

                # check if the stack of 2 is close to the stack of 3
                top_puck = env.layers[2][0]

                if env.layers[1][0].top is None:
                    second_top_puck = env.layers[1][0]
                elif env.layers[1][1].top is None:
                    second_top_puck = env.layers[1][1]
                else:
                    raise ValueError("The second top puck is obstructed; shouldn't happen.")

                dist = env.euclid(top_puck, [second_top_puck.x, second_top_puck.y])

                if dist > top_puck.radius + second_top_puck.radius + env.tolerance:
                    return False

                # check if the sixth puck is close to the stack of 2
                if env.layers[0][0].top is None:
                    sixth_puck = env.layers[0][0]
                elif env.layers[0][1].top is None:
                    sixth_puck = env.layers[0][1]
                elif env.layers[0][2].top is None:
                    sixth_puck = env.layers[0][2]
                else:
                    raise ValueError("The sixth puck is obstructed; shouldn't happen.")

                dist = env.euclid(second_top_puck, [sixth_puck.x, sixth_puck.y])

                if dist > second_top_puck.radius + sixth_puck.radius + env.tolerance:
                    return False

                return True

        return check_goal

    @staticmethod
    def check_goal_row(num_pucks):

        def check_goal(env):

            pucks = []
            for i in env.layers:
                for j in i:
                    pucks.append(j)

            xs = []
            ys = []

            for puck in pucks:
                pos = np.where((env.action_grid == [puck.x, puck.y]).sum(axis=2) == 2)
                xs.append(pos[0][0])
                ys.append(pos[1][0])

            xs = np.array(xs)
            ys = np.array(ys)

            for x in np.unique(xs):
                tmp_ys = ys[xs == x]
                longest_seq = 1
                sorted_tmp_ys = list(sorted(tmp_ys))

                for i in range(len(sorted_tmp_ys) - 1):
                    if sorted_tmp_ys[i] == sorted_tmp_ys[i + 1] - 1:
                        longest_seq += 1
                    else:
                        longest_seq = 1

                    if longest_seq == num_pucks:
                        return True

            for y in np.unique(ys):
                tmp_xs = xs[ys == y]
                longest_seq = 1
                sorted_tmp_xs = list(sorted(tmp_xs))

                for i in range(len(sorted_tmp_xs) - 1):
                    if sorted_tmp_xs[i] == sorted_tmp_xs[i + 1] - 1:
                        longest_seq += 1
                    else:
                        longest_seq = 1

                    if longest_seq == num_pucks:
                        return True

            return False

        return check_goal

    @staticmethod
    def check_goal_row_diag(num_pucks):

        def check_goal(env):

            pucks = []
            for i in env.layers:
                for j in i:
                    pucks.append(j)

            xs = []
            ys = []

            for puck in pucks:
                pos = np.where((env.action_grid == [puck.x, puck.y]).sum(axis=2) == 2)
                xs.append(pos[0][0])
                ys.append(pos[1][0])

            xs = np.array(xs)
            ys = np.array(ys)

            xs_sorted_idx = np.argsort(xs)

            longest_seq_a = 1
            longest_seq_b = 1

            for i in range(len(xs) - 1):

                j_1 = xs_sorted_idx[i]
                j_2 = xs_sorted_idx[i + 1]

                if xs[j_1] == xs[j_2] - 1:
                    if ys[j_1] == ys[j_2] - 1:
                        longest_seq_a += 1
                    else:
                        longest_seq_a = 1

                    if ys[j_1] == ys[j_2] + 1:
                        longest_seq_b += 1
                    else:
                        longest_seq_b = 1

                    if longest_seq_a == num_pucks or longest_seq_b == num_pucks:
                        return True

            return False

        return check_goal


class ContinuousMultiTask(ContinuousPuckStack):

    def __init__(self, num_blocks, num_pucks, check_goal_fcs, num_actions, min_radius=7, max_radius=12, tolerance=8,
                 no_contact=True, height_noise_std=0.05, render_always=False, only_one_goal=False,
                 random_shape=False, action_failure_prob=0.0):

        assert num_blocks == num_actions

        self.check_goal_fcs = check_goal_fcs
        self.only_one_goal = only_one_goal

        ContinuousPuckStack.__init__(
            self, num_blocks, num_pucks, 5, num_actions, min_radius=min_radius, max_radius=max_radius,
            no_contact=no_contact, height_noise_std=height_noise_std, render_always=render_always,
            random_shape=random_shape, action_failure_prob=action_failure_prob
        )

    def step(self, action):

        if self.hand is None:
            if action >= (self.num_actions ** 2):
                change = False
            else:
                position = self.action_grid[action // self.num_actions, action % self.num_actions]
                change = self.pick(position)
        else:
            if action < (self.num_actions ** 2):
                change = False
            else:
                action = action % (self.num_actions ** 2)
                position = self.action_grid[action // self.num_actions, action % self.num_actions]
                change = self.place(position)

        self.current_step += 1

        dones = self.check_goal()
        rewards = np.zeros(len(dones), np.float32) + self.NEGATIVE_REWARD
        rewards[dones] = self.POSITIVE_REWARD

        self.render()

        if self.current_step >= self.max_steps:
            dones[:] = True

        if self.only_one_goal:
            return self.get_state(), rewards[0], dones[0], {}
        else:
            return self.get_state(), rewards, dones, {}

    def check_goal(self):

        return np.array([check_goal_fc(self) for check_goal_fc in self.check_goal_fcs], dtype=np.bool)
