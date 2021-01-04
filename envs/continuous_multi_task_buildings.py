import numpy as np
from skimage.transform import resize
from envs.continuous_puck_stack import ContinuousPuckStack


class CheckGoalFactory:

    SIDE_TOP = 0
    SIDE_BOTTOM = 1
    SIDE_LEFT = 2
    SIDE_RIGHT = 3

    @classmethod
    def check_goal_puck_stacking(cls, stack_height):

        def check_goal(env):

            if len(env.layers) < stack_height:
                return False

            roof = None
            for obj in env.layers[stack_height - 1]:
                # assume there is only one roof
                if isinstance(obj, ContinuousMultiTask.Roof):
                    roof = obj
                    break

            if roof is None:
                return False

            roof_pos = cls.get_pos(env, roof)

            for layer_idx in range(stack_height - 2):
                for obj in env.layers[layer_idx]:
                    pos = cls.get_pos(env, obj)
                    if pos == roof_pos:
                        break
                else:
                    return False

            return True

        return check_goal

    @classmethod
    def check_b1_side(cls, side, stack_height):

        def check_goal(env):

            if len(env.layers) < stack_height:
                return False

            roof = None
            for obj in env.layers[stack_height - 1]:
                # assume there is only one roof
                if isinstance(obj, ContinuousMultiTask.Roof):
                    roof = obj
                    break

            if roof is None:
                return False

            roof_pos = cls.get_pos(env, roof)

            for layer_idx in range(stack_height - 2):
                for obj in env.layers[layer_idx]:
                    pos = cls.get_pos(env, obj)
                    if pos == roof_pos:
                        break
                else:
                    return False

            for obj in env.layers[0]:
                pos = cls.get_pos(env, obj)

                if side == cls.SIDE_TOP:
                    if pos[0] + 1 == roof_pos[0]:
                        return True
                elif side == cls.SIDE_BOTTOM:
                    if pos[0] - 1 == roof_pos[0]:
                        return True
                elif side == cls.SIDE_LEFT:
                    if pos[1] + 1 == roof_pos[1]:
                        return True
                elif side == cls.SIDE_RIGHT:
                    if pos[1] - 1 == roof_pos[1]:
                        return True
                else:
                    raise ValueError("Invalid side.")

            return False

        return check_goal

    @classmethod
    def check_b2_side(cls, side):

        if side == cls.SIDE_TOP:
            def check_goal(env):
                return cls.check_b1_side(cls.SIDE_TOP, 2)(env) and cls.check_b1_side(cls.SIDE_BOTTOM, 2)(env)
        elif side == cls.SIDE_LEFT:
            def check_goal(env):
                return cls.check_b1_side(cls.SIDE_LEFT, 2)(env) and cls.check_b1_side(cls.SIDE_RIGHT, 2)(env)
        else:
            raise ValueError("Invalid side, only top or left.")

        return check_goal

    @staticmethod
    def get_pos(env, obj):
        pos = np.where((env.action_grid == [obj.x, obj.y]).sum(axis=2) == 2)
        return pos[0][0], pos[1][0]


class ContinuousMultiTask(ContinuousPuckStack):

    class Roof:

        def __init__(self, x, y, radius, random_shape=False):
            """
            Puck.
            :param x:           X coordinate.
            :param y:           Y coordinate.
            :param radius:      Radius.
            """

            self.x = x
            self.y = y
            self.radius = radius

            self.bottom = None
            self.top = None

            self.mask = None
            self.mask_roof()

        def mask_roof(self):

            mask_x, mask_y = np.mgrid[:self.radius * 2, :self.radius * 2]
            distance = (mask_x - self.radius) ** 2 + (mask_y - self.radius) ** 2
            self.mask = (distance < self.radius ** 2)

            self.mask = np.concatenate(
                [np.linspace(0.25, 1, self.radius), np.linspace(1, 0.25, self.radius)],
                axis=0
            )
            self.mask = np.ones((2 * self.radius, 2 * self.radius), dtype=np.float32) * self.mask

    class Puck:

        SHAPE_PUCK = 0
        SHAPE_SQUARE = 1
        SHAPE_RECTANGLE = 2
        SHAPE_PLUS = 3
        SHAPE_DONUT = 4
        SHAPES = [SHAPE_PUCK, SHAPE_SQUARE, SHAPE_RECTANGLE, SHAPE_PLUS, SHAPE_DONUT]

        def __init__(self, x, y, radius):
            """
            Puck.
            :param x:           X coordinate.
            :param y:           Y coordinate.
            :param radius:      Radius.
            """

            self.x = x
            self.y = y
            self.radius = radius

            self.bottom = None
            self.top = None

            self.mask = None
            self.mask_square()

        def mask_square(self):

            self.mask = np.ones((self.radius * 2, self.radius * 2), dtype=np.bool)

    def __init__(self, num_blocks, num_pucks, num_roofs, check_goal_fcs, num_actions, min_radius=7, max_radius=12,
                 no_contact=True, height_noise_std=0.05, render_always=False, only_one_goal=False,
                 random_shape=False, action_failure_prob=0.0):

        assert num_blocks == num_actions

        self.check_goal_fcs = check_goal_fcs
        self.only_one_goal = only_one_goal
        self.num_roofs = num_roofs

        ContinuousPuckStack.__init__(
            self, num_blocks, num_pucks, 5, num_actions, min_radius=min_radius, max_radius=max_radius,
            no_contact=no_contact, height_noise_std=height_noise_std, render_always=render_always,
            random_shape=random_shape, action_failure_prob=action_failure_prob
        )


    def reset(self):
        """
        Reset the environment.
        :return:        State.
        """

        while True:

            self.layers = [[] for _ in range(self.num_pucks + self.num_roofs)]
            self.internal_depth[:, :] = 0
            self.depth[:, :] = 0
            self.hand = None
            self.current_step = 0

            for idx in range(self.num_pucks + self.num_roofs):

                radius = np.random.randint(self.min_radius, self.max_radius)
                position = self.action_grid[np.random.randint(0, self.num_actions),
                                            np.random.randint(0, self.num_actions)]

                while not (self.check_free_ground(position, radius) and self.check_in_world(position, radius)):
                    position = self.action_grid[
                        np.random.randint(0, self.num_actions), np.random.randint(0, self.num_actions)]

                if idx >= self.num_pucks:
                    puck = self.Roof(position[0], position[1], radius)
                else:
                    puck = self.Puck(position[0], position[1], radius)

                self.layers[0].append(puck)

            if not np.any(self.check_goal()):
                # the initial state cannot reach the goal of stacking pucks, but the goals of other environments
                # that extend this class can be reached (e.g. continuous component)
                break

        self.render()

        return self.get_state()

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

    def place(self, position):
        """
        Attempt to place a puck in a position.
        :param position:    List of x and y coordinates.
        :return:            True if successful place, otherwise False.
        """

        if not self.check_in_world(position, self.hand.radius):
            return False

        if self.check_free_ground(position, self.hand.radius):
            self.hand.x = position[0]
            self.hand.y = position[1]
            self.layers[0].append(self.hand)
            self.hand = None
        else:
            # find the top puck of the stack
            top_puck_idx, top_puck_layer_idx = self.get_puck(position)

            if top_puck_idx is None or top_puck_layer_idx is None:
                # the puck would fall off of the stack
                return False

            top_puck = self.layers[top_puck_layer_idx][top_puck_idx]

            # check if top puck is really on the top
            if top_puck.top is not None:
                return False

            # check if to puck is roof
            if isinstance(top_puck, self.Roof):
                return False

            # find the bottom puck of the stack
            bottom_puck = top_puck
            while bottom_puck.bottom is not None:
                bottom_puck = bottom_puck.bottom

            # make sure the center of mass is within the top and the bottom puck
            if self.euclid(top_puck, position) > top_puck.radius or \
                    self.euclid(bottom_puck, position) > bottom_puck.radius:
                return False

            # move puck
            top_puck.top = self.hand
            self.hand.bottom = top_puck

            self.hand.x = position[0]
            self.hand.y = position[1]
            self.layers[top_puck_layer_idx + 1].append(self.hand)
            self.hand = None

        return True

    def render(self):
        """
        Render the depth image.
        :return:        None.
        """

        self.internal_depth[:, :] = 0

        for layer_idx, layer in enumerate(self.layers):
            for puck in layer:
                top_right = [puck.x - puck.radius, puck.y - puck.radius]
                bottom_left = [puck.x + puck.radius, puck.y + puck.radius]

                mask = puck.mask.astype(np.float32)

                self.internal_depth[top_right[0]: bottom_left[0], top_right[1]: bottom_left[1]] = \
                    (mask + layer_idx) + np.random.normal(0, self.height_noise_std, size=puck.mask.shape) * (mask > 0)

        if self.depth_size == self.size:
            self.depth = self.internal_depth
        else:
            self.depth = resize(
                self.internal_depth, output_shape=(self.depth_size, self.depth_size), preserve_range=True
            )
