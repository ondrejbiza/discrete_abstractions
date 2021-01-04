import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from gym import spaces


class ContinuousPuckStack:

    BLOCK_SIZE = 28
    POSITIVE_REWARD = 10
    NEGATIVE_REWARD = 0

    class Puck:

        SHAPE_PUCK = 0
        SHAPE_SQUARE = 1
        SHAPE_RECTANGLE = 2
        SHAPE_PLUS = 3
        SHAPE_DONUT = 4
        SHAPES = [SHAPE_PUCK, SHAPE_SQUARE, SHAPE_RECTANGLE, SHAPE_PLUS, SHAPE_DONUT]

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

            if not random_shape:
                self.mask_puck()
            else:
                shape = np.random.choice(self.SHAPES)

                if shape == self.SHAPE_PUCK:
                    self.mask_puck()
                elif shape == self.SHAPE_SQUARE:
                    self.mask_square()
                elif shape == self.SHAPE_RECTANGLE:
                    self.mask_rectangle()
                elif shape == self.SHAPE_PLUS:
                    self.mask_plus()
                elif shape == self.SHAPE_DONUT:
                    self.mask_donut()

        def mask_puck(self):

            mask_x, mask_y = np.mgrid[:self.radius * 2, :self.radius * 2]
            distance = (mask_x - self.radius) ** 2 + (mask_y - self.radius) ** 2
            self.mask = (distance < self.radius ** 2)

        def mask_square(self):

            self.mask = np.ones((self.radius * 2, self.radius * 2), dtype=np.bool)

        def mask_rectangle(self):

            self.mask = np.ones((self.radius * 2, self.radius * 2), dtype=np.bool)

            frac = int(self.radius / 3)

            if np.random.choice([0, 1]) == 0:
                self.mask[:frac] = False
                self.mask[-frac:] = False
            else:
                self.mask[:, :frac] = False
                self.mask[:, -frac:] = False

        def mask_plus(self):
            self.mask = np.zeros((self.radius * 2, self.radius * 2), dtype=np.bool)

            frac = int(self.radius / 1.5)

            self.mask[frac: -frac] = True
            self.mask[:, frac: -frac] = True

        def mask_donut(self):

            self.mask = np.ones((self.radius * 2, self.radius * 2), dtype=np.bool)

            frac = int(self.radius / 2)

            self.mask[frac:-frac, frac:-frac] = False


    def __init__(self, num_blocks, num_pucks, goal, num_actions, min_radius=7, max_radius=12, no_contact=True,
                 height_noise_std=0.0, render_always=False, depth_size=None, check_x_y=None, random_shape=False,
                 action_failure_prob=0.0):
        """
        Initialize the puck stacking environment.
        :param num_blocks:      Number of blocks.
        :param num_pucks:       Number of pucks.
        :param goal:            Number of pucks to stack.
        :param num_actions:     Number of actions.
        :param min_radius:      Minimum puck radius.
        :param max_radius:      Maximum puck radius.
        :param no_contact:      Do not allow contact between pucks.
        :param render_always:   Re-render the depth image even if the state does not change to better
                                simulate the noisy in the observation.
        """

        self.num_blocks = num_blocks
        self.num_pucks = num_pucks
        self.goal = goal
        self.num_actions = num_actions
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.no_contact = no_contact
        self.height_noise_std = height_noise_std
        self.render_always = render_always
        self.depth_size = depth_size
        self.check_x_y = check_x_y
        self.random_shape = random_shape
        self.action_failure_prob = action_failure_prob

        self.size = self.num_blocks * self.BLOCK_SIZE
        self.max_steps = max(20, 5 * goal)
        self.hand = None
        if self.depth_size is None:
            self.depth_size = self.size

        self.internal_depth = np.zeros((self.size, self.size), dtype=np.float32)
        self.depth = np.zeros((self.depth_size, self.depth_size), dtype=np.float32)

        self.observation_space = spaces.Tuple(
            [spaces.Box(np.zeros([self.depth_size, self.depth_size, 1]), np.ones([self.depth_size, self.depth_size, 1]),
                        dtype=np.float32),
             spaces.Discrete(2)])

        self.action_grid = None
        self.create_action_grid()

        self.layers = None
        self.pucks = None
        self.current_step = None
        self.reset()

    def create_action_grid(self):
        """
        Create the environment's grid.
        :return:        None.
        """

        size = self.num_blocks * self.BLOCK_SIZE
        step = size / self.num_actions
        half_step = step / 2

        self.action_grid = np.zeros((self.num_actions, self.num_actions, 2), dtype=np.int32)

        for i in range(self.num_actions):
            for j in range(self.num_actions):
                x = half_step + i * step
                y = half_step + j * step
                self.action_grid[i, j] = [x, y]

    def reset(self):
        """
        Reset the environment.
        :return:        State.
        """

        while True:

            self.layers = [[] for _ in range(self.num_pucks)]
            self.internal_depth[:, :] = 0
            self.depth[:, :] = 0
            self.hand = None
            self.current_step = 0

            for _ in range(self.num_pucks):

                radius = np.random.randint(self.min_radius, self.max_radius)
                position = self.action_grid[np.random.randint(0, self.num_actions),
                                            np.random.randint(0, self.num_actions)]

                while not (self.check_free_ground(position, radius) and self.check_in_world(position, radius)):
                    position = self.action_grid[
                        np.random.randint(0, self.num_actions), np.random.randint(0, self.num_actions)]

                puck = self.Puck(position[0], position[1], radius, random_shape=self.random_shape)
                self.layers[0].append(puck)

            if not np.any(self.check_goal()):
                # the initial state cannot reach the goal of stacking pucks, but the goals of other environments
                # that extend this class can be reached (e.g. continuous component)
                break

        self.render()

        return self.get_state()

    def step(self, action):
        """
        Execute an action in the environment.
        :param action:      Action index.
        :return:            Next state, reward, done, and an empty dictionary (legacy reasons).
        """

        if self.action_failure_prob is not None and self.action_failure_prob > 0.0 and \
            np.random.uniform(0, 1) < self.action_failure_prob:
            change = False
        else:
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

        if change:
            self.render()
            if self.check_goal():
                return self.get_state(), self.POSITIVE_REWARD, True, {}
        elif self.render_always:
            self.render()

        if self.current_step >= self.max_steps:
            return self.get_state(), self.NEGATIVE_REWARD, True, {}
        else:
            return self.get_state(), self.NEGATIVE_REWARD, False, {}

    def pick(self, position):
        """
        Pick a puck in a position.
        :param position:    List of x and y coordinates.
        :return:            True if successful pick, False otherwise.
        """

        # find puck
        puck_idx, layer_idx = self.get_puck(position)

        if puck_idx is None:
            return False

        puck = self.layers[layer_idx][puck_idx]

        # check if the puck is on top
        if not self.check_on_top(position, puck.radius, layer_idx):
            return False

        # update stack pointers
        if puck.bottom is not None:
            puck.bottom.top = None
            puck.bottom = None

        # move puck
        self.hand = puck
        del self.layers[layer_idx][puck_idx]

        return True

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

    def get_state(self):
        """
        Get the state of the environment.
        :return:    Tuple of a depth image with the shape (size, size, 1) and a binary hand state.
        """

        return np.expand_dims(self.depth, axis=-1), int(self.hand is not None)

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
                self.internal_depth[top_right[0]: bottom_left[0], top_right[1]: bottom_left[1]][puck.mask] = \
                    np.random.normal(layer_idx + 1, self.height_noise_std, size=puck.mask.shape)[puck.mask]

        if self.depth_size == self.size:
            self.depth = self.internal_depth
        else:
            self.depth = resize(
                self.internal_depth, output_shape=(self.depth_size, self.depth_size), preserve_range=True
            )

    def check_free_ground(self, position, radius):
        """
        Check if the ground is free to place a puck.
        :param position:    Place position.
        :param radius:      Puck radius.
        :return:            True if valid place, False otherwise.
        """

        for layer in self.layers:
            for puck in layer:
                dist = self.euclid(puck, position)

                if self.no_contact:
                    if dist <= radius + puck.radius:
                        return False
                else:
                    if dist < radius + puck.radius:
                        return False

        return True

    def check_on_top(self, position, radius, layer_idx):
        """
        Check if a puck in a given position with a given radius is on top of a stack.
        :param position:        Puck position.
        :param radius:          Puck radius.
        :param layer_idx:       Puck height.
        :return:                True if on top, False otherwise.
        """

        if len(self.layers) > layer_idx + 1:
            # check each puck in the layer above
            for puck in self.layers[layer_idx + 1]:
                dist = self.euclid(puck, position)

                if dist < radius + puck.radius:
                    return False

        return True

    def check_in_world(self, position, radius):
        """
        Check if placing a puck in a certain position does not overflow the boundaries of the world.
        :param position:    Place position.
        :param radius:      Puck radius.
        :return:            True if valid, False otherwise.
        """

        if position[0] - radius < 0 or position[1] - radius < 0:
            return False

        if position[0] + radius >= self.size or position[1] + radius >= self.size:
            return False

        return True

    def check_goal(self):
        """
        Check if the goal was reached.
        :return:    True if goal reached, False otherwise.
        """

        good_stack = len(self.layers[self.goal - 1]) > 0

        good_position = True
        if good_stack and self.check_x_y is not None:
            good_position = self.layers[self.goal - 1][0].x == self.check_x_y[0] and \
                            self.layers[self.goal - 1][0].y == self.check_x_y[1]

        return good_stack and good_position

    def get_puck(self, position):
        """
        Get the top pucks at a position.
        :param position:    List of x and y coordinates.
        :return:            Puck index in a layer and its height or two Nones if nothing found.
        """

        for layer_idx in reversed(range(len(self.layers))):
            for puck_idx, puck in enumerate(self.layers[layer_idx]):
                if self.euclid(puck, position) < puck.radius:
                    return puck_idx, layer_idx

        return None, None

    def show_state(self, save_path=None, draw_grid=False):

        image = cp.deepcopy(self.depth)

        if draw_grid:
            for i in range(1, self.num_blocks):
                image[i * 28: i * 28 + 1] = 0.5
                image[:, i * 28: i * 28 + 1] = 0.5
            image[:1, :] = 0.5
            image[-1:, :] = 0.5
            image[:, :1] = 0.5
            image[:, -1:] = 0.5

        fig = plt.figure()
        plt.imshow(3 - image, vmin=0, vmax=3, cmap="gray")
        plt.axis("off")
        plt.tight_layout()

        coords = []

        def onclick(event):
            x, y = event.xdata, event.ydata
            coords.append([x, y])
            plt.close()

        fig.canvas.mpl_connect("button_press_event", onclick)

        if save_path is not None:
            plt.imsave(save_path, 3 - image, vmin=0, vmax=3, cmap="gray")

        plt.show()

        coords = coords[-1]

        dists = np.sum(np.square(self.action_grid - np.array([coords[1], coords[0]])), axis=-1)
        action_pos = np.unravel_index(dists.argmin(), dists.shape)

        return action_pos

    @staticmethod
    def euclid(puck, position):
        """
        Calculate the euclidean distance between a puck and a position.
        :param puck:        Puck with x and y coordinates.
        :param position:    List of x and y coordinates.
        :return:            Euclidean distance.
        """

        return np.sqrt((puck.x - position[0]) ** 2 + (puck.y - position[1]) ** 2)
