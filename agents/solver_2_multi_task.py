import copy as cp
import os, time, operator
import numpy as np
import vis_utils


class SolverMultiTask:

    def __init__(self, env, agent, max_time_steps, learning_start, train_freq, active_task_indices, max_episodes=None,  rewards_file=None,
                 animate=False, animate_from=0, gif_save_path=None, gif_save_limit=None, gif_save_only_successful=False,
                 gif_save_from=None, max_depth_value=3, print_freq=1, train=True, remember=True, collect_data=None,
                 collect_pre=None, collect_post=None, early_stop_threshold=None):
        """
        Initialize a solver.
        :param env:                             An environment.
        :param agent:                           An agent for the environment.
        :param max_time_steps:                  Maximum time steps to run the agent for.
        :param learning_start:                  Learning starts at this time step.
        :param train_freq:                      Train every N time steps.
        :param max_episodes:                    Maximum episodes to run for.
        :param rewards_file:                    Where to save per-episode rewards.
        :param animate:                         Show an animation of the environment.
        :param animate_from:                    Start the animation after this time step.
        :param gif_save_path:                   Save gifs of episodes.
        :param gif_save_limit:                  Maximum number of gifs to save.
        :param gif_save_only_successful:        Save only gifs of successful episodes.
        :param max_depth_value:                 Maximum depth values; used only for visualization purposes.
        :param print_freq:                      How often to print status.
        :param train:                           Train the agent.
        """

        self.env = env
        self.agent = agent
        self.max_time_steps = max_time_steps
        self.learning_start = learning_start
        self.train_freq = train_freq
        self.active_task_indices = active_task_indices
        self.max_episodes = max_episodes
        self.rewards_file = rewards_file
        self.animate = animate
        self.animate_from = animate_from
        self.gif_save_path = gif_save_path
        self.gif_save_limit = gif_save_limit
        self.gif_save_only_successful = gif_save_only_successful
        self.gif_save_from = gif_save_from
        self.max_depth_value = max_depth_value
        self.print_freq = print_freq
        self.train = train
        self.remember = remember
        self.collect_data = collect_data
        self.collect_pre = collect_pre
        self.collect_post = collect_post
        self.early_stop_threshold = early_stop_threshold

        self.episode_rewards = None
        self.actions_to_save = None
        self.animator = None
        self.gif_saver = None
        self.num_saved_gifs = None

        self.print_template = "steps: {:d}, episodes: {:d}, mean 100 episode reward: "
        for i in range(len(self.active_task_indices)):
            self.print_template += "{:.1f}, "
        self.print_template += "% time spent exploring {:.0f}, time elapsed {:.2f}"

        self.reset()

    def reset(self):
        """
        Reset the solver.
        :return:        None.
        """

        # clear lists for rewards and abstract actions
        self.episode_rewards = [[] for _ in range(len(self.active_task_indices))]
        self.current_episode = 0
        self.actions_to_save = []

        # initialize new animator
        self.animator = None
        if self.animate:
            self.animator = vis_utils.Animator(0.2, max_value=self.max_depth_value)

        # initialize new gif saver
        self.gif_saver = None
        self.num_saved_gifs = 0
        if self.gif_save_path is not None:
            self.gif_saver = vis_utils.Saver(max_value=self.max_depth_value)

            # maybe create a directory for the gifs
            if not os.path.isdir(self.gif_save_path):
                os.makedirs(self.gif_save_path)

    def run(self):
        """
        Run the solver.
        :return:        None.
        """

        # initialize the environment
        observation = self.env.reset()
        current_episode_rewards = []
        goal_idx = np.random.choice(self.active_task_indices)

        timer_start = time.time()

        # Iterate over time steps
        for t in range(self.max_time_steps):

            # maybe visualize the depth image
            if self.animate and (self.animate_from is None or len(self.episode_rewards) >= self.animate_from):
                self.animator.next(observation[0][:, :, 0])

            if self.gif_saver is not None and \
                    (self.gif_save_limit is None or self.num_saved_gifs < self.gif_save_limit) and \
                    (self.gif_save_from is None or len(self.episode_rewards) >= self.gif_save_from):
                self.gif_saver.add(observation[0][:, :, 0])

            # maybe terminate due to episodes limit
            if self.max_episodes is not None and len(self.episode_rewards) > self.max_episodes:
                break

            # get action
            env_action, action, q_values = self.agent.act_v2(observation, t, self.active_task_indices.index(goal_idx))

            payload_1 = None
            if self.collect_pre is not None:
                payload_1 = self.collect_pre(self.env, action)

            # act
            observation = cp.deepcopy(observation)
            new_observation, rewards, dones, _ = self.env.step(env_action)

            payload_2 = None
            if self.collect_post is not None:
                payload_2 = self.collect_post(self.env)

            # remember
            new_observation = cp.deepcopy(new_observation)

            if self.remember:
                self.agent.remember(observation, action, rewards, new_observation, dones)

            if self.collect_data is not None:
                self.collect_data(observation, action, rewards, new_observation, dones, q_values, payload_1, payload_2)

            # learn
            if self.train and self.learning_start is not None and t > self.learning_start and t % self.train_freq == 0:
                self.agent.learn(t)

            # store episode rewards and handle the end of episode
            current_episode_rewards.append(rewards)
            if dones[goal_idx]:

                # maybe show the last observation
                if self.animate and (self.animate_from is None or self.current_episode >= self.animate_from):
                    self.animator.next(new_observation[0][:, :, 0])

                # maybe save the last observation
                if self.gif_saver is not None and \
                        (self.gif_save_limit is None or self.num_saved_gifs < self.gif_save_limit) and \
                        (self.gif_save_from is None or self.current_episode >= self.gif_save_from):

                    self.gif_saver.add(new_observation[0][:, :, 0])

                    if not self.gif_save_only_successful or rewards[goal_idx] == 10.0:
                        self.gif_saver.save_gif(os.path.join(self.gif_save_path, "episode_{:d}.gif".format(self.current_episode)))
                        self.num_saved_gifs += 1

                    self.gif_saver.reset()

                new_observation = self.env.reset()
                self.episode_rewards[self.active_task_indices.index(goal_idx)].append(
                    np.sum(np.array(current_episode_rewards)[:, goal_idx])
                )
                current_episode_rewards = []

                # print status
                if self.print_freq is not None and self.current_episode % self.print_freq == 0:

                    mean_100ep_reward = [
                        np.mean(self.episode_rewards[i][-100:]) for i in range(len(self.active_task_indices))
                    ]

                    timer_final = time.time()

                    exploration_value = self.agent.exploration_schedule.value(t) if self.agent.exploration_schedule is not None else 0

                    print(self.print_template.format(
                        t, self.current_episode, *mean_100ep_reward, 100 * exploration_value, timer_final - timer_start)
                    )

                    timer_start = timer_final

                    if self.early_stop_threshold is not None and np.mean(mean_100ep_reward) >= self.early_stop_threshold:
                        break

                goal_idx = np.random.choice(self.active_task_indices)
                self.current_episode += 1

            observation = new_observation

        # save learning curve
        if self.rewards_file is not None:
            for idx, task_idx in enumerate(self.active_task_indices):
                np.savetxt(self.rewards_file + "_task_{:d}.dat".format(task_idx), self.episode_rewards[idx])
