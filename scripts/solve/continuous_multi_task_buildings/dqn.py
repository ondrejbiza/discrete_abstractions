import argparse
import os
import pickle
import numpy as np
from agents.dqn_multi_task import DQNMultiTask
from agents.solver_2_multi_task import SolverMultiTask
from abstraction.transition import Transition
import constants
from envs.continuous_multi_task_buildings import ContinuousMultiTask, CheckGoalFactory

LEARNING_STARTS = 60
DEFAULT_BUFFER_SIZE = 10000
TRAIN_FREQ = 1
PRINT_FREQ = 1


def collect_data_factory(transitions, limit):

    def collect_data(observation, action, reward, new_observation, done, q_values, payload_1, payload_2):

        t = Transition(
            observation, action, reward, new_observation, False, done
        )

        if len(transitions) >= limit:
            del transitions[0]

        transitions.append(t)

    return collect_data


def set_q_values_for_transitions(transitions, agent, batch_size):

    depths_batch = []
    hand_states_batch = []
    next_depths_batch = []
    next_hand_states_batch = []
    q_values = []
    next_q_values = []

    for idx in range(len(transitions)):

        transition = transitions[idx]

        depths_batch.append(transition.state[0])
        hand_states_batch.append(transition.state[1])

        next_depths_batch.append(transition.next_state[0])
        next_hand_states_batch.append(transition.next_state[1])

        if len(depths_batch) >= batch_size or idx == len(transitions) - 1:
            # full batch or last transition
            depths_batch = np.stack(depths_batch, axis=0)
            hand_states_batch = np.stack(hand_states_batch, axis=0)

            next_depths_batch = np.stack(next_depths_batch, axis=0)
            next_hand_states_batch = np.stack(next_hand_states_batch, axis=0)

            tmp_q_values = agent.session.run(agent.get_q, feed_dict={
                agent.depth_pl: depths_batch,
                agent.hand_states_pl: hand_states_batch
            })

            tmp_next_q_values = agent.session.run(agent.get_q, feed_dict={
                agent.depth_pl: next_depths_batch,
                agent.hand_states_pl: next_hand_states_batch
            })

            q_values.append(tmp_q_values)
            next_q_values.append(tmp_next_q_values)

            depths_batch = []
            hand_states_batch = []

            next_depths_batch = []
            next_hand_states_batch = []

    q_values = np.concatenate(q_values, axis=0)
    next_q_values = np.concatenate(next_q_values, axis=0)

    for tmp_q_values, tmp_next_q_values, transition in zip(q_values, next_q_values, transitions):

        transition.q_values = tmp_q_values
        transition.next_q_values = tmp_next_q_values


def main(args):

    assert args.active_indices is not None

    # gpus
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # setup environment
    fcs = [
        CheckGoalFactory.check_goal_puck_stacking(2),
        CheckGoalFactory.check_goal_puck_stacking(3),
        CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_TOP, 2),
        CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_BOTTOM, 2),
        CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_LEFT, 2),
        CheckGoalFactory.check_b1_side(CheckGoalFactory.SIDE_RIGHT, 2)
    ]

    env = ContinuousMultiTask(
        args.num_blocks, args.num_pucks, args.num_roofs, fcs, args.num_actions, height_noise_std=args.height_noise_std,
        action_failure_prob=args.action_failure_prob
    )
    env.initStride = args.init_env_stride  # stride for initial puck placement
    env.stride = args.env_stride  # stride for action specification

    # setup the agent
    state_shape = (args.num_blocks * 28, args.num_blocks * 28, 1)
    output_shape = (args.num_actions, args.num_actions)

    agent = DQNMultiTask(
        env, state_shape, output_shape, args.num_filters, args.filter_sizes, args.strides, args.hiddens,
        args.learning_rate, args.batch_size, constants.OPT_MOMENTUM, args.exploration_fraction, 1.0,
        args.final_epsilon, args.max_time_steps, args.active_indices, buffer_size=args.buffer_size,
        prioritized_replay=not args.disable_prioritized_replay, target_net=not args.disable_target_network,
        target_update_freq=args.target_update_freq, target_size=args.target_size
    )

    agent.start_session(args.num_cpu, args.gpu_memory_fraction)

    # maybe load weights
    if args.load_weights:
        agent.load(args.load_weights)
        print("Loaded weights.")

    # initialize a solver
    transitions = []
    collect_data = collect_data_factory(transitions, args.save_exp_num)

    t_collect_data = None
    if args.save_exp_path is not None and not args.save_exp_after_training:
        t_collect_data = collect_data

    solver = SolverMultiTask(
        env, agent, args.max_time_steps, learning_start=LEARNING_STARTS, train_freq=TRAIN_FREQ,
        active_task_indices=args.active_indices, max_episodes=args.max_episodes, rewards_file=args.rewards_file,
        animate=args.animate, animate_from=args.animate_from,
        gif_save_path=args.save_gifs_path, gif_save_limit=args.save_limit,
        gif_save_only_successful=args.save_only_successful, max_depth_value=args.num_pucks,
        collect_data=t_collect_data
    )

    # solve the environment
    solver.run()

    # save the weights of the network
    if args.save_weights is not None:
        agent.save(args.save_weights)

    # maybe run trained DQN
    if args.save_exp_path is not None and args.save_exp_after_training:

        agent.exploration_fraction = 1.0
        agent.init_explore = args.save_exp_eps
        agent.final_explore = args.save_exp_eps
        agent.setup_exploration_()

        solver = SolverMultiTask(
            env, agent, args.save_exp_num, learning_start=args.save_exp_num, train_freq=TRAIN_FREQ, train=False,
            max_episodes=args.save_exp_num * 100, collect_data=collect_data, active_task_indices=args.active_indices
        )
        solver.run()

    # maybe save the collected experience
    if args.save_exp_path is not None:

        if args.save_q_values:
            set_q_values_for_transitions(transitions, agent, args.batch_size)

        save_dir = os.path.dirname(args.save_exp_path)
        if len(save_dir) > 0 and not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        with open(args.save_exp_path, "wb") as file:
            pickle.dump(transitions, file)

    # stop session
    agent.stop_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train a DQN on multiple tasks in the buildings world.")

    parser.add_argument("num_blocks", type=int, default=4, help="size of the environment")
    parser.add_argument("num_actions", type=int, default=4, help="number of actions")
    parser.add_argument("num_pucks", type=int, default=3, help="number of pucks in the environment")
    parser.add_argument("num_roofs", type=int, default=1, help="number of roofs in the environment")
    parser.add_argument("--active-indices", default=None, type=int, nargs="+", help="which tasks to train on")
    parser.add_argument("--height-noise-std", type=float, default=0.0, help="height noise")
    parser.add_argument("--action-failure-prob", type=float, default=0.0, help="probability that action fails")

    parser.add_argument("--num-filters", type=int, nargs="+", default=[16, 32], help="number of filters in the DQN")
    parser.add_argument("--filter-sizes", type=int, nargs="+", default=[3, 3], help="filter sizes")
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 2], help="conv layers strides")
    parser.add_argument("--hiddens", type=int, nargs="+", default=[64], help="hidden fully-connected layers after conv")
    parser.add_argument("--target-size", type=int, default=None, help="resize input image to this size")

    parser.add_argument("--learning-rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--target-update-freq", type=int, default=100, help="update frequency of the target network")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
                        help="how many time steps to explore expressed as the faction of the maximum number of "
                             "time steps")
    parser.add_argument("--final-epsilon", type=float, default=0.1,
                        help="value of epsilon after the end of exploration")
    parser.add_argument("--disable-prioritized-replay", default=False, action="store_true",
                        help="disable prioritized replay")
    parser.add_argument("--disable-target-network", default=False, action="store_true", help="disable target DQN")
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="buffer size")

    parser.add_argument("--init-env-stride", type=int, default=28,
                        help="stride for the placement of objects in the environment")
    parser.add_argument("--env-stride", type=int, default=28, help="stride for the actions")
    parser.add_argument("--max-time-steps", type=int, default=40000, help="maximum number of time steps to run for")
    parser.add_argument("--max-episodes", type=int, default=2000, help="maximum number of episodes to run for")
    parser.add_argument("--rewards-file", default=None, help="where to save the per-episode rewards")

    parser.add_argument("--animate", default=False, action="store_true", help="show an animation of the environment")
    parser.add_argument("--animate-from", type=int, default=0, help="from which episode to start the animation")

    parser.add_argument("--save-gifs-path", help="save path for gifs of episodes")
    parser.add_argument("--save-only-successful", default=False, action="store_true",
                        help="save only the successful episodes")
    parser.add_argument("--save-limit", type=int, help="maximum number of episodes to save")

    parser.add_argument("--gpus", help="comma-separated list of GPUs to use")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.1,
                        help="a fraction of GPU memory to use; None for all")
    parser.add_argument("--num-cpu", type=int, default=None, help="number of CPUs to use")

    parser.add_argument("--save-exp-path", help="where to save the collected experience")
    parser.add_argument("--save-exp-num", type=int, default=20000, help="number of transitions to save")
    parser.add_argument("--save-q-values", default=False, action="store_true", help="save q-values for each transition")
    parser.add_argument("--save-exp-after-training", default=False, action="store_true",
                        help="collect dataset after training")
    parser.add_argument("--save-exp-eps", type=float, default=0.1, help="dataset collection exploration factor")

    parser.add_argument("--save-weights", help="where to save the weights of the network")
    parser.add_argument("--load-weights", help="load weights of the network")

    parsed = parser.parse_args()
    main(parsed)
