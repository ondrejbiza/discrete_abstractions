import random, numpy, argparse
from minatar import Environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--replayoff", "-r", action="store_true")
    parser.add_argument("--targetoff", "-t", action="store_true")
    parser.add_argument("--ramp-difficulty", default=False, action="store_true")
    parser.add_argument("--sticky-actions", default=False, action="store_true")
    parser.add_argument("--save-dataset", default=False, action="store_true")
    parser.add_argument("--num-frames", type=int, default=5000000)
    args = parser.parse_args()

    env = Environment(
        args.game, sticky_action_prob=0.1 if args.sticky_actions else 0.0, difficulty_ramping=args.ramp_difficulty
    )

    num_episodes = 100
    num_actions = env.num_actions()

    reward_per_episode = []
    episode_rewards = []

    env.reset()

    for i in range(10000000):

        s = env.state()

        action = random.randrange(num_actions)
        reward, terminated = env.act(action)

        episode_rewards.append(reward)

        if terminated:
            reward_per_episode.append(numpy.sum(episode_rewards))
            episode_rewards = []

            if len(reward_per_episode) == num_episodes:
                break

            env.reset()

    print(numpy.mean(reward_per_episode))


if __name__ == '__main__':
    main()
