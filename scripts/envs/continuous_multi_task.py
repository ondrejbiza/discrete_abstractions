import argparse
from envs.continuous_multi_task import ContinuousMultiTask, CheckGoalFactory

BASE_PATH = "."


def main(args):

    fcs = [
        CheckGoalFactory.check_goal_puck_stacking(2), CheckGoalFactory.check_goal_puck_stacking(3),
        CheckGoalFactory.check_goal_row(2), CheckGoalFactory.check_goal_row(3),
        CheckGoalFactory.check_goal_two_stacks(2), CheckGoalFactory.check_goal_stairs(3),
        CheckGoalFactory.check_goal_row_diag(2), CheckGoalFactory.check_goal_row_diag(3)
    ]

    env = ContinuousMultiTask(
        args.size, args.num_pucks, fcs, args.num_actions, height_noise_std=args.height_noise_std
    )

    while True:

        x, y = env.show_state(draw_grid=args.draw_grid)
        action = int(x) * args.num_actions + int(y)

        if env.hand is not None:
            action += args.num_actions ** 2

        s, r, d, _ = env.step(action)

        print("rewards:", r)
        print("dones:", d)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--num-actions", type=int, default=112)
    parser.add_argument("--num-pucks", type=int, default=3)
    parser.add_argument("--goal", type=int, default=3)

    parser.add_argument("--height-noise-std", type=float, default=0.0)
    parser.add_argument("--save", default=False, action="store_true")
    parser.add_argument("--draw-grid", default=False, action="store_true")
    parser.add_argument("--depth-size", type=int, default=None)

    parsed = parser.parse_args()
    main(parsed)
