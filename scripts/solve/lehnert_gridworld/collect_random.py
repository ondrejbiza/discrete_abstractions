import argparse
import random
import pickle
from envs.lehnert_gridworld import LehnertGridworld
from abstraction.transition import Transition


def main(args):

    env = LehnertGridworld()
    transitions = []

    for i in range(args.num_exp):

        state = env.reset()
        bisim_state = env.get_bisim_state()
        action = random.choice(LehnertGridworld.ACTIONS)
        next_state, reward, _, _ = env.step(action)
        bisim_next_state = env.get_bisim_state()

        t = Transition(
            state, action, reward, next_state, False, False, state_block=bisim_state, next_state_block=bisim_next_state
        )
        transitions.append(t)

    with open(args.save_path, "wb") as file:
        pickle.dump(transitions, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect random data from Column World.")

    parser.add_argument("num_exp", type=int, help="number of transitions to save")
    parser.add_argument("save_path", help="where to save the transitions")

    parsed = parser.parse_args()
    main(parsed)
