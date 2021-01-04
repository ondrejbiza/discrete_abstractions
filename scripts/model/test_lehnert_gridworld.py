import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from envs.lehnert_gridworld import LehnertGridworld
from model.LehnertGridworldModel import LehnertGridworldModel
import constants
import config_constants as cc


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    env = LehnertGridworld()

    model_config = {
        cc.HEIGHT: 30, cc.WIDTH: 3, cc.WEIGHT_DECAY: 0.0001, cc.OPTIMIZER: args.optimizer,
        cc.LEARNING_RATE: 0.0005, cc.FC_ONLY: not args.conv, cc.DROPOUT_PROB: args.dropout_prob
    }

    model = LehnertGridworldModel(model_config)
    model.build()
    model.start_session()
    model.load(args.load_model_path)

    state = env.reset()

    while True:

        plt.imshow(state)
        plt.show()

        action_str = input("action (w,a,s,d): ")

        if action_str == "w":
            action = LehnertGridworld.A_UP
        elif action_str == "s":
            action = LehnertGridworld.A_DOWN
        elif action_str == "a":
            action = LehnertGridworld.A_LEFT
        elif action_str == "d":
            action = LehnertGridworld.A_RIGHT

        reward, next_state = model.predict(np.array([state]), np.array([action]))
        reward = reward[0]
        next_state = next_state[0]

        next_state = next_state.reshape(30 * 3)
        pos = next_state.argmax()

        next_state = np.zeros(30 * 3, dtype=np.float32)
        next_state[pos] = 1.0
        next_state = next_state.reshape((30, 3))

        state = next_state

        print("action {:d}, reward {:.2f}".format(action, reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("load_model_path", help="model weights path")
    parser.add_argument("--optimizer", default=constants.OPT_MOMENTUM)
    parser.add_argument("--conv", default=False, action="store_true", help="use convolutions")
    parser.add_argument("--dropout-prob", type=float, default=0.0, help="dropout probability")

    parsed = parser.parse_args()
    main(parsed)
