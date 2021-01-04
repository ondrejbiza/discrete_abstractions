import argparse
import os
import matplotlib
from runners.bisim.lehnert_gridworld.LehnertGridworldExactPartition import LehnertGridworldExactPartitionRunner
from utils.logger import Logger
from utils.saver import Saver
import constants
import config_constants as cc

SAVE_PREFIX = "{:s}/lehnert_gridworld_exact_v1"
SAVE_TEMPLATE = "{:s}_round_to_{:d}"


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    matplotlib.use("pdf")

    saver = Saver(None)

    if args.save:
        dir_variables = [
            SAVE_PREFIX.format(args.base_dir), args.round_to
        ]
        dir_switches = [
            "hard_t"
        ]

        saver.create_dir_name(SAVE_TEMPLATE, dir_variables, dir_switches, args)
        saver.create_dir(add_run_subdir=True)

    saver.save_by_print(args, "settings")

    logger = Logger(save_file=saver.get_save_file("main", "log"), print_logs=True)
    logger.silence_tensorflow()

    model_config = {
        cc.HEIGHT: 30, cc.WIDTH: 3, cc.WEIGHT_DECAY: 0.0001, cc.OPTIMIZER: constants.OPT_ADAM,
        cc.LEARNING_RATE: 0.0005, cc.FC_ONLY: True, cc.DROPOUT_PROB: 0.0
    }

    runner_config = {
        cc.LOAD_MODEL_PATH: args.load_model_path, cc.ROUND_TO: args.round_to, cc.HARD_T: args.hard_t, cc.SAVER: saver,
        cc.LOGGER: logger
    }

    runner = LehnertGridworldExactPartitionRunner(runner_config, model_config)
    runner.setup()

    runner.main_training_loop()
    runner.evaluate_and_visualize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run exact partition iteration over a model.")

    parser.add_argument("load_model_path", help="model load path")

    parser.add_argument("--round-to", default=32, type=int, help="round model predictions to")
    parser.add_argument("--hard-t", default=False, action="store_true", help="argmax over next states probabilities")

    parser.add_argument("--save", default=False, action="store_true", help="save results")
    parser.add_argument("--base-dir", default="results", help="base save dir")

    parsed = parser.parse_args()
    main(parsed)
