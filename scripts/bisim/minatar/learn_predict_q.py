import argparse
import os
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D         # don't delete this, necessary for 3d projection
import constants
from utils.logger import LearnExpectationContinuousLogger
from utils.saver import Saver
from runners.bisim.minatar.Q import QRunnerMinAtar

SAVE_PREFIX = "results/minatar/q_predictor_bisim_v1"
SAVE_TEMPLATE = "{:s}_{:s}_{:d}_blocks_{:g}_elr_{:s}_opt_{:d}_steps"


def main(args):

    np.random.seed(2019)

    if not args.show_graphs:
        matplotlib.use("pdf")

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    saver = Saver(None)

    if args.save:
        dir_variables = [
            SAVE_PREFIX, args.game, args.num_blocks, args.encoder_learning_rate, args.encoder_optimizer, args.num_steps
        ]
        dir_switches = [
            "no_sample", "only_one_q_value" "disable_batch_norm"
        ]

        saver.create_dir_name(SAVE_TEMPLATE, dir_variables, dir_switches, args)
        saver.create_dir(add_run_subdir=True)

    saver.save_by_print(args, "settings")

    logger = LearnExpectationContinuousLogger(save_file=saver.get_save_file("main", "log"), print_logs=True)
    logger.silence_tensorflow()

    if args.game == constants.GAME_BREAKOUT:
        num_actions = 6
    else:
        raise ValueError("Unknown game.")

    runner = QRunnerMinAtar(
        args.load_path, num_actions, logger, saver, args.num_blocks,
        args.encoder_learning_rate, args.weight_decay, args.encoder_optimizer, args.num_steps,
        disable_batch_norm=args.disable_batch_norm, disable_softplus=args.disable_softplus,
        no_sample=args.no_sample, only_one_q_value=args.only_one_q_value,
        validation_freq=args.validation_freq, validation_fraction=args.validation_fraction, summaries=args.summaries,
        load_model_path=args.load_model_path, zero_sd_after_training=args.zero_sd_after_training
    )

    runner.setup()
    runner.main_training_loop()

    if args.save_model:
        runner.save_model()

    runner.evaluate_and_visualize()
    runner.close_model_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Learn to predict q-values in MinAtar games.")

    parser.add_argument("load_path", help="dataset load path")
    parser.add_argument("game", help="game")

    parser.add_argument("--num-blocks", type=int, default=32, help="continuous latent space size")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="validation fraction of the dataset")
    parser.add_argument("--validation-freq", type=int, default=1000, help="validation frequency")
    parser.add_argument("--oversample", default=False, action="store_true", help="oversample rewards>0")
    parser.add_argument("--disable-softplus", default=False, action="store_true",
                        help="do not use softplus on samples sds")
    parser.add_argument("--no-sample", default=False, action="store_true",
                        help="do not sample from encoder Gaussian")
    parser.add_argument("--only-one-q-value", default=False, action="store_true", help="use only one q-value")
    parser.add_argument("--zero-sd-after-training", default=False, action="store_true",
                        help="zero encoder sd after training")

    parser.add_argument("--encoder-learning-rate", type=float, default=0.001, help="encoder learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="encoder weight decay")
    parser.add_argument("--encoder-optimizer",
                        help=", ".join([constants.OPT_ADAM, constants.OPT_MOMENTUM, constants.OPT_SGD]),
                        default=constants.OPT_ADAM)
    parser.add_argument("--num-steps", type=int, default=50000, help="number of training steps")
    parser.add_argument("--disable-batch-norm", default=False, action="store_true", help="disable batch norm")

    parser.add_argument("--save", default=False, action="store_true", help="save results")
    parser.add_argument("--show-graphs", default=False, action="store_true", help="show graphs")
    parser.add_argument("--summaries", default=False, action="store_true", help="save tf summaries")
    parser.add_argument("--save-model", default=False, action="store_true", help="save model weights")
    parser.add_argument("--load-model-path", help="load model path")
    parser.add_argument("--gpus", help="comma-separated list of gpus to use")

    parsed = parser.parse_args()
    main(parsed)
