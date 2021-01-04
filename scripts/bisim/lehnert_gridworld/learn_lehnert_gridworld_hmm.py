import argparse
import os
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D         # don't delete this, necessary for 3d projection
from utils.saver import Saver
from utils.logger import Logger
import constants
import config_constants as cc
from runners.bisim.lehnert_gridworld.LehnertGridworldHMM import LehnertGridworldHMMRunner

SAVE_PREFIX = "{:s}/lehnert_gridworld_model_hmm_v1"
SAVE_TEMPLATE = "{:s}_{:g}_elr_{:s}_opt_{:d}_steps_{:d}_blocks_{:d}_components_{:g}_beta1_{:g}_beta2{:s}"


def main(args):

    np.random.seed(2019)
    matplotlib.use("pdf")

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    saver = Saver(None)

    if args.save:
        dir_variables = [
            SAVE_PREFIX.format(args.base_dir), args.learning_rate, args.optimizer, args.num_steps, args.num_blocks,
            args.num_components, args.beta1, args.beta2,
            "" if args.data_limit is None else "_{:d}_dl".format(args.data_limit)
        ]
        dir_switches = [
            "oversample"
        ]

        saver.create_dir_name(SAVE_TEMPLATE, dir_variables, dir_switches, args)
        saver.create_dir(add_run_subdir=True)

    saver.save_by_print(args, "settings")

    logger = Logger(save_file=saver.get_save_file("main", "log"), print_logs=True)
    logger.silence_tensorflow()

    runner_config = {
        cc.LOAD_PATH: args.load_path, cc.SAVER: saver, cc.LOGGER: logger, cc.OVERSAMPLE: args.oversample,
        cc.VALIDATION_FRACTION: args.validation_fraction, cc.VALIDATION_FREQ: args.validation_freq,
        cc.BATCH_SIZE: args.batch_size, cc.LOAD_MODEL_PATH: args.load_model_path,
        cc.NUM_STEPS: args.num_steps, cc.DATA_LIMIT: args.data_limit
    }

    model_config = {
        cc.HEIGHT: args.height, cc.WIDTH: args.width, cc.WEIGHT_DECAY: args.weight_decay, cc.OPTIMIZER: args.optimizer,
        cc.LEARNING_RATE: args.learning_rate, cc.NUM_BLOCKS: args.num_blocks, cc.NUM_COMPONENTS: args.num_components,
        cc.BETA0: args.beta0, cc.BETA1: args.beta1, cc.BETA2: args.beta2,
        cc.MIXTURES_MU_INIT_SD: args.mixtures_mu_init_sd,  cc.MIXTURES_SD_INIT_MU: args.mixtures_sd_init_mu,
        cc.MIXTURES_SD_INIT_SD: args.mixtures_sd_init_sd, cc.MODEL_LEARNING_RATE: args.model_learning_rate,
        cc.ONE_Q_VALUE: False, cc.FC_ONLY: not args.conv
    }

    runner = LehnertGridworldHMMRunner(runner_config, model_config)

    runner.setup()
    runner.main_training_loop()

    if args.save_model:
        runner.save_model()

    runner.evaluate_and_visualize()

    if args.save_embeddings:
        runner.save_embeddings()

    runner.close_model_session()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Train a HMM in Column World.")

    parser.add_argument("load_path", help="dataset load path")
    parser.add_argument("height", type=int, help="world height")
    parser.add_argument("width", type=int, help="world width")

    parser.add_argument("--validation-fraction", type=float, default=0.2, help="validation fraction of the dataset")
    parser.add_argument("--validation-freq", type=int, default=1000, help="validation frequency")
    parser.add_argument("--oversample", default=False, action="store_true", help="oversample rewards>0")
    parser.add_argument("--data-limit", type=int, default=None, help="number of transitions to use")

    parser.add_argument("--num-blocks", type=int, default=32, help="continuous latent space size")
    parser.add_argument("--num-components", type=int, default=6, help="discrete latent space size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--model-learning-rate", type=float, default=0.01, help="prior learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--optimizer",
                        help=", ".join([constants.OPT_ADAM, constants.OPT_MOMENTUM, constants.OPT_SGD]),
                        default=constants.OPT_MOMENTUM)
    parser.add_argument("--num-steps", type=int, default=20000, help="number of training steps")
    parser.add_argument("--batch-size", type=int, default=100, help="batch size")
    parser.add_argument("--beta0", type=float, default=1.0, help="q loss weight")
    parser.add_argument("--beta1", type=float, default=0.0001, help="entropy loss weight")
    parser.add_argument("--beta2", type=float, default=0.0001, help="prior loss weight")
    parser.add_argument("--mixtures-mu-init-sd", type=float, default=0.1, help="mixtures mu init sd")
    parser.add_argument("--mixtures-sd-init-mu", type=float, default=-1.0, help="mixtures log(sd) init mu")
    parser.add_argument("--mixtures-sd-init-sd", type=float, default=0.1, help="mixtures log(sd) init sd")
    parser.add_argument("--conv", default=False, action="store_true", help="use convolutions")

    parser.add_argument("--save", default=False, action="store_true", help="save results")
    parser.add_argument("--base-dir", default="results", help="base directory")
    parser.add_argument("--save-model", default=False, action="store_true", help="save model")
    parser.add_argument("--load-model-path", help="load model path")
    parser.add_argument("--save-embeddings", default=False, action="store_true", help="save learned embeddings")
    parser.add_argument("--gpus", help="comma-separated list of gpus to use")

    parsed = parser.parse_args()
    main(parsed)
