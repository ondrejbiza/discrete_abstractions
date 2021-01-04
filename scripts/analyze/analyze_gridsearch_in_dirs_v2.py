import argparse, os
import numpy as np


def main(args):

    # load all files and sum rewards
    dir_names = os.listdir(args.dir)
    dir_paths = [os.path.join(args.dir, file) for file in dir_names]

    files = []
    names = []

    for dir_path, dir_name in zip(dir_paths, dir_names):

        runs = os.listdir(dir_path)

        for run in runs:
            run_dir = os.path.join(dir_path, run)
            dir_file = os.path.join(run_dir, args.file_name)

            if os.path.isfile(dir_file):

                file = np.loadtxt(dir_file)

                if args.indices is not None:
                    file = file[args.indices]

                files.append(file)
                names.append(dir_name)

    if args.mean:
        sums = [np.mean(file) for file in files]
    else:
        sums = [np.sum(file) for file in files]

    last_n = []

    for file in files:
        if len(file.shape) == 0:
            last_n.append(file)
        else:
            if args.mean:
                last_n.append(np.mean(file[-args.last_n:]))
            else:
                last_n.append(np.sum(file[-args.last_n:]))

    # average reward sums over runs with the same settings
    settings = {}

    for idx, file_name in enumerate(names):

        prefix = file_name.split("run")[0]

        if not prefix in settings:
            settings[prefix] = []

        settings[prefix].append((sums[idx], last_n[idx]))

    settings = {key: (np.mean([v[0] for v in value]), np.std([v[0] for v in value]), np.mean([v[1] for v in value]),
                      np.std([v[1] for v in value])) for key, value in settings.items()}

    # find the best setting
    max_key = None
    max_value = None
    max_last_n = None

    for key, value in settings.items():

        to_consider = value[0]

        if max_value is None or (to_consider < max_value and args.min) or (to_consider > max_value and not args.min):
            max_key = key
            max_value = to_consider
            max_last_n = value[2]

    print()
    print("best setting:")
    print("{}\t{:.2f} (last 100 episode: {:.2f})".format(max_key, max_value, max_last_n))
    print()

    # maybe print average sums for all hyper-parameter settings
    if args.print_all:
        print("average sum of rewards for all settings:")
        for key in sorted(settings.keys()):
            value = settings[key]
            print("{}\t\t{:.2f} +- {:.2f} (last 100 episode: {:.2f} +- {:.2f})".format(key, *value))
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Analyze results of a gridsearch.")

    parser.add_argument("dir", help="directory containing directories containing directories for each run")
    parser.add_argument("file_name", help="files to load")

    parser.add_argument("-a", "--print-all", default=False, action="store_true",
                        help="print the average sum of rewards for all settings")
    parser.add_argument("-m", "--mean", default=False, action="store_true",
                        help="consider mean reward instead of the sum of rewards; the sum of rewards favors more "
                             "episodes, which in turn favors the symbolic agent")
    parser.add_argument("--last-n", type=int, default=100,
                        help="number of episodes to average from the end (the value is given in parentheses)")
    parser.add_argument("-i", "--indices", type=int, nargs="+", default=None, help="which line of the file to take")
    parser.add_argument("--min", default=False, action="store_true")

    parsed = parser.parse_args()
    main(parsed)
