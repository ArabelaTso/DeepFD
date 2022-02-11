import os
import argparse
import pickle
import pandas as pd
from keras.models import load_model
from SeedFaults.seedFaults import seed_single_fault, seed_multi_fault
from Config import FAULT_TYPE
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seed faults')
    parser.add_argument('--base_dir', '-bd',
                        default='./Program/MNIST/config_random_1ed98a53-6d6f-466d-81ba-ffd8a252f5b8',
                        help='base path to a model')
    parser.add_argument('--seed_fault', '-sf', default=1, type=int,
                        help='seed fault or not. 0: run original model, >= 1: seed faults')
    parser.add_argument('--fault_type', '-ft', default='lr', type=str, choices=['opt', 'lr', 'act', 'loss', 'epoch'],
                        help="Fault types. choices=['opt', 'lr', 'act','loss', 'epoch']")
    parser.add_argument('--faulty_val', '-fv', type=str, help="Value of fault. If not specify, assign randomly.")
    parser.add_argument('--max_fault', '-mf', default=20, type=int, help="Maximum number of faults.")
    args = parser.parse_args()

    # ########################################################
    # basic preparation
    base_dir = args.base_dir
    # config_path = args.config_path
    # check_interval = 'batch_' + str(args.check_interval)  # or "epoch_"
    # overwrite = args.overwrite
    seed_fault = args.seed_fault
    fault_type = args.fault_type if seed_fault == 1 else "UNK"
    fault_val = None
    max_fault = args.max_fault

    print("Process", base_dir)

    model_dir = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith("h5")][0]
    config_dir = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith("pkl")][0]

    # load model and training configuration
    model = load_model(model_dir)
    with open(config_dir, 'rb') as f:
        training_config = pickle.load(f)

    folder_names = [f for f in os.listdir(base_dir) if
                    (not f.endswith("h5")) and
                    (not f.endswith('pkl')) and
                    f != "raw_data"]

    # mkdir "origin" if it is not exists.
    if "origin" not in folder_names:
        origin_dir = os.path.join(base_dir, "origin")
        os.mkdir(origin_dir)
        with open(os.path.join(origin_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(training_config, f)
        model.save(os.path.join(origin_dir, 'model.h5'))

    suffix = ""
    # seed one fault if not seeded
    if seed_fault == 1:
        # if seed 1 fault, set an upper bound.
        if len(folder_names) - 1 > max_fault:
            print("Already enough faults ({}), skip seeding.".format(max_fault))
            exit(0)
        else:
            print("Not enough faults ({}), continue seeding.".format(len(folder_names) - 1))

        model, training_config, suffix = seed_single_fault(model, training_config, fault_type, fault_val)

    # seed at least two faults
    # rely on the stats_faulty.csv
    elif seed_fault > 1:
        stats_file = os.path.join(base_dir, "stats_faulty.csv")
        if not os.path.exists(stats_file):
            print("Cannot find stats_faulty.csv under the folder {}. Skip.".format(base_dir))
            pass
        content = pd.read_csv(stats_file)
        print("read {} in shape {}".format(stats_file, content.shape))
        # construct fault_dict
        print(content)
        fault_dict = content[content["is_faulty"] == '1'].loc[:, FAULT_TYPE[0]:FAULT_TYPE[-1]].to_dict("list")
        print("fault_dict", fault_dict)

        fault_dict = {k: [v for v in filter(lambda x: len(x), vlist)] for k, vlist in fault_dict.items()}

        print("fault_dict", fault_dict)
        # seed multiple fault
        model, training_config, suffix = seed_multi_fault(model, training_config, fault_dict, num=seed_fault)

    print("Seed Fault: ", suffix)

    # create folder with suffix
    if suffix not in folder_names:
        fault_dir = os.path.join(base_dir, suffix)
        os.mkdir(fault_dir)
        with open(os.path.join(fault_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(training_config, f)
        model.save(os.path.join(fault_dir, 'model.h5'))
