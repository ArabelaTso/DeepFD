import os
from CodeBook.Utils.Utils import *
from CodeBook.Utils.Module import *
from keras.models import load_model
import argparse
import pickle
import random
import importlib
from CodeBook.Config import params
from CodeBook.SeedFaults.seedFaults import seed_single_fault, seed_multi_fault
from CodeBook.Config import FAULT_TYPE
import pandas as pd
from itertools import combinations

sys.path.append('./data')
sys.path.append('./Utils')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run programs.')
    parser.add_argument('--base', '-bs', default='./Evaluation', help='Base directory. Default: ./RQ2/')
    parser.add_argument('--dataset', '-ds', default='MNIST', help="Dataset.",
                        choices=["MNIST", "MNIST2", "CIFAR-10", "Blob", "Circle", "Reuters", "IMDB"])
    parser.add_argument('--seed_fault', '-sf', default=0, type=int,
                        help='seed fault or not. 0: run original model, >= 1: seed faults in this number.')
    parser.add_argument('--fault_type', '-ft', type=str, choices=FAULT_TYPE,
                        help="Fault types. choices=[{}]".format(",".join(FAULT_TYPE)))
    parser.add_argument('--faulty_val', '-fv', type=str, help="Value of fault. If not specify, assign randomly.")
    parser.add_argument('--max_fault', '-mf', default=20, type=int, help="Maximum number of faults.")

    args = parser.parse_args()

    parent_dir = args.base
    dataset = args.dataset
    base_dir = os.path.join(parent_dir, dataset)

    num_seeded_fault = args.seed_fault
    fault_type = args.fault_type if args.fault_type is not None else random.sample(FAULT_TYPE, 1)[0]
    fault_val = args.faulty_val
    max_fault = args.max_fault

    # check path validation
    if not os.path.exists(parent_dir) or not os.path.isdir(parent_dir):
        raise ValueError("{} is an invalid base directory.".format(parent_dir))
    if not os.path.exists(os.path.join(parent_dir, dataset)):
        raise ValueError("{} is an invalid path, please check.".format(base_dir))

    print("{}\nWalking Through Directory: {}".format("#" * 20, base_dir))

    model_cnt = 0
    total_seed_faults = 0

    for model in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, model)):
            print("\nModel {}: {}".format(model_cnt, model))
            model_path = os.path.join(base_dir, model)
            model_cnt += 1
        else:
            continue

        # load model and training configuration
        model_dir = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith("h5")][0]
        config_dir = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith("pkl")][0]

        model = load_model(model_dir)
        with open(config_dir, 'rb') as f:
            training_config = pickle.load(f)

        # list existing mutants
        cur_mutants = set([mutant for mutant in os.listdir(model_path)
                           if os.path.isdir(os.path.join(model_path, mutant)) and
                           mutant != "raw_data"])
        # if "origin" folder does not exist, construct one
        if "origin" not in cur_mutants:
            origin_dir = os.path.join(model_path, "origin")
            os.makedirs(origin_dir, exist_ok=True)
            with open(os.path.join(origin_dir, 'config.pkl'), 'wb') as f:
                pickle.dump(training_config, f)
            model.save(os.path.join(origin_dir, 'model.h5'))
        else:
            cur_mutants.remove("origin")

        # ########################################################################################
        # Necessary check before seeding fault(s)
        if num_seeded_fault == 0:
            print("No need to seed fault. Continue.")
            continue
        # if num_seeded_fault > 0, need to seed faults.
        gen_mutant_list = []

        # Count how many faults already exist.
        existing_fault_cnt = list(filter(lambda x: len(x.split("__")) == num_seeded_fault, cur_mutants))

        # check whether there are enough faults
        # if already enough, no need to seed faults
        # else, seed more faults
        if len(existing_fault_cnt) >= max_fault:
            print("Already enough faults ({}), skip seeding.".format(max_fault))
            continue
        else:
            print("Not enough faults ({}), continue seeding.".format(len(existing_fault_cnt)))

        total_combination_each_model = 0

        # ########################################################################################
        # seed fault(s)
        if num_seeded_fault == 1:
            # if seed 1 fault, set an upper bound.
            model, training_config, mutant_str = seed_single_fault(model, training_config, fault_type, fault_val)
            gen_mutant_list.append(mutant_str)
        # seed at least two faults
        # rely on the stats_faulty.csv under model directory
        elif num_seeded_fault > 1:
            stats_file_path = os.path.join(model_path, "stats_faulty.csv")
            if not os.path.exists(stats_file_path):
                print("Cannot find stats_faulty.csv under the folder {}. Skip.".format(model_path))
                continue
            content = pd.read_csv(stats_file_path)

            # for debug only
            # print("read {} in shape {}".format(stats_file_path, content.shape))
            # print(content)

            # construct fault_dict
            fault_dict = content[content["is_faulty"].astype('float32') == 1.].loc[:,
                         FAULT_TYPE[0]:FAULT_TYPE[-1]].to_dict("list")

            fault_dict = {k: [v for v in filter(lambda x: x == x, vlist)] for k, vlist in fault_dict.items()}
            fault_dict = {k: v for k, v in fault_dict.items() if len(v) > 0}

            # for debug only
            # print("fault_dict", fault_dict)

            # get fault combinations
            fault_dict_list = []

            if len(fault_dict.keys()) < num_seeded_fault:
                print("No combination can be found. Skip seeding.")
                continue
            else:
                fault_type_combination = [c for c in combinations(fault_dict.keys(), num_seeded_fault)]
                for keys in fault_type_combination:
                    cur_dict = {}
                    for key in keys:
                        for v in fault_dict[key]:
                            cur_dict[key] = v
                        fault_dict_list.append(cur_dict)
                        # for v1 in fault_dict[k1]:
                        #     for v2 in fault_dict[k2]:
                        #         fault_dict_list.append({k1: v1, k2: v2})

            print("Fault combination: {}".format(len(fault_dict_list)))
            total_combination_each_model += len(fault_dict_list)

            # sample faults
            num_sample_fault = min(max_fault - len(existing_fault_cnt), len(fault_dict_list))
            print("Sample {} faults.".format(num_sample_fault))

            # seed faults
            for sclt_dict in random.sample(fault_dict_list, num_sample_fault):
                # seed multiple fault
                model, training_config, mutant_str = seed_multi_fault(model, training_config, fault_dict,
                                                                      num=num_seeded_fault)
                gen_mutant_list.append(mutant_str)

            print(
                "Total {}-fault combinations for this model: {}".format(num_seeded_fault, total_combination_each_model))

        # create folder with suffix
        for mutant_str in gen_mutant_list:
            fault_dir = os.path.join(model_path, mutant_str)
            if not os.path.exists(fault_dir):
                os.mkdir(fault_dir)
                with open(os.path.join(fault_dir, 'config.pkl'), 'wb') as f:
                    pickle.dump(training_config, f)
                model.save(os.path.join(fault_dir, 'model.h5'))

                print("mkdir {}".format(fault_dir))

        total_seed_faults += len(gen_mutant_list)

    print("Dataset: {} | Model: {} | Newly Seeded Faults: {}".format(dataset, model_cnt, total_seed_faults))
