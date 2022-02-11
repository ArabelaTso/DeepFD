import os
import sys
import shutil
from pathlib import Path

from CodeBook.SeedFaults.seedFaults import *
from collections import defaultdict
from CodeBook.Utils.Utils import *
from CodeBook.Utils.Module import *
import keras
import keras.optimizers as O
from keras.models import load_model
import argparse
import pickle
import importlib
from CodeBook.Config import params
from multiprocessing import Process
from multiprocessing import Pool
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.append('./data')
sys.path.append('./Utils')

sys.setrecursionlimit(100000)


def get_dataset(dataset_name, base_dir=""):
    data_set = {}

    if dataset_name == "customized":
        with open(os.path.join(base_dir, "dataset.pkl"), 'rb') as f:
            data_set = pickle.load(f)
        return data_set

    if dataset_name == "IRIS":
        iris = load_iris()
        data_set['x'], data_set['y'], data_set['x_val'], data_set['y_val'] = train_test_split(iris['data'], ['target'],
                                                                                              test_size=0.2)
        return data_set

    data_name = dataset_name.split('_')[0]
    data = importlib.import_module('{}'.format(data_name.lower()), package='data')

    if data_name == 'simplednn':
        choice = dataset_name.split('_')[-1]
        (x, y), (x_val, y_val) = data.load_data(method=choice)
    else:
        (x, y), (x_val, y_val) = data.load_data()
    preprocess_func = data.preprocess
    data_set['x'] = preprocess_func(x)
    data_set['x_val'] = preprocess_func(x_val)
    if dataset_name == 'cifar10' or dataset_name == 'mnist':
        labels = 10
        data_set['y'] = keras.utils.to_categorical(y, labels)
        data_set['y_val'] = keras.utils.to_categorical(y_val, labels)
    elif data_name == 'reuters':
        labels = 46
        data_set['y'] = keras.utils.to_categorical(y, labels)
        data_set['y_val'] = keras.utils.to_categorical(y_val, labels)
    elif data_name == 'imdb':
        labels = 2
        data_set['y'] = keras.utils.to_categorical(y, labels)
        data_set['y_val'] = keras.utils.to_categorical(y_val, labels)
    else:
        data_set['y'] = y
        data_set['y_val'] = y_val
    return data_set


def gpu_setup(enable=False, core_i=0, core_j=1):
    if enable:
        # tf.debugging.set_log_device_placement(True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("GPU total:", len(gpus))
        print("Run on GPU{} - {}".format(core_i, core_j))
        tf.config.experimental.set_visible_devices(gpus[core_i:core_j], 'GPU')
        #    if core_j <= len(gpus) and core_i < core_j:
        #        tf.config.experimental.set_visible_devices(gpus[core_i:core_j], 'GPU')
        #    else:
        #        tf.config.experimental.set_visible_devices(gpus[1:3], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def parse_train_config(config):
    opt_cls = getattr(O, config['optimizer'])
    opt = opt_cls(**config['opt_kwargs'])
    batch_size = config['batchsize']
    # add upper bound to batch size due to OOM
    if training_config['dataset'] == "reuters" or training_config['dataset'] == "imdb":
        batch_size = min(16, batch_size)
    epoch = config['epoch']
    loss = config['loss']
    callbacks = [] if 'callbacks' not in config.keys() else config['callbacks']

    return opt, batch_size, epoch, loss, callbacks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run programs.')
    parser.add_argument('--base', '-bs', default='./Evaluation', help='Base directory. Default: ./RQ2/')
    parser.add_argument('--dataset', '-ds', default='MNIST', help="Dataset.",
                        # choices=["MNIST", "MNIST2", "CIFAR-10", "Blob", "Circle", "Reuters", "IMDB"]
                        )
    parser.add_argument('--check_interval', '-ckit', default=256, type=int, help='detection interval')
    parser.add_argument('--gpu', '-g', default=1, choices=[0, 1], type=int, help="Enable GPU or not.")
    parser.add_argument('--core_i', '-ci', default=1, type=int, help="Run on which gpu core (i).")
    parser.add_argument('--core_j', '-cj', default=3, type=int, help="Run on which gpu core (j).")
    parser.add_argument('--core', '-c', default=0, type=int, help="Run on which gpu core.")
    parser.add_argument('--run', '-r', default=0, type=int, choices=[0, 1], help="Run or not.")
    parser.add_argument('--max_iter', '-iter', default=20, type=int, help="Times that one model repeats.")

    args = parser.parse_args()

    parent_dir = args.base
    dataset_name = args.dataset
    base_dir = os.path.join(parent_dir, dataset_name)
    check_interval = 'batch_' + str(args.check_interval)
    max_iter = max(1, args.max_iter)  # run at least once
    gpu_setup(args.gpu, args.core_i, args.core_j)

    # check path validation
    if not os.path.exists(parent_dir) or not os.path.isdir(parent_dir):
        raise ValueError("{} is an invalid base directory.".format(parent_dir))
    if not os.path.exists(os.path.join(parent_dir, dataset_name)):
        raise ValueError("{} is an invalid path, please check.".format(base_dir))

    print("{}\nWalking Through Directory: {}".format("#" * 20, base_dir))

    model_cnt = 0
    mutant_cnt = 0
    model_finished = 0
    mutant_need_train = 0
    iter_need_train = 0

    is_dataset_loaded = False

    start_time = time.time()
    time_list = []

    for model in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, model)):
            print("\n\nodel {}: {}".format(model_cnt, model))
            model_cnt += 1
        else:
            continue

        has_mutant_need_train = False
        for mutant in os.listdir(os.path.join(base_dir, model)):
            if os.path.isdir(os.path.join(base_dir, model, mutant)) and mutant != "raw_data":
                print("\nMutant {}: {}".format(mutant_cnt, mutant))
                mutant_dir = os.path.join(base_dir, model, mutant)
                mutant_cnt += 1
            else:
                continue

            iter_finish = 0
            # iters = [f for f in os.listdir(os.path.join(base_dir, model, mutant)) if f.startswith("iter_")]
            iter_remain = []
            for iter_index in range(1, max_iter + 1, 1):
                iter = "iter_{}".format(iter_index)
                if os.path.exists(os.path.join(base_dir, model, mutant, iter)):
                    if os.path.exists(os.path.join(base_dir, model, mutant, iter, "log_dir", "log.csv")):
                        iter_finish += 1
                    else:
                        shutil.rmtree(os.path.join(base_dir, model, mutant, iter))
                        print("No log.csv found! Remove the director ({})!".format(
                            os.path.join(base_dir, model, mutant, iter)))
                        iter_remain.append(iter)
                else:
                    iter_remain.append(iter)

            # print("iters", iters)
            print("Iter finished", iter_finish)
            print("Iter remained", iter_remain)

            if iter_finish >= max_iter:
                print("Pass!")
                continue
            else:
                print("!!! Need more training!"
                      "Current iter: {} | remain: {} | Mutant: {}".format(iter_finish,
                                                                          max_iter - iter_finish,
                                                                          os.path.join(base_dir, model, mutant)))
                mutant_need_train += 1
                iter_need_train += (max_iter - iter_finish)
                has_mutant_need_train = True

            if args.run:
                # load model and training configuration
                try:
                    model_dir = os.path.join(mutant_dir, "model.h5")
                    config_dir = os.path.join(mutant_dir, "config.pkl")
                except FileNotFoundError as fe:
                    print(fe)
                    continue

                print("Loading model and config...")
                print("model_dir", model_dir)
                print("config_dir", config_dir)
                model_structure = load_model(model_dir)

                with open(config_dir, 'rb') as f:
                    training_config = pickle.load(f)
                opt, batch_size, epoch, loss, callbacks = parse_train_config(training_config)
                print("opt: {} | batch: {} | epoch: {} | loss: {} | callbacks: {}".format(opt, batch_size, epoch, loss,
                                                                                          callbacks))

                # add earlystop to callbacks
                callbacks.append("estop")
                print("Model and configuration loaded.")

                # load dataset
                if is_dataset_loaded:
                    print("Dataset already {} loaded.".format(dataset_name))
                else:
                    dataset = get_dataset(training_config['dataset'])
                    is_dataset_loaded = True
                    print("Dataset {} loaded.".format(dataset_name))

                # whether these iters have failed run.
                iter_fail = False

                for remain_iter in iter_remain:
                    print("\n- Current Iteration {}".format(remain_iter))
                    iter_dir = os.path.join(base_dir, model, mutant, remain_iter)
                    if not os.path.exists(iter_dir):
                        os.makedirs(iter_dir, exist_ok=True)

                    # check
                    assert os.path.exists(iter_dir)

                    # debug only
                    # print("model_dir", model_dir)
                    # print("config_dir", config_dir)

                    save_dir = os.path.join(iter_dir, "result_dir")
                    log_dir = os.path.join(iter_dir, "log_dir")
                    new_issue_dir = os.path.join(iter_dir, "new_issue")

                    # ########################################################
                    # start training
                    train_start_time = time.time()
                    try:
                        print("Start training...")
                        train_result, _, _ = model_train(model=model_structure, train_config_set=training_config,
                                                         optimizer=opt,
                                                         loss=loss,
                                                         dataset=dataset, iters=epoch, batch_size=batch_size,
                                                         callbacks=callbacks, verb=0, checktype=check_interval,
                                                         autorepair=False,
                                                         save_dir=save_dir, determine_threshold=1, params=params,
                                                         log_dir=log_dir,
                                                         new_issue_dir=new_issue_dir, root_path=base_dir)
                        iter_need_train -= 1
                        print("Time: {:.2f}\n".format((time.time() - train_start_time)))
                    except Exception as e:
                        print("{} failed.".format(os.path.join(base_dir, model_dir, mutant_dir)))
                        print(e)
                        iter_fail = True
                        break
                if not iter_fail:
                    mutant_need_train -= 1
        if not has_mutant_need_train:
            model_finished += 1

        # reset timer
        duration = time.time() - start_time
        time_list.append(duration)
        print("Time for each model: %.2f s" % duration)
        start_time = time.time()

    print(
        "Dataset: {} | Model: {}/{} | Mutant: {}/{} | Iter remain {} | "
        "Time: {:.2f} s, Average {:.2f} s.".format(
            dataset_name, model_cnt, model_finished, mutant_cnt, mutant_need_train, iter_need_train, sum(time_list),
            sum(time_list) / len(time_list)))
