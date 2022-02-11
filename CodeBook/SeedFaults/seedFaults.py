import os
import random
import pickle
from copy import deepcopy
import CodeBook.Repair.repair as rp
import keras.optimizers as O
from CodeBook.Config import *
from keras.engine.saving import load_model
from CodeBook.Analyzer.ASTParser import traverse


def index_faults(faults_dict: dict):
    faults_indexs = {}
    faults_list = []
    i = 0
    for k, vals in faults_dict.items():
        for val in vals:
            # faults_indexs[i] = (k, val)
            faults_list.append((k, val))
            # faults_list.append("{}_{}".format(k, val))
            i += 1
    print("Total faults: ", i)
    return faults_list


def random_select_fault(faults_list: list, f_num=1):
    print("select {} fault(s).".format(f_num))
    random.randint(1, 10)
    return random.sample(faults_list, f_num)


def modify_loss(model, training_config: dict, value=None) -> (dict, str):
    """
    modify loss
    - if the original loss belongs to probabilistic loss, then sample a regression loss
    - vice versa
    """
    cur_loss = training_config["loss"]

    # losses = deepcopy(LOSS_POOL)
    # if cur_loss in losses:
    #     losses.remove(cur_loss)

    if value is None:
        if cur_loss in LOSS_POOL["prob"]:
            new_loss = random.sample(LOSS_POOL["regression"], 1)[0]
        elif cur_loss in LOSS_POOL["regression"]:
            new_loss = random.sample(LOSS_POOL["prob"], 1)[0]
        else:
            new_loss = random.sample([lo for ltype in LOSS_POOL for lo in LOSS_POOL[ltype]], 1)[0]
    else:
        new_loss = value

    training_config["loss"] = new_loss

    print("loss: [before]: {}, [after]: {}".format(cur_loss, new_loss))
    return model, training_config, "loss_{}".format(new_loss)


def modify_lr(model, training_config: dict, target=None) -> (dict, str):
    if "opt_kwargs" not in training_config or 'lr' not in training_config["opt_kwargs"]:
        # if lr is not in the training_config, then use the default lr value. (0.0001)
        training_config["opt_kwargs"] = {}
        training_config["opt_kwargs"]["lr"] = float("1e-2")

    cur_lr = training_config["opt_kwargs"]["lr"]
    new_lr = cur_lr

    if target is not None:
        new_lr = float(target)
    else:
        # direction decides on which direction the lr should be modified.
        # +1: larger than 1, smaller than 11
        # -1: smaller than 1e-10, larger than 1e-16
        # if the random lr == cur_lr, generate again. Upperbound = 10 times
        iter_cnt = 0
        while iter_cnt < 10:
            direction = random.sample([-1, 1], 1)[0]
            if direction == 1:
                new_lr = random.sample([float(n) for n in range(1, 11)], 1)[0]
            else:
                new_lr = float("1e-{}".format(random.sample([num for num in range(10, 16, 1)], 1)[0]))
            if new_lr != cur_lr:
                break
            else:
                iter_cnt += 1
    training_config["opt_kwargs"]["lr"] = new_lr
    print("LR: [before]: {}, [after]: {}".format(cur_lr, new_lr))
    return model, training_config, "lr_{}".format(new_lr)


def modify_epoch(model, training_config: dict, value=None) -> (dict, str):
    cur_epoch = training_config["epoch"]
    if value is not None:
        new_epoch = int(value)
    else:
        times = random.randint(1, 5) * 10
        new_epoch = max(1, int(cur_epoch / times))

    # make sure new_epoch != cur_epoch, except cur_epoch is minimized (1)
    if new_epoch == cur_epoch:
        new_epoch = max(1, min(10, new_epoch) - 1)

    training_config["epoch"] = new_epoch
    print("Epoch: [before]: {}, [after]: {}".format(cur_epoch, new_epoch))
    return model, training_config, "epoch_{}".format(new_epoch)


def modify_opt(model, training_config: dict, value=None) -> (dict, str):
    cur_opt = training_config["optimizer"]
    if value is not None:
        new_opt = value
    else:
        opt_list = deepcopy(OPT)
        if cur_opt in opt_list:
            opt_list.remove(cur_opt)
        new_opt = random.sample(opt_list, 1)[0]

    training_config["optimizer"] = new_opt

    print("Optimizer: [before]: {}, [after]: {}".format(cur_opt, new_opt))
    return model, training_config, "opt_{}".format(new_opt)


def validate_opt_kwargs(training_config):
    opt_cls = getattr(O, training_config["optimizer"])
    optimizer = opt_cls()
    kwargs = optimizer.get_config()
    training_config["opt_kwargs"] = {k: v for k, v in training_config["opt_kwargs"].items() if k in kwargs}
    return training_config


def modify_last_act(model, training_config, target_act=None):
    cur_act, new_act = "", ""
    if target_act is None:
        act_list = deepcopy(ACT)
        cur_act = model.layers[int(rp.last_layer(model.layers))].get_config()["activation"]
        if cur_act is None:
            print("Cannot find act in last layer!")
            exit(0)
        if cur_act in act_list:
            act_list.remove(cur_act)
        new_act = random.sample(act_list, 1)[0]
    else:
        new_act = target_act

    new_model = rp.modify_activations_in_last_layer(model, new_act)

    print("Activation in last layer: [before]: {}, [after]: {}".format(cur_act, new_act))
    return new_model, training_config, "act_{}".format(new_act)


def seed_single_fault(model, training_config, fault_type, fault_val=None):
    suffix = ""
    # seed faults by modifying training config
    if fault_type == "lr":
        model, training_config, suffix = modify_lr(model, training_config, fault_val)
    elif fault_type == "loss":
        model, training_config, suffix = modify_loss(model, training_config, fault_val)
    elif fault_type == "epoch":
        model, training_config, suffix = modify_epoch(model, training_config, fault_val)
    elif fault_type == "opt":
        model, training_config, suffix = modify_opt(model, training_config, fault_val)
    elif fault_type == "act":
        model, training_config, suffix = modify_last_act(model, training_config, fault_val)

    training_config = validate_opt_kwargs(training_config)
    return model, training_config, suffix


def seed_multi_fault(model, config, faulty_model_dict, num=2):
    suffix = []

    # randomly select num of faults, each type can only be selected once, regardless how many values in it.
    slct_fault_types = random.sample(faulty_model_dict.keys(), num)

    for slct_fault_type in slct_fault_types:
        model, config, cur_suffix = seed_single_fault(model, config, slct_fault_type,
                                                      random.sample(faulty_model_dict[slct_fault_type], 1)[0])
        suffix.append(cur_suffix)
    return model, config, "__".join(suffix)


# if __name__ == '__main__':
# traverse("../MNIST", "../MNIST", overwrite=False)
# training_config = {'epoch': 10,
#                    "opt_kwargs":{'lr': 0.01}}
# new, st = modify_lr(training_config)
# print(new, st)
# model = load_model(os.path.abspath("../Programs/MNIST/config_random_1ed98a53-6d6f-466d-81ba-ffd8a252f5b8/lenet_seed.h5"))
# new_model, faulty_str = modify_last_act(model)
# print(faulty_str)
#
# with open(os.path.abspath("../Programs/MNIST/config_random_1ed98a53-6d6f-466d-81ba-ffd8a252f5b8/config_Adadelta_67c5bec6-1ff3-39c0-bb6a-bf8577c7bee3.pkl"), 'rb') as f:  # input,bug type, params
#     training_config = pickle.load(f)
# modify_opt(training_config)
