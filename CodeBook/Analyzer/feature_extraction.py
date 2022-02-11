import os
import pandas as pd
from collections import defaultdict
from datetime import datetime
import CodeBook.Repair.repair as rp
from copy import deepcopy
import argparse
from CodeBook.Config import *
from CodeBook.Utils.analysis_utils import has_enough_feature, has_enough_sample, convert_bool2int
from CodeBook.Utils.FileHandler import read_csv


def _severity_val(exp: float, real: float) -> float:
    degree = (exp - real) / exp
    return degree


def _severity_degree(val):
    if val < SEVERITY_DICT["LOW"]:
        return "LOW"
    elif val > SEVERITY_DICT["HIGH"]:
        return "HIGH"
    else:
        return "MEDIUM"


def get_label_dict(input: str) -> dict:
    label_set = set()
    if input.startswith("mul"):
        segs = input.split("__")
        for seg in segs[1:]:
            fault_label = seg.split('_')[0]
            if "lb_{}".format(fault_label) in LABEL_DICT:
                label_set.add(fault_label)
            else:
                print("Invalid label name:", input)
    else:
        fault_label = input.split('_')[0]
        if "lb_{}".format(fault_label) in LABEL_DICT:
            label_set.add(fault_label)
        else:
            print("Invalid label name:", input)
    print("Extract labels from {}: {}".format(input, ",".join(label_set)))

    label_dict = deepcopy(LABEL_DICT)
    for lb in label_set:
        label_dict["lb_{}".format(lb)] = 1
    print("label_dict", label_dict)
    return label_dict


def update_agg_by_dict(agg, label_dict: dict) -> dict:
    for label in label_dict:
        agg[label].append(label_dict[label])
    return agg


def has_negative_last_act(model) -> bool:
    NEG_ACT = {"tanh", "softsign", "elu", "selu", "leakyrelu", 'LeakyReLU', 'ELU', 'ThresholdedReLU', "linear"}
    cur_act = model.layers[int(rp.last_layer(model.layers))].get_config()["activation"]
    if cur_act is None:
        # print("Cannot find act in last layer!")
        raise ValueError("Cannot find act in last layer! ")
    return False if cur_act in NEG_ACT else True


def output_dim_last_layer(model):
    return model.layers[int(rp.last_layer(model.layers))].output_shape[-1]

# if __name__ == '__main__':
#     # print(sys.argv[0])
#     # print(os.path.dirname(sys.argv[0]))
#     parser = argparse.ArgumentParser(description="Feature extraction.")
#     parser.add_argument('--dataset', '-ds', default='MNIST', help="Dataset name")
#     parser.add_argument('--parent_path', '-pp', default="Programs", help="Root path to the dataset")
#     parser.add_argument('--result_dir', '-rs', default="result_dir", help="Result directory")
#     parser.add_argument('--feat_file_name', '-ffn', default="monitor_features.csv", help="Name of Feature file")
#     args = parser.parse_args()
#
#     parent_path = args.parent_path
#     dataset = args.dataset
#     result_dir = args.result_dir
#     feat_file_name = args.feat_file_name
#
#     timestamp = datetime.now().strftime("%y%m%d%H%M%S")
#     output_path = os.path.join(parent_path, dataset, "stat_{}.csv".format(timestamp))
#
#     agg = defaultdict(list)
#
#     for index, program in enumerate(os.listdir(os.path.join(parent_path, dataset))):
#         print("\n", index, "Processing dataset {} under {}".format(dataset, parent_path))
#
#         program_dir = os.path.join(os.path.join(parent_path, dataset, program))
#         if not os.path.isdir(program_dir):
#             continue
#
#         for faulty_dir in os.listdir(program_dir):
#             print("Handling", program_dir, faulty_dir)
#
#             # if it is not a dir, continue
#             if not os.path.isdir(os.path.join(program_dir, faulty_dir)):
#                 print("Skip. Not a dir.")
#                 continue
#
#             # if monitor_features.csv does not exist, then skip
#             if not os.path.exists(os.path.join(program_dir, faulty_dir, "result_dir", "monitor_features.csv")):
#                 print("Skip. result_dir or monitor_features.csv do not exist.")
#                 continue
#
#             label_dict = get_label_dict(faulty_dir)
#
#             cur_acc = 0.0
#             # if best model under this setting is better than SAT_ACC, then reset its label
#             if os.path.exists(os.path.join(program_dir, faulty_dir, "result_dir", "checkpoint_model")):
#                 for file in os.listdir(os.path.join(program_dir, faulty_dir, "result_dir", "checkpoint_model")):
#                     try:
#                         cur_acc = max(float(file.replace(".h5", "").split("_")[-1]), cur_acc)
#                     except ValueError as e:
#                         continue
#
#                     if cur_acc >= SAT_ACC[dataset]:
#                         print("Reset Label to 'origin' because acc = {} >= {}".format(cur_acc, SAT_ACC[dataset]))
#                         label_dict = deepcopy(LABEL_DICT)
#                         break
#
#             sev_val = _severity_val(SAT_ACC[dataset], cur_acc)
#             sev_degree = _severity_degree(sev_val)
#
#             # concat path
#             file_path = os.path.join(program_dir, faulty_dir, result_dir, feat_file_name)
#
#             # read csv, extract features
#             df = read_csv(file_path)
#             if has_enough_feature(df, min_feature=10) and has_enough_sample(df, min_sample=5):
#                 # preprocess, convert bool dtype ot int if necessary
#                 df = convert_bool2int(df)
#
#                 # feature dimension is wrong
#                 # if df.shape[1] != 22:
#                 #     print("Incorrect feature dimension, should be 22, got", df.shape[1])
#                 #     continue
#
#                 # add program id, labels and label values
#                 agg["program_id"].append(program)
#                 agg["Folder_id"].append(faulty_dir)
#                 agg = update_agg_by_dict(agg, label_dict)
#
#                 # agg["label"].append(label)
#                 # agg["severity_val"].append(sev_val)
#                 # agg["severity"].append(sev_degree)
#
#                 agg = extract_feature(df, agg)
#             print("Done\n")
#
#     # for debug only
#     # for k, v in agg.items():
#     #     print(k, len(v))
#
#     agg_df = pd.DataFrame.from_dict(agg)
#     agg_df.to_csv(output_path)
#     print("Export output to ", output_path)
