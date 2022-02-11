from collections import defaultdict
from CodeBook.Config import *
import pandas as pd
import numpy as np


def parse_fault_str(faulty_str: str) -> (list, dict):
    fault_value_dict = defaultdict(str)
    fault_label_dict = defaultdict(int)

    if faulty_str == "origin":
        return [""] * len(FAULT_TYPE), fault_label_dict
    # if input.startswith("mul"):
    segs = faulty_str.split("__")
    for seg in segs:
        fault_label = seg.split('_')[0]
        if fault_label in FAULT_TYPE:
            fault_value_dict[fault_label] = "_".join(seg.split('_')[1:])
        else:
            print("Fail to parse faulty string {} into list.".format(faulty_str))

    output_list = []
    for fault in FAULT_TYPE:
        output_list.append(fault_value_dict[fault])
        if len(fault_value_dict[fault]) > 0:
            fault_label_dict[fault] = 1
        # print(fault, fault_value_dict[fault])
    return output_list, fault_label_dict


def extract_feature_old(df: pd.DataFrame, x_dict: dict):
    cols = df.columns
    features = {k: OPERATORS for k in cols}
    extracted_feat = df.agg(features).to_dict()

    for para, values in extracted_feat.items():
        for p, v in values.items():
            key = "ft_{}_{}".format(para, p)

            # handle exceptional value
            if type(v) == str and v == "False":
                v = 0.0

            if type(v) == str and v == "1":
                v = 1.0

            if type(v) != float:
                print("Type", type(v), v)
            x_dict[key].append(v)
    return x_dict


def extract_feature(df: pd.DataFrame):
    feature_dict = {}

    features = {k: OPERATORS for k in df.columns}
    extracted_feat = df.agg(features).to_dict()

    for para, values in extracted_feat.items():
        for p, v in values.items():
            key = "ft_{}_{}".format(para, p)

            # handle exceptional value
            if type(v) == str and (v == "0" or v == v == "False"):
                v = 0.0

            if type(v) == str and (v == "1" or v == "True"):
                v = 1.0

            if type(v) != float:
                print("Type", type(v), v)
            feature_dict[key] = v
    return feature_dict


def has_enough_sample(df: pd.DataFrame, min_sample=10) -> bool:
    # check if the csv has too few samples.
    return df.shape[0] >= min_sample


def has_enough_feature(df: pd.DataFrame, min_feature=5) -> bool:
    # check if the csv has too few feature.
    return df.shape[1] >= min_feature


def convert_bool2int(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, df.dtypes == 'bool'] = df.loc[:, df.dtypes == 'bool'].astype('float32')
    return df


def cal_metrics_voting(predict_list):
    # shape of predict_list (n_classifier, n_label)
    predict_list = np.array(predict_list)
    # threshold = predict_list.shape[0] / 2
    threshold = 1
    # get union of the diagnosed faults
    pred_voting = (np.sum(predict_list, axis=0) >= threshold).astype(int)  # if # of 1 > # of 0 -> 1, else -> 0
    return pred_voting

