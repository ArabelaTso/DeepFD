import os
import time
import random
import pickle
import argparse
import pandas as pd
from joblib import dump, load
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import MultiLabelBinarizer
import sys

sys.path.append("./")
from CodeBook.Utils.analysis_utils import cal_metrics_voting
from CodeBook.Analyzer.lineLoc import lineLoc


def label_encoding(raw_labels, labelEncoder=None):
    if labelEncoder is None:
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder = labelEncoder.fit(raw_labels)
    enc_label = labelEncoder.transform(raw_labels)
    # print('classes: {}'.format(', '.join(labelEncoder.classes_)))
    return labelEncoder, enc_label


def label_decoder(le, enc_label):
    return le.inverse_transform(enc_label)


def multi_label_binarizer(raw_labels, labelEncoder=None):
    if labelEncoder is None:
        labelEncoder = preprocessing.MultiLabelBinarizer()
        labelEncoder = labelEncoder.fit(raw_labels)
    enc_label = labelEncoder.transform(raw_labels)
    # print('classes: {}'.format(', '.join(labelEncoder.classes_)))
    return labelEncoder, enc_label


def train(x_train, y_train, n_features):
    clf = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=n_features)).fit(x_train, y_train)
    return clf


def predict(clf, x_test):
    return clf.predict(x_test)


def balance(df_all, FOCUS, n_sample=100):
    sampled = df_all[df_all[FOCUS] == "origin"].sample(n=n_sample, random_state=1)
    df_new = pd.concat([sampled, df_all[df_all[FOCUS] != "origin"]])
    print("Sample before: {}, after: {}".format(df_all.shape, df_new.shape))
    return df_new


def generateReport(dataset, label_dict):
    # shape of label_dict: {0: {"TP":xx, "TF":xx,...}, 1: {..}.., ..}
    df = pd.DataFrame.from_dict(label_dict)
    dataset = dataset.split("/")[1]
    outdir = './Logs'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    df.to_csv("{}/{}_label.csv".format(outdir, dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MultiLable Classifier Training.")
    parser.add_argument('--program_dir', '-pd', default='./Evaluations', help='program path.')
    parser.add_argument('--dataset', '-ds', default='Subjects', help='dataset path.')
    parser.add_argument('--model_dir', '-md', default='./Classifiers', help='Output classifier path.')
    parser.add_argument('--filename', '-fn', default="summary.csv", help='Output csv file name.')
    parser.add_argument('--overwrite', '-ow', default=0, choices=[0, 1], type=int, help='Overwrite or not.')
    parser.add_argument('--iteration', '-it', default=0, help='Iteraction focused.')
    parser.add_argument('--impact', '-imp', type=float, default=0.05, help='Upperbound impact.')
    parser.add_argument('--threshold', '-thr', type=float, default=0.8, help='Lowerbound for validation accuracy')

    args = parser.parse_args()

    # configurations
    program_dir = args.program_dir
    dataset_dir = args.dataset
    file = args.filename
    if "all" in args.model_dir:
        # use all the pretrained model to diagnose
        model_dir = os.listdir("./Classifiers/")
        model_dir = ["./Classifiers/" + path for path in model_dir]
    else:
        model_dir = [args.model_dir]
    overwrite = args.overwrite
    iteraction = max(0, args.iteration)
    upper_impact = args.impact
    lower_threshold = args.threshold

    base_dir = os.path.join(program_dir, dataset_dir)

    # print out the dataset name
    # print("\n\n=================================\nWorking on model: {}".format(args.model_dir))

    # read csv
    df = pd.read_csv(os.path.join(base_dir, file))
    print("Read in {}, shape:{}".format(file, df.shape))

    # fill nan as 0.0
    df = df.fillna(0.0)
    # print("max", df.max())

    # select samples by iteration.
    if iteraction > 0:
        df = df[df["Unnamed: 2"] == iteraction]
        print("Select Iteraction: {}. Shape: {}".format(iteraction, df.shape))
    else:
        print("Use all iteration.")

    # write separately because otherwise, got warning: Boolean Series key will be reindexed to match DataFrame index
    df_slct = df[df["Unnamed: 1"] == "origin"]
    df_slct = df_slct[df_slct["ft_val_accuracy"] < lower_threshold]
    model_list = set(df_slct["Unnamed: 0"].unique())

    df = df[~df["Unnamed: 0"].isin(model_list)]
    # print("Lowerbound for validation accuracy {}. Shape: {}".format(lower_threshold, df.shape))

    # debug only
    # print("DataFrame:\n", df)

    # spilt features (ft_) and labels(lb_)
    features = list(filter(lambda x: x.startswith("ft_"), df.columns))
    labels = list(filter(lambda x: x.startswith("lb_"), df.columns))
    X = df[features]
    Y = df[labels]

    # preprocessing X to get the normalized data
    iter_num = 9

    mask_index = ~(np.max(X, axis=0) == np.min(X, axis=0))
    mask_index = mask_index.values
    X.loc[:, mask_index] = (X.loc[:, mask_index] - np.min(X.loc[:, mask_index], axis=0)) / (
            np.max(X.loc[:, mask_index], axis=0) - np.min(X.loc[:, mask_index], axis=0))
    X.loc[:, ~mask_index] = 0.0
    X = X.astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # print info
    print("\nLabel Stats: {} ({})".format(len(labels), ", ".join(labels)))
    print("\nNumber of features: {}. Number of labels: {}".format(len(features), len(labels)))

    # specify classifiers' name
    names = [
        "Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
    ]

    C = len(names)
    predictList = []
    Y = Y.to_numpy()
    # print("the total label count is: ")
    # print(np.sum(Y, axis=0))

    modelPredictList = []
    for model in model_dir:
        predictList = []
        for cls_name in names:
            start_time = time.time()

            classifier_dir = os.path.join(model)
            os.makedirs(classifier_dir, exist_ok=True)

            pkl_path = os.path.join(classifier_dir, "{}.pkl".format(cls_name))
            clf = load(pkl_path)

            predictList.append(predict(clf, X))

        N = Y.shape[0]
        total_acc, total_prec, total_rec, total_fsc = 0, 0, 0, 0
        predictList = np.array(predictList)
        modelPredictList.append(predictList)

        for i in range(0, N, iter_num):
            predict = predictList[:, i:i + iter_num, :]  # shape of predict: [n_classifier, iter, 5]
            predict_sum = np.sum(predict, axis=1)  # shape of predict_sum: [n_classifier, 5]
            print("{}\nWorking on case: ".format("=" * 20, (i / 9)))
            print("working on subject: ", df["Unnamed: 0"][i])

            # WARNING, this line is added to alleviate the training bias problem in the prediction stage
            # Theoretically, this line should be `predict_sum_shift = predict_sum`
            predict_sum_shift = predict_sum - np.array([0, 2, 3, 6, 5])
            predict_sum_shift[predict_sum_shift <= 0] = 0
            predict_sum_shift[predict_sum_shift > 0] = 1

            pred_voting = cal_metrics_voting(predict_sum_shift)  # shape of pred: [n_classifier, 5]
            # if precision != 0:
            # print(predict_sum)
            label_name = ['optimizer', 'lr', 'act', 'loss', 'epoch']

            gt_label = np.where(Y[i] == 1)[0]
            gt_list = []
            for label_index in gt_label:
                gt_list.append(label_name[label_index])
            print("The ground truth faults: {}".format(gt_list))

            # print("the label is: {}, the predicted label is: {}.".format(Y[i], pred_voting))
            lineLoc(program_dir, dataset_dir, pred_voting, label_name, df["Unnamed: 0"][i])
            print("case end \n{}\n".format("=" * 20))
