import os
import pandas as pd
from sklearn.decomposition import PCA
from CodeBook.Utils.FileHandler import find_file_by_suffix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from CodeBook.MultiLabelClassification.MultiLabelClassifier import label_encoding
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.ticker as plticker


def pca_decompose(X, n_comp=None):
    if n_comp is None:
        n_comp = max(1, X.shape[1] / 2)
    pca = PCA(n_components=int(n_comp))
    pca.fit(X)
    print(pca.explained_variance_ratio_)


def balance(df_all, FOCUS):
    sampled = df_all[df_all[FOCUS] == "origin"].sample(n=100, random_state=1)
    return pd.concat([sampled, df_all[df_all[FOCUS] != "origin"]])


if __name__ == '__main__':
    # configurations
    root_path = "../Programs/"
    dataset = "CIFAR-10"
    file = find_file_by_suffix(os.path.join(root_path, dataset), suffix=".csv")[-1]
    FOCUS = "label"  # "severity"   "severity_val"
    topK = 10
    score_func = chi2  # "f_classif  chi2 f_regression

    df = pd.read_csv(os.path.join(root_path, dataset, file))
    print("Read in {}, shape:{}".format(str(os.path.join(root_path, dataset, file)), df.shape))

    if dataset == "MNIST":
        df = balance(df, FOCUS)

    X = df[list(filter(lambda x: x.startswith("ft_"), df.columns))]
    # X = X[list(filter(lambda x: (not x.startswith("ft_acc_")) and (not x.startswith("ft_loss_")), X.columns))]
    Y = df[FOCUS]

    features = np.array(X.columns)

    # debug only
    # print("features", features)
    # print(X.describe())

    # fill nan in X
    X = X[X > 0]
    X = X.fillna(0.0).astype("float")

    # specify classifiers' name
    names = ["Nearest Neighbors",
             "RBF SVM",
             "Gaussian Process",
             "Decision Tree",
             "Random Forest",
             "Neural Net",
             "AdaBoost",
             "Naive Bayes",
             "QDA"
             ]

    # define classifiers
    classifiers = [
        KNeighborsClassifier(n_neighbors=len(set(Y))),
        SVC(gamma=2, C=4, decision_function_shape='ovr'),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=4),
        RandomForestClassifier(max_depth=4, n_estimators=16),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    #  #############################################################################
    history = []

    for topK in range(1, 21, 1):
        cur = []
        print("topk ", topK)
        # print(len(history), len(history[0]))

        SLK = SelectKBest(score_func, k=topK)
        cur_X = SLK.fit_transform(X, Y)
        masks = SLK.get_support()

        slct_features = []
        masks = SLK.get_support()

        slct_features = []

        i = 0
        for b, f in zip(masks, features):
            if b:
                slct_features.append(f)
                # debug only
                # print(i, f)
            i += 1

        print(cur_X.shape)
        print("Dataset: {}, Target: {}, Score func: {}".format(dataset, FOCUS, score_func))
        print("Top {}:\n{}".format(topK, slct_features))

        slct_set = {"_".join(x.split("_")[1:-1]) for x in slct_features}
        print("\nKey features:{}\n{}".format(len(slct_set), slct_set))

        le, cur_Y = label_encoding(Y)
        cur_X = pd.DataFrame(cur_X, columns=slct_features)

        # 0 #############################################################################
        # Create a list of the feature names

        # Instantiate the visualizer
        # visualizer = FeatureCorrelation(feature_names=slct_features)
        #
        # visualizer.fit(X, Y)  # Fit the data to the visualizer
        # visualizer.show()  # Finalize and render the figure

        X_train, X_test, y_train, y_test = train_test_split(cur_X, cur_Y, test_size=0.3, random_state=42)
        print(X_train.shape, X_test.shape)

        print("\nPrediction Result using top {} features, predicting {}".format(topK, FOCUS))
        for name, clf in zip(names, classifiers):
            try:
                clf.fit(X_train, y_train)
            except ValueError as e:
                print("Failed: {}".format(name))
                print(e)
                continue

            # when model fitted, evaluate it
            score = clf.score(X_test, y_test)
            print(name, score)
            cur.append(score)
        history.append(cur)

    # print("history", history, len(history), len(history[0]))
    history = np.transpose(np.array(history))
    print(history.shape)

    fig, ax = plt.subplots()
    for name, line in zip(names, history):
        ax.plot(np.arange(1, len(line) + 1, 1), np.array(line), label=name)

    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
    ax.set_xlabel('Top K features')
    ax.set_ylabel('Precision')
    ax.grid(True)

    ax.legend(loc="lower left")
    plt.show()

    # 1 ##################
    # pca_decompose(X, 10)

    # 2 ###################################
    # svc = SVC(kernel="linear", C=1)
    # rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    # rfe.fit(X, Y)
    # ranking = rfe.ranking_.reshape(X[0].shape)
    #
    # # Plot pixel ranking
    # plt.matshow(ranking, cmap=plt.cm.Blues)
    # plt.colorbar()
    # plt.title("Ranking of pixels with RFE")
    # plt.show()

    # 3 #############################################################################
    # Classification accuracy without selecting features: 0.955
    # Classification accuracy after univariate feature selection: 0.910

    # plt.figure(1)
    # plt.clf()
    #
    # X_indices = np.arange(X.shape[-1])

    # #############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function to select the four
    # most significant features
    # selector = SelectKBest(f_classif, k=4)
    # selector.fit(X_train, y_train)
    # scores = -np.log10(SLK.pvalues_)
    # scores /= scores.max()
    # plt.bar(X_indices - .45, scores, width=.2,
    #         label=r'Univariate score ($-Log(p_{value})$)')
    #
    # # ###########
    # # Compare to the weights of an SVM
    # clf = make_pipeline(MinMaxScaler(), LinearSVC())
    # clf.fit(X_train, y_train)
    # print('Classification accuracy without selecting features: {:.3f}'
    #       .format(clf.score(X_test, y_test)))
    #
    # svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
    # svm_weights /= svm_weights.sum()
    #
    # plt.bar(X_indices - .25, svm_weights, width=.5, label='SVM weight')
    #
    # clf_selected = make_pipeline(
    #     SelectKBest(f_classif, k=4), MinMaxScaler(), LinearSVC()
    # )
    # clf_selected.fit(X_train, y_train)
    # print('Classification accuracy after univariate feature selection: {:.3f}'
    #       .format(clf_selected.score(X_test, y_test)))
    #
    # svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    # svm_weights_selected /= svm_weights_selected.sum()
    #
    # plt.bar(X_indices[SLK.get_support()] - .05, svm_weights_selected,
    #         width=.5, label='SVM weights after selection')
    #
    # plt.title("Comparing feature selection")
    # plt.xlabel('Feature number')
    # plt.yticks(())
    # plt.axis('tight')
    # plt.legend(loc='upper right')
    # plt.show()
