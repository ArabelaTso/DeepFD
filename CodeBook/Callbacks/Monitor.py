import os
import sys

sys.path.append('../Utils')
import matplotlib.pyplot as plt
import numpy as np
import keras
import datetime
from keras.models import load_model, Sequential
import keras.backend as K
import tensorflow as tf
from CodeBook.Utils.Logger import Logger
import keras.backend as K
import copy
import time
import pickle
import uuid
from collections import defaultdict

logger = Logger()


def inclusion_ratio(x, nparray):
    return float(np.sum(nparray == x)) / float(nparray.size)


def has_NaN_or_Inf(output):
    if output[0] is None:
        return False
    output = np.array(output)
    result = (np.isnan(output).any() or np.isinf(output).any())
    return result


def is_constant_array(output):
    return len(np.unique(np.array(output))) == 1


def gradient_zero_radio(gradient_list):
    kernel = []
    bias = []
    total_zero = 0
    total_size = 0
    for i in range(len(gradient_list)):
        zeros = np.sum(gradient_list[i] == 0)
        total_zero += zeros
        total_size += gradient_list[i].size
        if i % 2 == 0:
            kernel.append(zeros / gradient_list[i].size)
        else:
            bias.append(zeros / gradient_list[i].size)
    total = float(total_zero) / float(total_size)
    return total, kernel, bias


def get_weights(model, x, batch_size):
    trainingExample = x[:batch_size, ...]
    inp = model.input
    layer_outputs = []
    if model.layers[0].get_config()['name'].split('_')[0] == 'input':
        for layer in model.layers[1:]:
            layer_outputs.append(layer.output)
    else:
        for layer in model.layers[0:]:
            layer_outputs.append(layer.output)
    functor = K.function([inp] + [K.learning_phase()], layer_outputs)
    outputs = functor([trainingExample, 0])
    wts = model.get_weights()
    return outputs, wts


def max_delta_acc(acc_list):
    max_delta = 0
    for i in range(len(acc_list) - 1):
        if acc_list[i + 1] - acc_list[i] > max_delta:
            max_delta = acc_list[i + 1] - acc_list[i]
    return max_delta


def gradient_norm(gradient_list):
    assert len(gradient_list) % 2 == 0
    norm_kernel_list = []
    norm_bias_list = []
    for i in range(int(len(gradient_list) / 2)):
        # average_kernel_list.append(np.mean(np.abs(gradient_list[2*i])))
        # average_bias_list.append(np.mean(np.abs(gradient_list[2*i+1])))
        norm_kernel_list.append(np.linalg.norm(np.array(gradient_list[2 * i])))
        norm_bias_list.append(np.linalg.norm(np.array(gradient_list[2 * i + 1])))
    return norm_kernel_list, norm_bias_list


def ol_judge(history, threshold, rate):
    acc = history['acc']
    maximum = []
    minimum = []
    count = 0
    for i in range(len(acc)):
        if i == 0 or i == len(acc) - 1:
            continue
        if acc[i] - acc[i - 1] >= 0 and acc[i] - acc[i + 1] >= 0:
            maximum.append(acc[i])
        if acc[i] - acc[i - 1] <= 0 and acc[i] - acc[i + 1] <= 0:
            minimum.append(acc[i])
    for i in range(min(len(maximum), len(minimum))):
        if maximum[i] - minimum[i] >= threshold:
            count += 1
    if count >= rate * len(acc):
        return True
    else:
        return False


def loss_issue(feature_dic, history, total_epoch, satisfied_acc, checkgap,
               unstable_threshold=0.05, judgment_point=0.3, unstable_rate=0.25, epsilon=10e-3, sc_threshold=0.01):
    train_loss = history['loss']
    train_acc = history['acc']
    test_loss = history['val_loss']
    test_acc = history['val_acc']
    count = 0

    feature_dic['loss'] = train_loss[-1] if train_loss[-1] else 0.0
    feature_dic['acc'] = train_acc[-1] if train_acc[-1] else 0.0
    feature_dic['val_loss'] = test_loss[-1] if test_loss[-1] else 0.0
    feature_dic['val_acc'] = test_acc[-1] if test_acc[-1] else 0.0

    if train_loss:
        if has_NaN_or_Inf(test_loss) or has_NaN_or_Inf(train_loss) or (
                (not test_loss[-1] is None) and test_loss[-1] >= 1e+5):
            feature_dic['nan_loss'] += 1
            return feature_dic

        if has_NaN_or_Inf(train_acc) or has_NaN_or_Inf(test_acc):
            # or ((not train_acc[-1] is None) and test_acc[-1] >= 1e+5):
            feature_dic['nan_acc'] += 1
            return feature_dic
        current_epoch = len(train_loss)

        unstable_count = 0
        total_count = current_epoch - 1
        if current_epoch >= judgment_point * total_epoch:
            if test_acc[-1] is not None:
                if (train_acc[-1] <= 0.9 and train_acc[-1] - test_acc[-1] >= 0.1) \
                        or (train_acc[-1] > 0.9 and train_acc[-1] - test_acc[-1] >= 0.07):
                    feature_dic['test_not_well'] += 1

            if has_NaN_or_Inf(train_acc) or has_NaN_or_Inf(test_acc):
                feature_dic['nan_acc'] = True

            bad_count = 0

            for i in range(total_count):
                try:
                    if ((train_loss[-i - 1] - train_loss[-i - 2]) < -epsilon) and (
                            (test_loss[-i - 1] - test_loss[-i - 2]) > epsilon):  # train decrease and test increase.
                        bad_count += 1
                except TypeError:
                    pass
                else:
                    feature_dic['test_turn_bad'] = bad_count
                    break

            if ol_judge(history, unstable_threshold, unstable_rate):
                feature_dic['unstable_loss'] = True
            if not test_acc[0] is None and not train_acc[0] is None:
                if max(test_acc) < satisfied_acc or max(train_acc) < satisfied_acc:
                    feature_dic['not_converge'] = True
                if max_delta_acc(test_acc) < sc_threshold and max_delta_acc(train_acc) < sc_threshold:
                    feature_dic['sc_accuracy'] = True
            # if loss increases
            if len(train_loss) > 1:
                feature_dic['increase_loss'] += int(history['loss'][-1] > history['loss'][-2])
            # if acc decrease
            if len(train_acc) > 1:
                feature_dic['decrease_acc'] += int(history['acc'][-1] < history['acc'][-2])

    return feature_dic


def weights_issue(feature_dic, weights, last_weights, threshold_large=5, threshold_change=0.1):
    """[summary]
    :param
        weights ([type]): [description]
        threshold ([type]): [description]
        'large_weight':0,
        'nan_weight':False,
        'weight_change_little':0,
         feature_dic: dict of features
    """
    for i in range(len(weights)):
        if has_NaN_or_Inf(weights[i]):
            feature_dic['nan_weight'] = True
            return feature_dic
    for j in range(len(weights)):
        if np.abs(weights[j]).max() > threshold_large:
            feature_dic['large_weight'] += 1
            break

    for cur, last in zip(weights, last_weights):
        if np.mean(cur) == np.mean(last):
            feature_dic['cons_mean_weight'] += 1
        if np.std(cur) == np.std(last):
            feature_dic['cons_std_weight'] += 1

    return feature_dic


def gradient_issue(feature_dic, gradient_list, threshold_low=1e-3, threshold_low_1=1e-4, threshold_high=70,
                   threshold_die_1=0.7):
    [norm_kernel, avg_bias, gra_rate], \
    [total_ratio, kernel_ratio, bias_ratio, max_zero] \
        = gradient_message_summary(gradient_list)

    for i in range(len(gradient_list)):
        if has_NaN_or_Inf(gradient_list[i]):
            feature_dic['nan_gradient'] = True
            return feature_dic

    if gra_rate < threshold_low and norm_kernel[0] < threshold_low_1:
        feature_dic['vanish_gradient'] += 1
    if gra_rate > threshold_high:
        feature_dic['explode_gradient'] += 1
    if total_ratio >= threshold_die_1:  # or max_zero>=threshold_die_2
        feature_dic['died_relu'] += 1
    return feature_dic


def gradient_message_summary(gradient_list):
    total_ratio, kernel_ratio, bias_ratio = gradient_zero_radio(
        gradient_list)
    max_zero = max(kernel_ratio)

    norm_kernel, norm_bias = gradient_norm(gradient_list)
    gra_rate = (norm_kernel[0] / norm_kernel[-1])
    return [norm_kernel, norm_bias, gra_rate], [total_ratio, kernel_ratio, bias_ratio, max_zero]


class IssueMonitor:
    def __init__(self, total_epoch, satisfied_acc, params, determine_threshold=1):
        """[summary]

        Args:
            model ([model(keras)]): [model]
            history ([dic]): [training history, include loss, val_loss, acc, val_acc]
            gradient_list ([list]): [gradient of the weights in the first batch]
        """
        self.satisfied_acc = satisfied_acc
        self.total_epoch = total_epoch
        self.determine_threshold = determine_threshold
        self.issue_list = []
        self.last_weight = []

        self.feature = {
            'not_converge': False,  #
            'unstable_loss': False,  #
            'nan_loss': 0,  #
            'test_not_well': 0,  # test acc and train acc has big gap
            'test_turn_bad': 0,
            # 'not_trained_well':0,
            'sc_accuracy': False,

            'died_relu': False,  #
            'vanish_gradient': False,  #
            'explode_gradient': False,  #
            'nan_gradient': False,  #

            'large_weight': 0,  #
            'nan_weight': False,  #
            'weight_change_little': 0,  #

            # newly added
            'decrease_acc': 0,
            'increase_loss': 0,
            'cons_mean_weight': 0,
            'cons_std_weight': 0,
            'nan_acc': 0
        }
        self.params = params
        self.initial_feature = copy.deepcopy(self.feature)

    def determine(self, model, history, gradient_list, checkgap):
        # no issue model should has train or test acc better than satisfied acc and no unstable.
        self.history = history
        self.gradient_list = gradient_list
        self.weights = model.get_weights()
        self.feature.update(self.history)
        start_time = time.time()
        self.feature = gradient_issue(self.feature, self.gradient_list, threshold_low=self.params['beta_1'],
                                      threshold_low_1=self.params['beta_2'],
                                      threshold_high=self.params['beta_3'], threshold_die_1=self.params['gamma'])
        print("Time for gradient: {:.2f}".format((time.time() - start_time)))
        start_time = time.time()

        self.feature = weights_issue(self.feature, self.weights, self.last_weight)
        print("Time for weight: {:.2f}".format((time.time() - start_time)))
        start_time = time.time()

        self.feature = loss_issue(self.feature, self.history, total_epoch=self.total_epoch,
                                  satisfied_acc=self.params['Theta'], checkgap=checkgap,
                                  unstable_threshold=self.params['zeta'], unstable_rate=self.params['eta'],
                                  sc_threshold=self.params['delta'])
        print("Time for loss: {:.2f}".format((time.time() - start_time)))

        self.last_weight = self.weights

        # issue determine.

        if not self.issue_list:
            if self.feature['nan_loss'] or self.feature['nan_weight'] or self.feature['nan_gradient']:
                self.issue_list.append('explode')
            if self.feature['not_converge'] or self.feature['sc_accuracy'] == True:
                if self.feature['died_relu'] >= self.determine_threshold:
                    self.issue_list.append('relu')
                elif self.feature['explode_gradient'] >= self.determine_threshold:
                    #  or (self.feature['large_weight']>=self.determine_threshold):
                    self.issue_list.append('explode')
                elif self.feature['vanish_gradient'] >= self.determine_threshold:
                    self.issue_list.append('vanish')
                elif self.feature['sc_accuracy']:
                    self.issue_list.append('not_converge')
            # if self.feature['test_turn_bad'] + self.feature['test_not_well']>self.determine_threshold:
            #     self.issue_list.append('overfit')
            elif self.feature['unstable_loss']:
                self.issue_list.append('unstable')
            # if self.feature['test_turn_bad']>self.determine_threshold or self.feature['test_not_well']>self.determine_threshold:self.issue_list.append('overfit')
            self.issue_list = list(set(self.issue_list))
        return self.issue_list

    def get_features(self):
        return self.feature
