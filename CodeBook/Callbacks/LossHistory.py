import os
import sys

sys.path.append('.')
import csv
import numpy as np
import keras
from CodeBook.Utils.Logger import Logger
import keras.backend as K
import CodeBook.Callbacks.Monitor as mn
import pickle
import time
from collections import defaultdict

# import modules as md

logger = Logger()

default_param = {'beta_1': 1e-3,
                 'beta_2': 1e-4,
                 'beta_3': 70,
                 'gamma': 0.7,
                 'zeta': 0.03,
                 'eta': 0.2,
                 'delta': 0.01,
                 'alpha_1': 0,
                 'alpha_2': 0,
                 'alpha_3': 0,
                 'Theta': 0.7
                 }


class LossHistory(keras.callbacks.Callback):
    def __init__(self, training_data, model, batch_size, total_epoch, save_dir, determine_threshold=5,
                 satisfied_acc=0.7,
                 checktype='epoch_5', satisfied_count=3, retrain=False, pkl_dir=None, solution=None,
                 params={}):  # only support epoch method now
        """[summary]

        Args:
            training_data ([list]): [training data]
            model ([model]): [untrained model]
            batch_size ([int]): [batch size]
            save-dir([str]):[the dir to save the detect result]
            checktype (str, optional): [checktype,'a_b', a can be chosen from ['epoch', 'batch'], b is number, it means the monitor will check \
            the gradient and loss every 'b' 'a'.]. Defaults to 'epoch_5'.
            satisfied_acc (float, optional): [the satisfied accuracy, when val accuracy beyond this, the count ++, when the count is bigger or\
                equal to satisfied_count, training stop.]. Defaults to 0.7.
            satisfied_count (int, optional): []. Defaults to 3.
        """
        self.evaluated_gradients = 0
        self.trainX = training_data[0]
        self.trainy = training_data[1]
        self.batch_size = batch_size
        self.model = model
        self.satisfied_acc = satisfied_acc
        self.satisfied_count = satisfied_count
        self.count = 0
        self.checktype = checktype.split('_')[0]
        self.checkgap = int(checktype.split('_')[-1])
        self.issue_list = []
        self.feature_list = []
        self.save_dir = save_dir
        if not os.path.exists(save_dir):  # record monitor and repair message
            os.makedirs(save_dir)
        self.pkl_dir = pkl_dir
        self.retrain = retrain
        self.total_epoch = total_epoch
        self.determine_threshold = determine_threshold
        self.params = params
        if self.params == {}:
            self.params = default_param

        self.f = None
        self.history = defaultdict(list)
        self.history['loss'] = []
        self.history['acc'] = []
        self.history['val_loss'] = []
        self.history['val_acc'] = []

        self.Monitor = mn.IssueMonitor(total_epoch, self.satisfied_acc, self.params, self.determine_threshold)

        self.start_time = time.time()

        # name of log file
        self.log_name = '{}_{}.log'.format('monitor', 'detection')
        if self.retrain:
            self.log_name = '{}_{}.log'.format('monitor', 'repair')
            self.solution = solution
        self.log_name = os.path.join(self.save_dir, self.log_name)

        # name of features file
        self.feature_name = os.path.join(self.save_dir, '{}_{}.csv'.format('monitor', 'features'))

        if os.path.exists(self.log_name):
            # avoid the repeat writing
            os.remove(self.log_name)

        if os.path.exists(self.feature_name):
            # avoid the repeat writing
            os.remove(self.feature_name)

        # open two files
        self.log_file = open(self.log_name, 'a+')
        self.feature_log_file = open(self.feature_name, 'a+')

        # write head of two files
        self.log_file.write(
            '{},{},{},{},{}\n'.format('checktype', 'current_epoch', 'issue_list', 'time_usage', 'Describe'))

        self.feature_log_file.write(','.join(self.Monitor.get_features().keys()) + '\n')

    def on_train_begin(self, logs=None):
        weights = self.model.trainable_weights  # get trainable weights
        grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)

        # input, corresponding label, weight of each sample(all of them are 1), learning rate(we set it to 0)
        symb_inputs = [self.model._feed_inputs, self.model._feed_targets, self.model._feed_sample_weights,
                       K.learning_phase()]
        self.f = K.function(symb_inputs, grads)
        if self.retrain:
            self.log_file.write(
                '-----Using {} solution to retrain Detail can be found in the directory!-----\n'.format(self.solution))

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))

        if self.checktype == 'batch' and batch % self.checkgap == 0:
            trainingExample = self.trainX[0: self.batch_size]
            trainingY = self.trainy[0: self.batch_size]
            x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
            # output_grad = self.f(x + y + sample_weight)
            self.evaluated_gradients = self.f([x, y, sample_weight, 0])
            gradient_list = []
            for i in range(len(self.evaluated_gradients)):
                if isinstance(self.evaluated_gradients[i], np.ndarray):
                    gradient_list.append(self.evaluated_gradients[i])

            self.issue_list = self.Monitor.determine(self.model, self.history, gradient_list, self.checkgap)
            self.feature_list = self.Monitor.get_features()
            # self.issue_list = md.filtered_issue(self.issue_list)

            self.evaluated_gradients = None

            dictwriter_object = csv.DictWriter(self.feature_log_file, fieldnames=self.feature_list.keys())
            dictwriter_object.writerow(self.feature_list)
            self.feature_log_file.flush()

            if not self.retrain:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
                                                                  str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'NO training problems now. You need to train this '
                                                                  'new model more times.'))
                    self.log_file.flush()
                    print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'Found Issue Stop Training! Starting the repair '
                                                                  'procedure.'))
                    self.log_file.flush()
                    # self.log_file.close()
                    # self.model.stop_training = True
            else:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{,{}\n'.format(self.checktype, batch, self.issue_list,
                                                                  str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'NO training problems now. You need to train this '
                                                                  'new model more times.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    # print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, batch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'Found Issue Stop Training! Starting the repair '
                                                                  'procedure.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    # self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))

        if epoch % self.checkgap == 0 and self.checktype == 'epoch':
            trainingExample = self.trainX[0: self.batch_size, ]
            trainingY = self.trainy[0:self.batch_size]
            x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
            # output_grad = self.f(x + y + sample_weight)
            self.evaluated_gradients = self.f([x, y, sample_weight, 0])
            gradient_list = []
            for i in range(len(self.evaluated_gradients)):
                if isinstance(self.evaluated_gradients[i], np.ndarray):
                    gradient_list.append(self.evaluated_gradients[i])

            self.issue_list = self.Monitor.determine(self.model, self.history, gradient_list, self.checkgap)
            self.feature_list = self.Monitor.get_features()
            # self.issue_list = md.filtered_issue(self.issue_list)

            self.evaluated_gradients = None

            dictwriter_object = csv.DictWriter(self.feature_log_file, fieldnames=self.feature_list.keys())
            dictwriter_object.writerow(self.feature_list)
            self.feature_log_file.flush()

            if not self.retrain:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'NO training problems now. You need to train this '
                                                                  'new model more times.'))
                    self.log_file.flush()
                    print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'Found Issue Stop Training! Starting the repair '
                                                                  'procedure.'))
                    self.log_file.flush()
                    # self.log_file.close()
                    # self.model.stop_training = True
            else:
                if not self.issue_list:
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time), 'No Issue now'))
                    self.log_file.flush()
                elif self.issue_list == ['need_train']:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'NO training problems now. You need to train this '
                                                                  'new model more times.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    # print('NO training problems now. You need to train this new model more iterations.')
                else:
                    self.issue_list = list(set(self.issue_list))
                    self.log_file.write('{},{},{},{},{}\n'.format(self.checktype, epoch, self.issue_list,
                                                                  str(time.time() - self.start_time),
                                                                  'Found Issue Stop Training! Starting the repair '
                                                                  'procedure.'))
                    self.log_file.flush()
                    self.log_file.write('-------------------------------\n')
                    self.log_file.flush()
                    self.model.stop_training = False

    def on_train_end(self, logs=None):
        if self.retrain and self.issue_list == []:
            self.log_file.write('------------Solved!-----------\n')
            self.log_file.flush()

        solution_dir = os.path.join(self.save_dir, 'solution')
        if self.retrain:
            solution_dir = self.pkl_dir
        if not os.path.exists(solution_dir):
            os.makedirs(solution_dir)
        issue_path = os.path.join(solution_dir, 'issue_history.pkl')
        tmpset = {'issue_list': self.issue_list, 'history': self.history, 'feature': self.feature_list}
        with open(issue_path, 'wb') as f:
            pickle.dump(tmpset, f)
        self.log_file.close()
        self.feature_log_file.close()
        print('Finished Training')
