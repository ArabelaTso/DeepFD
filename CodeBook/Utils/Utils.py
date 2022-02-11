import os
import sys

sys.path.append('.')
import matplotlib.pyplot as plt
import csv
import numpy as np
import keras
from CodeBook.Utils.TimeCounter import TimeHistory
from keras.models import load_model
from CodeBook.Callbacks.LossHistory import LossHistory
from CodeBook.Utils.Logger import Logger

logger = Logger()
from keras.callbacks import ModelCheckpoint
import pickle
import time
import CodeBook.Utils.Module as md


# default_param = {'beta_1': 1e-3,
#                  'beta_2': 1e-4,
#                  'beta_3': 70,
#                  'gamma': 0.7,
#                  'zeta': 0.03,
#
#                  'eta': 0.2,
#                  'delta': 0.01,
#                  'alpha_1': 0,
#                  'alpha_2': 0,
#                  'alpha_3': 0,
#                  'Theta': 0.7
#                  }

def save_model(model, path):
    """[summary]

    Args:
        model ([model]): [a model you want to save]
        path ([str]): [the path you want to save the model]
    """
    try:
        model.save(path, model)
        model.summary()
        logger.info('Saved model!')
    except:
        logger.error(sys.exc_info())


def has_NaN(output):
    output = np.array(output)
    result = (np.isnan(output).any() or np.isinf(output).any())
    return result


def generate_fig(array_dic, path, method=2):
    """
    :params array_dic: a dictionary contains multi-arrays, was used to be the data of the figure
    :params path: a string, the path you want to save the fig
    :params method: int method. 1 means only one figure, 2 means 121,122 subplot
    """
    if method == 1:
        plt.figure(figsize=(9, 6))
        plt.subplot(121)
        a = []
        for key, value in array_dic.items():
            a.append(value)
        assert len(a) == 2
        plt.plot(a[0], label='train')
        plt.plot(a[1], label='test')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('indicator', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
    if method == 2:
        plt.figure(figsize=(16, 6))
        plt.subplot(121)
        plt.plot(array_dic['acc'], label='train')
        plt.plot(array_dic['val_acc'], label='test')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('accuracy', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        plt.subplot(122)
        plt.plot(array_dic['loss'], label='train')
        plt.plot(array_dic['val_loss'], label='test')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    plt.savefig(path, dpi=300)


def read_csv(csv_path, epoch):
    csvFile = open(csv_path, 'r')
    reader = csv.reader(csvFile)
    result = []
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        item = [float(x) for x in item]
        result.append(item)
    csvFile.close()
    x_axis = []
    with open(csv_path, 'r') as f:
        tmp = len(f.readlines()) - 1
        if tmp <= epoch:
            epoch = tmp
    for i in range(epoch):
        x_axis.append(i)
    while ([] in result):
        result.remove([])
    result = np.array(result)
    tmp_dic = {}
    tmp_dic['acc'] = result[:, 3]
    tmp_dic['val_acc'] = result[:, 1]
    tmp_dic['loss'] = result[:, 2]
    tmp_dic['val_loss'] = result[:, 0]
    return tmp_dic


def check_point_model(model_cur, model_dir, model_name, config, history):
    if os.path.exists(model_name):
        model = load_model(model_name)
        os.remove(model_name)
    else:
        model = model_cur
    if "val_accuracy" not in history.history:
        test_acc = 0.0
    else:
        test_acc = max(history.history['val_accuracy'])
    model_path = os.path.join(model_dir, 'best_model_{}.h5'.format(test_acc))
    model.save(model_path)


def pack_train_config(opt, loss, dataset, epoch, batch_size, callbacks):
    config = {}
    config['opt'] = opt
    config['loss'] = loss
    config['data'] = dataset
    config['epoch'] = epoch
    config['batch_size'] = batch_size
    config['callbacks'] = callbacks
    return config


def model_train(model,
                train_config_set,
                optimizer,
                loss,
                dataset,
                iters,
                batch_size,
                log_dir,
                callbacks,
                root_path,
                new_issue_dir,
                verb=0,
                determine_threshold=1,
                save_dir='./tool_log',
                checktype='epoch_3',
                autorepair=True,
                modification_sufferance=3,  # 0-3 for model
                memory_limit=False,
                satisfied_acc=0.7,
                strategy='balance',
                params={}
                ):
    """[summary]
    Args:
        model ([model loaded by keras or str]): [a model you want to train or a model path(string)]
        train_config_set ([dict]): [a dict with all training configurations, using as a backup]
        optimizer ([str]): [the optimizer you want to use]
        loss ([str]): [usually 'categorical_crossentropy' or 'binary_crossentropy']
        dataset ([dic]): [a dictionary which contains 'x''y''x_val''y_val']
        iters ([int]): [max iterations in training]
        batch_size ([int]): [batch_size in training]
        log_dir ([str]): [the directory you want to save the training log (a csv file)]
        callbacks ([list]): [a list of the callbacks you want to use in the training. e.g., tensorboard , reducelr, earlystop]
        root_path ([str]): [the directory you want to save the result of each solution (a csv file)]
        new_issue_dir ([str]): [the directory you want to save the model with new training problems after repaired the existing one]
        verb (int, optional): [model.fit, verbose]. Defaults to 0.
        determine_threshold(int, optional): [the alpha value in training, not be used now, will be removed later]. Defaults to 1.
        save_dir (str, optional): [the dir you want to save all result(include the training report, trained model with each solution)].\
            Defaults to './tool_log'.
        checktype (str, optional): ['epoch_xx', xx is a number, it relates to the problem checker interval]. Defaults to 'epoch_3'.
        autorepair (bool, optional): [whether the user want our tools to auto repair or not, if not our tools will return the problems \
            and corresponding solutions, if yes, will return trained model and description and logs ]. Defaults to True.
        modification_sufferance (int, optional): [The sufferance to the model modification of the solutions. The greater it is, \
            the more the solution can modify the model]. Defaults to 3.
        memory_limit (bool, optional): [The memory limitation of the solutions. While True, some solutions which requires greater\
            memory will be disabled]. Defaults to False.
        satisfied_acc(float,optional):[the satisfied accuracy in training, not be used now, will be removed later] Default to be 0.7.
        strategy (str, optional): [chosen from ['balance','efficient','structure',it will determine the solution order when solving the problem ]]. Defaults to 'balance'.
        params (dict, optional): [the configurable parameters dict.]. Default to be {}

    Returns:
        [type]: [if autorepair is True, return a trained model and the log/description file path.\
            if autorepair is False, only return the problems and the corresponding solution description]
    """
    save_dir = os.path.abspath(save_dir)
    log_dir = os.path.abspath(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if isinstance(model, str):
        model_path = model
        model = load_model(model_path)
    # K.clear_session()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    callbacks = [n for n in callbacks if (
            n.__class__ != LossHistory and n.__class__ != ModelCheckpoint)]

    # add earlystopping and reduceLR callbacks if necessary
    if 'estop' in callbacks:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
                                                       verbose=0, mode='auto', baseline=None,
                                                       restore_best_weights=False))
        callbacks.remove('estop')
    if 'ReduceLR' in callbacks:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                           patience=5, min_lr=0.001))
        callbacks.remove('ReduceLR')

    # add two callbacks
    checkpoint_name = "train_best.h5"
    checkpoint_dir = os.path.join(save_dir, 'checkpoint_model')
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks.append(ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'))
    callbacks.append(
        LossHistory(training_data=[dataset['x'], dataset['y']], model=model, determine_threshold=determine_threshold,
                    batch_size=batch_size, save_dir=save_dir, total_epoch=iters, satisfied_acc=satisfied_acc,
                    checktype=checktype, params=params))  # issue in lstm network

    callbacks_new = list(set(callbacks))

    # fit model
    history = model.fit(dataset['x'], dataset['y'], batch_size=batch_size, validation_data=(
        dataset['x_val'], dataset['y_val']), epochs=iters, verbose=verb, callbacks=callbacks_new)
    check_point_model(model, checkpoint_dir, checkpoint_path, dataset, history)

    result = history.history
    time_callback = TimeHistory()
    log_path = os.path.join(log_dir, 'log.csv')
    if 'val_loss' in result.keys():
        time_callback.write_to_csv(result, log_path, iters)

    solution_dir = os.path.join(save_dir, 'solution')
    issue_path = os.path.join(solution_dir, 'issue_history.pkl')
    with open(issue_path, 'rb') as f:  # input,bug type,params
        output = pickle.load(f)
    issues = output['issue_list']

    trained_path = ""

    if issues:
        if autorepair:
            # auto repair
            train_config = pack_train_config(optimizer, loss, dataset, iters,
                                             batch_size, callbacks)
            start_time = time.time()
            rm = md.Repair_Module(
                model=model,
                training_config=train_config,
                issue_list=issues,
                sufferance=modification_sufferance,
                memory=memory_limit,
                satisfied_acc=satisfied_acc,
                checktype=checktype,
                determine_threshold=determine_threshold,
                config_set=train_config_set,
                root_path=root_path
            )  # train_config need to be packed and issue need to be read.
            result, model, trained_path, test_acc, history, issue_list, now_issue = rm.solve(solution_dir,
                                                                                             new_issue_dir=new_issue_dir)
            tmpset = {}
            tmpset['time'] = time.time() - start_time
            tmpset['test_acc'] = test_acc
            tmpset['model_path'] = trained_path
            tmpset['history'] = history
            tmpset['initial_issue'] = issue_list
            tmpset['now_issue'] = now_issue
            tmppath = os.path.join(save_dir, 'repair_result_total.pkl')
            with open(tmppath, 'wb') as f:
                pickle.dump(tmpset, f)
        else:
            print('You can find the description of the solution candidates in {}'.format('./path'))
    return result, model, trained_path


def model_retrain(model,
                  config,
                  satisfied_acc,
                  save_dir,
                  retrain_dir,
                  verb=1,
                  solution=None,
                  determine_threshold=5,
                  checktype='epoch_3'
                  ):
    retrain_dir = os.path.abspath(retrain_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if isinstance(model, str):
        model_path = model
        model = load_model(model_path)
    model.compile(loss=config['loss'], optimizer=config['opt'], metrics=['accuracy'])
    config['callbacks'] = [n for n in config['callbacks'] if
                           (n.__class__ != LossHistory and n.__class__ != ModelCheckpoint)]
    config['callbacks'].append(
        LossHistory(training_data=[config['data']['x'], config['data']['y']],
                    model=model,
                    batch_size=config['batch_size'],
                    save_dir=retrain_dir,
                    pkl_dir=save_dir,
                    total_epoch=config['epoch'],
                    determine_threshold=determine_threshold,
                    checktype=checktype,
                    satisfied_acc=satisfied_acc,
                    retrain=True,
                    solution=solution, params={}))
    checkpoint_name = "train_best.h5"
    checkpoint_dir = os.path.join(save_dir, 'checkpoint_model')
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    config['callbacks'].append(
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'))
    callbacks_new = list(set(config['callbacks']))
    history = model.fit(config['data']['x'], config['data']['y'], batch_size=config['batch_size'],
                        validation_data=(config['data']['x_val'], config['data']['y_val']),
                        epochs=config['epoch'], verbose=verb, callbacks=callbacks_new)
    check_point_model(model, checkpoint_dir, checkpoint_path, config, history)
    issue_path = os.path.join(save_dir, 'issue_history.pkl')
    with open(issue_path, 'rb') as f:
        output = pickle.load(f)
    new_issues = output['issue_list']
    if 'need_train' in new_issues:
        new_issues = []
    test_acc = history.history['val_accuracy'][-1]
    return model, new_issues, test_acc, history.history
