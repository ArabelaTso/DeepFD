import pickle


def pack_train_config(opt, loss, dataset, epoch, batch_size, lr, callbacks=[]):
    # dataset: MNIST, CIFAR-10, Reuters, Circle, Blob, IMDB, "customized"
    config = {}
    config['optimizer'] = opt
    config['loss'] = loss
    config['dataset'] = dataset
    config['epoch'] = epoch
    config['batchsize'] = batch_size
    config['callbacks'] = callbacks
    config['opt_kwargs'] = {"lr": lr}
    return config


def save_config(config, save_path):
    # `config.pkl`
    with open(save_path, 'wb') as f:
        pickle.dump(config, f)


def show_pkl_file(file_path):
    f = open(file_path, 'rb')
    data = pickle.load(f)
    print(data)
    return data
