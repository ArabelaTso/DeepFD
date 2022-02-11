from keras.models import load_model


def load(file):
    return load_model(file)


if __name__ == '__main__':
    model = load("../CasesRepaired/lenet_seed.h5")
    model.summary()

