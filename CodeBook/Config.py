OPERATORS = ["mean", "std", "skew", "median", "var", "sem", "max", "min"]
HEGE  = {"MNIST": 0.9689, "CIFAR-10": 0.6348, "Blob": 0.7627, "Circle": 0.8143, "Reuters": 0.5175, "IMDB": 0.8454}  #
FAULT_TYPE = ['opt', 'lr', 'act', 'loss', 'epoch']
LABEL_DICT = {'lb_opt': 0, 'lb_lr': 0, 'lb_act': 0, 'lb_loss': 0, 'lb_epoch': 0}
LOSS_POOL = {
    "prob": {"categorical_crossentropy", "sparse_categorical_crossentropy", "binary_crossentropy"},
    "regression": {"mean_absolute_error", "mean_absolute_percentage_error", "mean_squared_error"}
}

ACT = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "linear"]

OPT = ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad"]

SEVERITY_DICT = {"LOW": 0.2, "HIGH": 0.6}
params = {'beta_1': 1e-3,
          'beta_2': 1e-4,
          'beta_3': 70,
          'gamma': 0.7,
          'zeta': 0.03,
          'eta': 0.2,
          'delta': 0.01,
          'alpha_1': 0,
          'alpha_2': 0,
          'alpha_3': 0,
          'Theta': 0.6
          }