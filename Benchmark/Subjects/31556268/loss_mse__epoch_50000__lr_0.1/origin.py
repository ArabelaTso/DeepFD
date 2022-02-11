from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np
# from Utils.utils import save_pkl, pack_train_config
from sklearn.model_selection import train_test_split


model = Sequential()  # two layers
model.add(Dense(input_dim=2, output_dim=4, init="glorot_uniform"))
model.add(Activation("sigmoid"))
model.add(Dense(input_dim=4, output_dim=1, init="glorot_uniform"))
model.add(Activation("sigmoid"))

# change l2 to clipnorm due to updated keras version
# sgd = SGD(l2=0.0, learning_rate=0.05, decay=1e-6, momentum=0.11, nesterov=True)
sgd = SGD(clipnorm=0.0, learning_rate=0.05, decay=1e-6, momentum=0.11, nesterov=True)

model.compile(loss='mean_absolute_error', optimizer=sgd)

print("begin to train")
list1 = [1, 1]
label1 = [0]
list2 = [1, 0]
label2 = [1]
list3 = [0, 0]
label3 = [0]
list4 = [0, 1]
label4 = [1]
train_data = np.array((list1, list2, list3, list4))  # four samples for epoch = 1000
label = np.array((label1, label2, label3, label4))
X_train, X_test, y_train, y_test = train_test_split(train_data, label, test_size=0.2)

model.fit(X_train, y_train, nb_epoch=1000, batch_size=4, verbose=1, shuffle=True)
list_test = [0, 1]
test = np.array((list_test, list1))
classes = model.predict(test)

print(classes)


# ########################################################
# record necessary files
# cus_dataset = {}
# cus_dataset['x'] = X_train
# cus_dataset['y'] = y_train
# cus_dataset['x_val'] = X_test
# cus_dataset['y_val'] = y_test

# model.save("./model.h5")
# config = pack_train_config(opt='sgd', loss='mean_absolute_error', dataset='customized', epoch=1000, batch_size=4,
#                            lr=0.01,
#                            callbacks=[])
# save_pkl(config, "./config.pkl")
# save_pkl(cus_dataset, "./dataset.pkl")
