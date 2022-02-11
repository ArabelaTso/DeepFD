from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np


model = Sequential()  # two layers
model.add(Dense(input_dim=2, output_dim=4, init="glorot_uniform"))
model.add(Activation("sigmoid"))
model.add(Dense(input_dim=4, output_dim=1, init="glorot_uniform"))
model.add(Activation("sigmoid"))

# change l2 to clipnorm due to updated keras version
# sgd = SGD(l2=0.0, learning_rate=0.05, decay=1e-6, momentum=0.11, nesterov=True)
sgd = SGD(clipnorm=0.0, learning_rate=0.05, decay=1e-6, momentum=0.11, nesterov=True)
# fix 1 loss = mean_absolute_error -> mse, fix 2 optimizer=sgd -> adam
model.compile(loss='mse', optimizer='adam')

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
# fix 3 epoch 1,000 -> 50,000
model.fit(train_data, label, nb_epoch=50000, batch_size=4, verbose=1, shuffle=True)
list_test = [0, 1]
test = np.array((list_test, list1))
classes = model.predict(test)

print(classes)
