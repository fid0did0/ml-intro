import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def to_one_hot(labels, num_classes=10):
    labels = np.asarray(labels)
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1.0
    return one_hot

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)

# Filter for digits 0 and 1 in training set
train_filter = (y_train == 0) | (y_train == 1)
x_train_01 = x_train[train_filter].astype('float32')/256.0
y_train_01 = y_train[train_filter].astype('int')
y_train_01 = to_one_hot(y_train_01, num_classes=2)
print(x_train_01.shape)
print(y_train_01.shape)

test_filter = (y_test == 0) | (y_test == 1)
x_test_01 = x_test[test_filter].astype('float32')/256.0
y_test_01 = y_test[test_filter].astype('int')
y_test_01 = to_one_hot(y_test_01, num_classes=2)
print(y_test_01.shape)

# define cnn model
def define_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

conv_model=define_model()
#conv_model.summary()

learning_history=conv_model.fit(x_train_01, y_train_01, 
                  batch_size=32,
                  validation_split=0.02,
                  epochs=50)

loss=learning_history.history['loss']
epoch=range(len(loss))
val_acc=learning_history.history['val_accuracy']

conv_model.summary()

#plt.plot(epoch, loss, epoch, val_acc, 'r')
plt.semilogy(epoch, loss, epoch, val_acc, 'r')
plt.ylabel('loss / val_acc')
plt.show()

conv_model.save('zero_one_oh_classifier.keras')

