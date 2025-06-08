import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x_train=np.linspace(-3.0, 3.0, 256)
np.random.shuffle(x_train)
y_train=(x_train**2-2*x_train+3)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(1,)))
model.add(tf.keras.layers.Dense(5, activation='relu', bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.1)))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['mean_absolute_error'])

model.fit(x_train,
          y_train,
          batch_size=16,
          epochs=1000,
          verbose=1)

x_test=np.linspace(-3.333, 3.333, 1024)
y_true=(x_test**2-2*x_test+3)
y_test=model.predict(x_test)

print(x_test.shape)
# plot
fig, ax = plt.subplots()
ax.plot(x_test, y_true, color='red')
ax.plot(x_test, y_test)
#ax.plot(x, yd1)
#ax.plot(p0[0], p0[1], 'x', markeredgewidth=2, color='red')

#ax1.set(xlim=(-5, 5))
#ax2.set(xlim=(-5, 5))
       #ylim=(0, 8)
ax.grid(visible=True)

plt.show()

model.summary()
print('--- 1st Layer ---------------------------')
print(model.get_layer('dense').get_weights())
print('--- 2nd Layer ---------------------------')
print(model.get_layer('dense_1').get_weights())