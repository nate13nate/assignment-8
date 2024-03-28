from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot as plt

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
from keras.datasets import fashion_mnist
import numpy as np
(x_train, _), (x_test, _) = fashion_mnist.load_data()

plt.imshow(x_test[0], cmap='gray') # test data before reconstruction
plt.show()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#introducing noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

history = autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_noisy))

prediction = autoencoder.predict(x_test[:1])[0]
prediction = prediction.reshape(28, 28) * 255
plt.imshow(prediction, cmap='gray')
plt.show()

for i in range(len(history.history['accuracy'])):
    x = history.history['accuracy'][i] * 100
    y = history.history['val_accuracy'][i] * 100
    plt.plot(x, y, 'bo')
    plt.annotate('Epoch ' + str(i), xy=[x, y])

plt.xlabel('Training Accuracy')
plt.ylabel('Validation Accuracy')
plt.show()

for i in range(len(history.history['loss'])):
    x = history.history['loss'][i] * 100
    y = history.history['val_loss'][i] * 100
    plt.plot(x, y, 'bo')
    plt.annotate('Epoch ' + str(i), xy=[x, y])

plt.xlabel('Training Loss')
plt.ylabel('Validation Loss')
plt.show()
