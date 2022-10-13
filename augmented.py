# plot augumented data

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load cyphar dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
cyphar = x_train[:6]

# flip the first 2 images vertically
cyphar[0] = np.flipud(cyphar[0])
cyphar[1] = np.flipud(cyphar[1])

# rotate image 3 and 4 of random angles
cyphar[2] = np.rot90(cyphar[2])
cyphar[3] = np.rot90(cyphar[3])

# adjust lightness of images 5 and 6
cyphar[4] = cyphar[4] * 0.5
cyphar[5] = cyphar[5] * 1.5


# plot the images
fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(cyphar[0])
ax[0, 1].imshow(cyphar[1])
ax[0, 2].imshow(cyphar[2])
ax[1, 0].imshow(cyphar[3])
ax[1, 1].imshow(cyphar[4])
ax[1, 2].imshow(cyphar[5])
plt.show()

