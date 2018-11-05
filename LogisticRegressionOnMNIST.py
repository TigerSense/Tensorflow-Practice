import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import utils


mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten = True)

train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = tf.data.shuffle(1000)
val_data = tf.data.Dataset.from_tensor_slices(val)
test_data = tf.data.Dataset.from_tensor_slices(test)


w = tf.get_variable('weights', shape = (784, 10), initializer = tf.random_normal_initializer(0, 0.01))
b = tf.get_variable('bias', shape = (1,10), initializer = tf.zeros_initializer())

iterator = train_data.make_initializable_iterator();
X, Y = iterator.get_next();

class_scores = w * X + b;
loss = 
