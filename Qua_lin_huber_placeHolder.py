import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils


DATA_FILE = "data/birth_life_2010.txt"
data, n_samples = utils.read_birth_life_data(DATA_FILE)

X = tf.placeholder(tf.float32, name = 'x')
Y = tf.placeholder(tf.float32, name = 'y')

wl = tf.get_variable('weights',initializer = tf.constant(0.0))
bl = tf.get_variable('bias', initializer = tf.constant(0.0))

wq = tf.get_variable('weights_1', initializer = tf.constant(0.0))
uq = tf.get_variable('weights_2', initializer = tf.constant(0.0))
bq = tf.get_variable('biasq', initializer = tf.constant(0.0))

wlh = tf.get_variable('weightsh', initializer = tf.constant(0.0))
blh = tf.get_variable('biash', initializer = tf.constant(0.0))


Y_q = wq * X * X + uq * X + bq
Y_l = wl * X + bl
Y_lh = wlh * X + blh

loss_q = tf.square(Y - Y_q, name = 'loss_q')
loss_l = tf.square(Y - Y_l, name = 'loss_l')
loss_lh = utils.huber_loss(Y, Y_lh)

optimizer_q = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss_q)
optimizer_l = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss_l)
optimizer_lh = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss_lh)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for x,y in data:
            sess.run(optimizer_q, feed_dict = {X:x, Y:y})
            sess.run(optimizer_l, feed_dict = {X:x, Y:y})
            sess.run(optimizer_lh, feed_dict = {X:x, Y:y})
    wl_out, bl_out = sess.run([wl,bl])
    wq_out, uq_out, bq_out = sess.run([wq, uq, bq])
    wlh_out, blh_out = sess.run([wlh, blh])
       
plt.plot(data[:,0], data[:,1],'bo',label = 'Real data')
plt.plot(data[:,0], wl_out * data[:,0] + bl_out,'r', label = 'Linearly predicted label')
plt.plot(data[:,0], wlh_out * data[:,0] + blh_out,'k', label = 'Huber Linearly predicted label')
plt.plot(data[:,0], wq_out * data[:,0] * data[:,0] + uq_out * data[:, 0] + bq_out, 'g+', label = 'Qudra predicted label')

plt.legend()
plt.show()


