import tensorflow as tf
import matplotlib.pyplot as plt
import utils

DATA_FILE = "data/birth_life_2010.txt"
data, n_samples = utils.read_birth_life_data(DATA_FILE) # data is numpy array

dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
#iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

w = tf.get_variable('weights', initializer = tf.constant(0.0))
b = tf.get_variable('bias', initializer = tf.constant(0.0))

wh = tf.get_variable('weights_h', initializer = tf.constant(0.0))
bh = tf.get_variable('bias_h', initializer = tf.constant(0.0))

Y_predict = w *X + b
Y_predicth = wh * X + bh

loss = tf.square(Y - Y_predict, name = 'loss')
loss_h = utils.huber_loss(Y, Y_predicth)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
optimizerh = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss_h)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(iterator.initializer)
    
        try:
            while True:
               sess.run([optimizer, optimizerh])
                   
        except tf.errors.OutOfRangeError:
            pass
      
    w_out, b_out, wh_out, bh_out = sess.run([w, b, wh, bh])
    print(w_out)
    print(b_out)
    print(wh_out)
    print(bh_out)
    

# print the results

plt.plot(data[:,0], data[:,1],'bo', label = 'real data')
plt.plot(data[:,0], w_out * data[:,0] + b_out, 'r', label='square loss')
plt.plot(data[:,0], wh_out * data[:,0] + bh_out, 'k', label = 'huber loss')
plt.legend()
plt.show()
    
        
        

