import tensorflow as tf
import utils

DATA_FILE = "data/birth_life_2010.txt"
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Y_predicted = W*X+b

X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32, name = 'Y')

w = tf.get_variable('weight', initializer = tf.constant(0.0))
b = tf.get_variable('bias', initializer = tf.constant(0.0))

Y_predicted = w * X + b

loss = tf.square(Y - Y_predicted, name = 'loss')

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for x, y in data:
            sess.run(optimizer, feed_dict = {X : x, Y : y})

    w_out, b_out = sess.run([w,b])
    print(w_out)
    print(b_out)
    
