# -*- coding: utf-8 -*-
# @Time : 18-3-5 上午10:44
# @Author : XXX
# @Site : 
# @File : 基础_2.py
# @Software: PyCharm

import tensorflow as tf

sess = tf.Session()

# 1.初始化要用的参数
a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

# 2.实现公式
prediction = tf.add(tf.multiply(a, x_data), b)
loss = tf.square(tf.subtract(prediction, 50))

init = tf.initialize_all_variables()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

for i in range(20):
    sess.run(train_step, feed_dict={x_data : x_val})
    a_val = sess.run(a)
    b_val = sess.run(b)
    prediction_val = sess.run(prediction, feed_dict={x_data : x_val})
    write = tf.summary.FileWriter("./nn_2", sess.graph)
    print(str(a_val) + " * " + str(x_val) + " + " + str(b_val) + " = " + str(prediction_val))
write.close()