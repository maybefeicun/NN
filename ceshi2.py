# -*- coding: utf-8 -*-
# @Time : 18-3-5 下午8:07
# @Author : chen
# @Site : 
# @File : ceshi2.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np

sess = tf.Session()

a = tf.Variable(tf.constant([[1.]]))
b = [1., 2.]
b = np.reshape(b, [2, 1])
b_data = tf.placeholder(dtype=tf.float32, shape=[2, 1])
c = tf.Variable(tf.constant([[1.]]))
result = tf.add(tf.multiply(b_data, a), c)

init = tf.initialize_all_variables()
sess.run(init)

print(sess.run(result, feed_dict={b_data : b}))