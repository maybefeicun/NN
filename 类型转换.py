# -*- coding: utf-8 -*-
# @Time : 18-3-6 下午8:32
# @Author : chen
# @Site : 
# @File : 类型转换.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np

sess = tf.Session()

a = np.random.normal(size=10)
print(a)
a = tf.cast(a, tf.float32)
print(sess.run(a))