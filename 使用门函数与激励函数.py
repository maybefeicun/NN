# -*- coding: utf-8 -*-
# @Time : 18-3-5 上午11:15
# @Author : XXX
# @Site : 
# @File : 使用门函数与激励函数.py
# @Software: PyCharm
"""
难点我觉得在于处理[None, 1]*[1, 1] + [1, 1]的理解，可以看看ceshi2.py的运行结果
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
tf.set_random_seed(5)
np.random.seed(42)

# 1.初始化参数
batch = 50
a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
b1 = tf.Variable(tf.random_normal(shape=[1, 1]))
a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1, 1]))
x = np.random.normal(2, 0.1, 200) # 参数含义最后一个为生成的随机数个数
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 2.引入激活函数
sigmoid_activation = tf.sigmoid(tf.add(tf.multiply(x_data, a1), b1))
relu_activation = tf.nn.relu(tf.add(tf.multiply(x_data, a2), b2))

# 3.建立损失函数
loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

# 4.定义优化器
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step_sigmoid = my_opt.minimize(loss1)
train_step_relu = my_opt.minimize(loss2)
init = tf.initialize_all_variables()
sess.run(init)

# 5.开始训练,保存损失函数和激励函数的返回值用来画图，这个理解理解便可
loss_vec_sigmoid = []
loss_vec_relu = []
activation_sigmoid = []
activation_relu = []

for i in range(750):
    """
    np.random.choice()的方法应该熟悉
    tf.transpose()的用法
    """
    rand_indices = np.random.choice(len(x), size=batch)
    # x_vals = tf.transpose(x[rand_indices]) 不能这样写，因为会导致x_vals与x_data的数据类型不一样
    x_vals = np.transpose([x[rand_indices]]) # 这里不能写成tf.transpose因为括号里面的数据并不是张量
    sess.run(train_step_sigmoid, feed_dict={x_data : x_vals})
    sess.run(train_step_relu, feed_dict={x_data : x_vals})

    a1_val = sess.run(a1)
    a2_val = sess.run(a2)
    b1_val = sess.run(b1)
    b2_val = sess.run(b2)

    loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals}))
    loss_vec_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))

    sigmoid_output = np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals}))
    relu_output = np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals}))

    write = tf.summary.FileWriter("./nn_3", sess.graph)

    print("a1 = " + str(a1_val) + "\t" + "b1 = " + str(b1_val))
    print("a2 = " + str(a2_val) + "\t" + "b2 = " + str(b2_val))
    if i % 50 == 0:
        print('sigmoid = ' + str(np.mean(sigmoid_output)) + ' relu = ' + str(np.mean(relu_output)))
write.close()
# print(sess.run(a1))
# 6.画图
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(loss_vec_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()