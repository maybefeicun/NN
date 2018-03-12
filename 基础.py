# -*- coding: utf-8 -*-
# @Time : 18-3-5 上午9:36
# @Author : XXX
# @Site : 
# @File : 基础.py
# @Software: PyCharm

import tensorflow as tf

sess = tf.Session()

#1. 初始化变量，常数，占位符（占位符一般与常数有关系）
a = tf.Variable(tf.constant(4.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

#2. 带入计算公式计算预测结果
multiplication = tf.multiply(a, x_data)

#3. 计算误差值
loss = tf.square(tf.subtract(multiplication, 50))

#4. 初始化变量并运行结果
init = tf.initialize_all_variables()
sess.run(init)

#5. 建立优化器（一般使用梯度下降的方式）
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

#6. 开始训练运行
print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data : x_val})
    a_val = sess.run(a) # 这个地方要注意下
    mult_output = sess.run(multiplication, feed_dict={x_data: x_val})
    writer = tf.summary.FileWriter('./nn_1', sess.graph)
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
writer.close()