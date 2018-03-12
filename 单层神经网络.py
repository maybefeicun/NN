# -*- coding: utf-8 -*-
# @Time : 18-3-6 上午9:36
# @Author : chen
# @Site : 
# @File : 单层神经网络.py
# @Software: PyCharm

"""
用来实现单个隐藏层的神经网络，在这里使用的数据集为iris
"""

import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import time

ops.reset_default_graph()
# 需要将 default graph 重新初始化，以保证内存中没有其他的 Graph

sess = tf.Session()
summary_write = tf.summary.FileWriter('tensorboard', tf.get_default_graph())
# merged_summary_op = tf.merge_all_summaries() # 这个要注意是用来画图处理的


# 1. 处理数据集
iris = datasets.load_iris() # 获取数据集
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# 2. 定义测试机与训练集(按8-2划分),同时通过min-max放缩法正则化x特征值为0-1的数值
"""
以下两行代码需要记住，基本使用这样的代码
"""
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

"""
min-max放缩法，这里我们需要注意这段代码
"""
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train)) # nan_to_num的方法需要注意
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# 3. 定义变量以及占位符
batch_size = 50
x_data = tf.placeholder(dtype=tf.float32, shape=[None, 3])
y_target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

"""
这里十分重要，定义了隐藏层的层数
以及每层网络的各个参数值
本个实验中的A1,A2,b1,b2的shape的定义原因在于本实验有三个输入值，五个隐藏节点，一个输出值
"""
hidden_layer_nodes = 10 # 从实验的结果来看隐藏层层数的设置很重要，如果设置为５结果会出现很大的波动
A1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))

"""
实现公式以及实现随时函数
"""
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1)) # 注意这里应该是写成matmul而不是mulply
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))
# loss = tf.reduce_mean(tf.square(tf.subtract(final_output, y_target)))
loss = tf.reduce_mean(tf.square(y_target - final_output))

my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# 将数据写入到计算图中
# with tf.name_scope('parameter'):
#     tf.summary.scalar('A1', A1)
#     tf.summary.scalar('A2', A2)
#     tf.summary.scalar('b1', b1)
#     tf.summary.scalar('b2', b2)

with tf.name_scope('loss'):
    tf.summary.scalar('loss', loss)

summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()
sess.run(init)

# 4. 训练过程
loss_vec = []
test_loss = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size) # http://blog.csdn.net/IAMoldpan/article/details/78707140
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data : rand_x, y_target : rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data : rand_x, y_target : rand_y})
    loss_vec.append(np.square(temp_loss))

    summary = sess.run(summary_op, feed_dict={x_data : rand_x, y_target : rand_y})
    """
    运行测试集
    """
    test_temp_loss = sess.run(loss, feed_dict={x_data : x_vals_test, y_target : np.transpose([y_vals_test])})
    # 在这犯了一个错误，应该是写成[...]但最后没写[]，这样就会导致在矩阵转换时出现错误
    test_loss.append(test_temp_loss)

    # 绘制计算图
    write = tf.summary.FileWriter("tensorboard")
    # merged_summary_op = tf.summary.merge_all()  # 上面的写法已经被舍去了
    # loss_str = sess.run(merged_summary_op)
    write.add_summary(summary, (i+1))
    time.sleep(0.2)

    if (i + 1) % 50 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))
write.close()

plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
