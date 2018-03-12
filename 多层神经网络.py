# -*- coding: utf-8 -*-
# @Time : 18-3-6 下午7:13
# @Author : chen
# @Site : 
# @File : 多层神经网络.py
# @Software: PyCharm

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import requests
from tensorflow.python.framework import ops

birth_weight_file = 'birth_weight.csv' # 数据存储的地址
# 1. 获取数据集
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = birth_data[0].split('\t')
    birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
    with open(birth_weight_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(birth_data)
        f.close()

# 从文件中读取数据
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        birth_data.append(row)

# 2. 处理数据集
birth_data = [[float(x) for x in row] for row in birth_data]
# birth_data = tf.cast(birth_data, tf.float32)
# Extract y-target (birth weight)

# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI']
# 设置输入值
x_vals = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])
# 设置真实输出值
y_vals = np.array([x[8] for x in birth_data])
"""
以上都是对数据集的处理
"""

ops.reset_default_graph()
sess = tf.Session()

# 3. 为了后面可以复现我们设置随机种子
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

batch_size = 100

# 4. 分割数据集，按8-2进行分割
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

# 样本数据集是以０到１为中心的，他将有利于激励函数操作的收敛
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# 5. 因为要许多随机数，所以写成函数这样就更加便捷快速
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (weight)

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return (bias)

# 6. 设置占位符
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 7. 设置全连接层,这个因为在隐藏层中使用了三次所以直接设置为一个函数即可
def fully_connected(input_layer, weights, biases):
    layer_output = tf.add(tf.matmul(input_layer, weights), biases)
    return (tf.nn.relu(layer_output))

# 8. 创建隐藏层，总共三层隐藏层,三个隐藏层的大小为25, 10, 3
"""
输入层与第一个隐藏层的weights=7*25,biases=25*1
第一个隐藏层与第二个隐藏层为weights=25*8,biases=10*1
第二个隐藏层与第三个隐藏层为weights=10*3,biases=3*1
第三个隐藏层与输出层的weights=3*1,biases=1*1
"""

# def to_float32(vals):
#     return tf.cast(vals, tf.float32)

# 第一个
weights_1 = init_weight(shape=[7, 25], st_dev=10.0)
bias_1 = init_bias(shape=[25], st_dev=10.0)
layer_1 = fully_connected(x_data, weights_1, bias_1)

# 第二个
weights_2 = init_weight(shape=[25, 10], st_dev=10.0)
bias_2 = init_bias(shape=[10], st_dev=10.0)
layer_2 = fully_connected(layer_1, weights_2, bias_2)

# 第三个
weights_3 = init_weight(shape=[10, 3], st_dev=10.0)
bias_3 = init_bias(shape=[3], st_dev=10.0)
layer_3 = fully_connected(layer_2, weights_3, bias_3)

# 第四个
weights_4 = init_weight(shape=[3, 1], st_dev=10.0)
bias_4 = init_bias(shape=[1], st_dev=10.0)
final_output = fully_connected(layer_3, weights_4, bias_4)

# 9. 构建随时函数与优化器，这个使用的是ＬＩ范数，以及另一个优化器
loss = tf.reduce_mean(tf.abs(y_target - final_output)) # abs()里面只有一个参数
my_opt = tf.train.AdamOptimizer(0.05) # 这里一开始写错了，真是服了，看了一百遍了
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# 10. 训练
loss_vec = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    sess.run(train_step, feed_dict={x_data : rand_x, y_target : rand_y})
    loss_temp = sess.run(loss, feed_dict={x_data : rand_x, y_target : rand_y})
    loss_vec.append(loss_temp)

    # 测试过程
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    if (i + 1) % 25 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(loss_temp))

plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

#