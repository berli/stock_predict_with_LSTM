#!/usr/bin/python
# -*- coding: utf8 -*-)

import tensorflow as tf

tf.enable_eager_execution()

lstm_num_units = 10
batch_size = 4
time_step = 3

input = tf.random_normal(shape=[time_step, batch_size, lstm_num_units], dtype=tf.float32)
cells = [
        tf.keras.layers.LSTMCell( lstm_num_units ),
        tf.keras.layers.LSTMCell( lstm_num_units ),
        tf.keras.layers.LSTMCell( lstm_num_units ),
]

#time_major = True
#cell: BasicLSTMCell，BasicRNNCell，GRUCell 的对象实例，自己定义的cell 内容
#inputs 如果是time_major=True，input的维度是[max_time, batch_size, input_size]，反之就是[batch_size,max_time, input_zise]，time_major 默认是False
#output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state= init_state, dtype=tf.float32, time_major = True)
inputs = tf.keras.Input((time_step, lstm_num_units))
print inputs
output = tf.keras.layers.RNN(cells)(inputs)
x = tf.layers.Dense(64, activation='relu')(inputs)
predictions = tf.layers.Dense(10, activation='softmax')(x)

# Instantiate the model given inputs and outputs.

model = tf.keras.compile(loss = 'mse', optimizer='adam')
model = tf.keras.Model(inputs=inputs, outputs=predictions)
model = tf.keras.fit(x = input, y = input)
print predictions
