#!/usr/bin/python
# -*- coding: utf8 -*-)

import tensorflow as tf

lstm_num_units = 10
batch_size = 4
time_step = 3

input = tf.random_normal(shape=[time_step, batch_size, lstm_num_units], dtype=tf.float32)

cell  = tf.nn.rnn_cell.BasicLSTMCell( lstm_num_units , forget_bias = 1.0, state_is_tuple = True)
init_state = cell.zero_state(batch_size, dtype = tf.float32)

#time_major = True
#cell: BasicLSTMCell，BasicRNNCell，GRUCell 的对象实例，自己定义的cell 内容
#inputs 如果是time_major=True，input的维度是[max_time, batch_size, input_size]，反之就是[batch_size,max_time, input_zise]，time_major 默认是False
output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state= init_state, dtype=tf.float32, time_major = True)

#time_major = False
inputs2  =  tf.reshape(input, [-1, time_step, lstm_num_units] )

output2, final_state2 = tf.nn.dynamic_rnn(cell, inputs2, initial_state= init_state, dtype=tf.float32, time_major = False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    print '-----------------time_major = True------------------'
    print 'output shape = ', sess.run(tf.shape(output))
    print 'output =',sess.run(output)
    print '---------------------------'
    print 'final_state shape = ', sess.run(tf.shape(final_state))
    print 'final_state = ', sess.run(final_state)
    
    print '-----------------time_major = False------------------'
    print 'inputs2 shape = ', sess.run(tf.shape(inputs2))
    print 'output2 =',sess.run(output2)
    print '---------------------------'
    print 'final_state2 shape = ', sess.run(tf.shape(final_state2))
    print 'final_state = ', sess.run(final_state2)

tf.enable_eager_execution()
cell  = tf.keras.layers.LSTMCell( lstm_num_units )

#time_major = True
#cell: BasicLSTMCell，BasicRNNCell，GRUCell 的对象实例，自己定义的cell 内容
#inputs 如果是time_major=True，input的维度是[max_time, batch_size, input_size]，反之就是[batch_size,max_time, input_zise]，time_major 默认是False
#output, final_state = tf.nn.dynamic_rnn(cell, input, initial_state= init_state, dtype=tf.float32, time_major = True)
output = tf.keras.layers.RNN(cell)

#time_major = False
inputs2  =  tf.reshape(input, [-1, time_step, lstm_num_units] )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    print '-----------------time_major = True------------------'
    print 'output shape = ', sess.run(tf.shape(output))
    print 'output =',sess.run(output)
    print '---------------------------'
    print 'final_state shape = ', sess.run(tf.shape(final_state))
    print 'final_state = ', sess.run(final_state)
    
    print '-----------------time_major = False------------------'
    print 'inputs2 shape = ', sess.run(tf.shape(inputs2))
    print 'output2 =',sess.run(output2)
    print '---------------------------'
    print 'final_state2 shape = ', sess.run(tf.shape(final_state2))
    print 'final_state = ', sess.run(final_state2)
