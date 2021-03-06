#!/usr/bin/python3
# -*- coding: utf8 -*-)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from sklearn.metrics import mean_absolute_error,mean_squared_error
import getopt as opt

input_size = 7       #特征数量
output_size = 1
summary_total = 200

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model_path", "model","the path for saving model")
tf.app.flags.DEFINE_string("data_file", "","data file")
tf.app.flags.DEFINE_integer('iteration', 5000,'iteration for training')
tf.app.flags.DEFINE_integer('summary_total', 200,'sample count for summary on tensorboard')
tf.app.flags.DEFINE_float('keep_prob', 1, 'input keep prob')
tf.app.flags.DEFINE_integer('lstm_layer_num', 2,'lstm layer num ')
tf.app.flags.DEFINE_float('learning_rate', 0.0003, 'Initial learning rate.')
lr = FLAGS.learning_rate
tf.app.flags.DEFINE_integer('lstm_num_units', 10 ,'lstm units num ')
lstm_num_units = FLAGS.lstm_num_units #LSTM每个单元中的单元数量，用来指有多少个隐藏层单元,同时也是输出维度的多少 hidden_dim

class rnn_lstm:
    def __init__(self, data_file):
        #——————————————————导入数据——————————————————————
        if( len(data_file) == 0):
            data_file = 'stock_data.csv'
        df = pd.read_csv(data_file)     #读入股票数据
        self.data = df.iloc[:,2:10].values  #前闭后开,取第[3,10)列
        self.TRAIN_END= int(len(self.data)*0.8);
        '''
        plt.figure()
        plt.plot(data)
        plt.show()
        '''
        #——————————————————定义权重和偏置参数——————————————————
        
        #输入层、输出层, 权重、偏置
        with tf.name_scope("weights"):
            self.weights = {
                 'in':tf.Variable(tf.random_normal([input_size, lstm_num_units])),
                 'out':tf.Variable(tf.random_normal([lstm_num_units, 1]))
                }
            self.biases = {
                'in':tf.Variable(tf.constant(0.1, shape = [lstm_num_units,])),
                'out':tf.Variable(tf.constant(0.1, shape = [1,]))
               }
        
    #——————————————————获取训练集———————————————————————
    #batch_size 每批训练大小
    def get_train_data(self, batch_size = 60,time_step = 20,train_begin = 0,train_end = 0):
        if( train_end == 0 ):
            train_end = self.TRAIN_END

        batch_index = []
        data_train = self.data[train_begin:train_end]
        '''
        标准化 Z-Score方法
        np.mean(data_train,axis = 0)计算每一列的均值
        np.std(data_train,axis = 0)每列的标准差
        '''
        normalized_train_data = (data_train-np.mean(data_train,axis = 0))/np.std(data_train,axis = 0)  
        print ("normalized_train_data:\n",normalized_train_data)
    
        print ("---------------------------------------------")
        #训练集
        train_x,train_y = [],[]  
        c  =  0;
        for i in range(len(normalized_train_data)-time_step):
           if i % batch_size == 0:
               batch_index.append(i)
           
           #取time_step个时间序列数据, 取前面7列数据
           x = normalized_train_data[i:i+time_step,:input_size]
           #取time_step个时间序列数据, 第7列数据,np.newaxis增加一个维度
           y = normalized_train_data[i:i+time_step, input_size, np.newaxis]
           #变成list是干啥？？？
           train_x.append(x.tolist())
           train_y.append(y.tolist())
    
        batch_index.append((len(normalized_train_data)-time_step))
    
        return batch_index,train_x,train_y
    
    #————————————————————获取测试集——————————————————————
    def get_test_data(self, time_step = 20, train_begin = 0, train_end = 0):
        if( train_end == 0 ):
            train_end = self.TRAIN_END
        #取begin后面的数据
        test_begin = train_end
        data_test = self.data[test_begin:]
        print('len(data_test)=',len(data_test))

        #计算每一列的均值
        mean = np.mean(data_test,axis = 0)
    
        #计算每一列的标准差
        std = np.std(data_test,axis = 0)
    
        #归一化使用z-score算法
        normalized_test_data = (data_test-mean)/std 

        #padding 
        padding = time_step - len(normalized_test_data)%time_step
        normalized_test_data = np.pad(normalized_test_data, (0, padding), 'constant')
        #有size个sample
        #size = (len(normalized_test_data) + time_step)//time_step 
        size = len(normalized_test_data)//time_step 
    
        test_x,test_y = [],[]
    
        i = 0;
        for i in range(size - 1 ):
           #前面7列特征数据
           x = normalized_test_data[ i*time_step:(i+1)*time_step, :input_size]
           #第7列标签
           y = normalized_test_data[ i*time_step:(i+1)*time_step, input_size]
    
           test_x.append(x.tolist())
           #是extennd,不是append
           test_y.extend(y)
    
        print('len(test_x)=',len(test_x))
        test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
        print('len(test_x)=',len(test_x))
        test_y.extend((normalized_test_data[(i+1)*time_step:,input_size]).tolist())
    
        return mean,std,test_x,test_y, padding
    
    
    #——————————————————实现lstm神经网络——————————————————
    def lstm(self, X):
        
        with tf.name_scope("weights"):
            batch_size = tf.shape(X)[0]
            time_step = tf.shape(X)[1]
            w_in = self.weights['in']
            b_in = self.biases['in']
        
        with tf.name_scope("inputs"):
            #将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
            input  =  tf.reshape(X,[-1,input_size])  
            inputs_data  =  tf.matmul(input, w_in) + b_in
            
            #将tensor转成3维，作为lstm cell的输入
            inputs_data  =  tf.reshape(inputs_data, [-1, time_step, lstm_num_units] ) 
    
        #num_units: 输出维度,图一中ht的维数，如果num_units=10,那么ht就是10维行向量, 官方解释:int, The number of units in the LSTM cell.
        #forget_bias：遗忘门的初始化偏置,默认是1.0，都不忘记, 0:都忘记
        #activation: 内部状态的激活函数,默认是: tanh.
        #cell = tf.nn.rnn_cell.LSTMCell( num_units = lstm_num_units )
        #定义lstm cell和dropout层
        def attn_cell():
            lstm_cell = tf.keras.layers.LSTMCell( units = lstm_num_units )
            with tf.name_scope('lstm_dropout'):
                return tf.contrib.rnn.DropoutWrapper( lstm_cell, output_keep_prob = FLAGS.keep_prob)
                #return tf.keras.layers.Dropout( lstm_cell, output_keep_prob = FLAGS.keep_prob)
    
        enc_cells = []
        for i in range(FLAGS.lstm_layer_num):
            enc_cells.append( attn_cell() )

        with tf.name_scope('lstm_cell_layers'):
            #mlstm_cell = tf.nn.rnn_cell.MultiRNNCell( enc_cells, state_is_tuple = True)
            mlstm_cell = tf.keras.layers.StackedRNNCells( enc_cells)

        #将 LSTM 中的状态初始化为全 0  数组，batch_size 给出一个 batch 的大小
        #返回[batch_size, 2*len(cells)],或者[batch_size, s]
        #这个函数只是用来生成初始化值的
        #init_state = mlstm_cell.zero_state(batch_size,dtype = tf.float32)
    
        #https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
        #cell: BasicLSTMCell，BasicRNNCell，GRUCell 的对象实例，自己定义的cell 内容
        #inputs_data:LSTM要处理的数据 如果是time_major=True，input的维度是[max_time, batch_size, input_size]，反之就是[batch_size, max_time, input_zise]；
        #time_major 默认是False
        #return: output_rnn里面，包含了所有时刻的输出H
        #        final_states里面，包含了最后一个时刻的输出 H 和 C；
        output_rnn,final_states = tf.nn.dynamic_rnn(mlstm_cell, inputs_data,initial_state = init_state, dtype = tf.float32)
        #output_rnn,final_states = tf.keras.layers.RNN(mlstm_cell)
        
        with tf.name_scope("outputs"):
            #自定义输出层
            output = tf.reshape(output_rnn,[-1,lstm_num_units]) 
            w_out = self.weights['out']
            bias_out = self.biases['out']
            pred = tf.matmul(output, w_out) + bias_out
    
        return pred,final_states
    
    #————————————————训练数据————————————————————
    
    def train_lstm(self, iteration, batch_size = 60,time_step = 20,train_begin = 0,train_end = 5800):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #定义命名空间，使用tensorboard进行可视化
        with tf.name_scope("inputs"):
            X = tf.placeholder(tf.float32, shape = [None,time_step,input_size])
            tf.summary.histogram('X', X)

        with tf.name_scope("target"):
            Y = tf.placeholder(tf.float32, shape = [None,time_step,output_size])
            tf.summary.histogram('Target', Y)
    
        #获取训练数据
        batch_index,train_x,train_y = self.get_train_data(batch_size,time_step,train_begin,train_end)
    
        with tf.variable_scope("my_lstm"):
            pred,_  =  self.lstm(X)
    
        #定义损失函数：二次代价函数（均方误差（MSE:Mean Square Error ））
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        #Adm梯度优化算法
        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(),max_to_keep = 15)
   
        with tf.name_scope("loss"):
            tf.summary.scalar('Loss', loss)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./log/', tf.get_default_graph())

        summary_interval = 0;
        if( iteration < summary_total):
            summary_interval = 1
        else:
            summary_interval = iteration/summary_total;
        print ('summary_interval=',summary_interval)
        sess_config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=12,
            gpu_options = tf.GPUOptions(allow_growth=True), 
            allow_soft_placement = True, 
            log_device_placement = False)
        with tf.train.MonitoredTrainingSession(
                                checkpoint_dir=FLAGS.model_path+"/",
                                config=sess_config,
                                save_checkpoint_secs=60) as sess:
            for i in range(iteration):  
                for step in range(len(batch_index)-1):
                    step, _, loss_ = sess.run([global_step, train_op,loss], feed_dict = {X:train_x[ batch_index[step]:batch_index[step+1] ], Y:train_y[ batch_index[step]:batch_index[step+1] ]})
                print("Number of iterations:",i," loss:",loss_)
    
    #————————————————预测数据————————————————————
    def inference_lstm(self, time_step = 20):
        X = tf.placeholder(tf.float32, shape = [None, time_step, input_size])
        mean, std, test_x, test_y,padding = self.get_test_data( time_step )
        
        #共享lstm()函数中定义的权重参数
        with tf.variable_scope("my_lstm", reuse = tf.AUTO_REUSE):
            pred, _ = self.lstm(X)
    
        saver = tf.train.Saver( tf.global_variables() )
        with tf.Session() as sess:
            #读取刚才训练好的模型参数
            module_file  =  tf.train.latest_checkpoint('model')
            saver.restore(sess, module_file)
            test_predict = []
    
            for step in range(len(test_x)):
              prob = sess.run(pred, feed_dict = { X:[ test_x[ step ] ] } )
              #reshape((-1))把prob变成一维
              predict = prob.reshape((-1))
    
              #保存本次的预测结果
              test_predict.extend(predict)
   
            test_y = np.array(test_y)*std[input_size] + mean[input_size]
            #今天的数据，是明天的结果,把标签提前一个
            #test_y = test_y[1:]
            
            test_predict = test_predict[:-padding]
            #还原到实际数据
            test_predict = np.array(test_predict)*std[input_size] + mean[input_size]
     
            #偏差程度
            acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)]) 
            print('acc:', acc)
            print("The accuracy of this predict:",1 - acc)
            #R Squared
            rs = mean_squared_error(test_y[:len(test_predict)],test_predict)/ np.var(test_y[:len(test_predict)])
            print('R Squared:', rs)
            print("The accuracy of this predict :",1 - rs)
    
            #以折线图表示结果
            plt.figure()
            #预测结果用蓝线表示
            plt.plot(list(range(len(test_predict))), test_predict, label='predict',color = 'b',)
            #验证标签用红线表示
            plt.plot(list(range(len(test_y))), test_y, label='test', color = 'r')
            plt.legend(loc= 'upper right', fontsize= 10)
            plt.savefig("./trend.png")
            plt.show()
            
def my_main(_):
    #默认值
    data_file = 'stock_data.csv'
    iteration = 5000

    print ('iteration=',FLAGS.iteration)
    print ('data_file=',FLAGS.data_file)
    lstm = rnn_lstm(FLAGS.data_file)
    #训练数据
    lstm.train_lstm(FLAGS.iteration)
    #预测数据
    lstm.inference_lstm()

if __name__ == "__main__":
    tf.app.run(main = my_main)

