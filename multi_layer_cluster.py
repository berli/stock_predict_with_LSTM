#!/usr/bin/python
# -*- coding: utf8 -*-)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
import sys
import os
import time
import kafka_producer


lstm_num_units = 64 #32 #128  
input_size = 1       #特征数量
output_size = 1

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.0003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('lstm_num_units', 64,'lstm units num ')
tf.app.flags.DEFINE_string("model_path", "model","the path for saving model")
tf.app.flags.DEFINE_string("train", "","train file")
tf.app.flags.DEFINE_string("inference", "","predict file")
tf.app.flags.DEFINE_integer("iteration", 100,"train iteration")
tf.app.flags.DEFINE_string("log_dir", "./log/", "checkpoint and model saving dir")
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("item", "","item name")
tf.app.flags.DEFINE_integer("item_id", 1,"item id")

tf.app.flags.DEFINE_float('keep_prob', 1, 'input keep prob')
tf.app.flags.DEFINE_integer('lstm_layer_num', 2,'lstm layer num ')

#python rnn_qps_cluster.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=ps 
#python rnn_qps_cluster.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=worker  --task_index=0 --item=item_name --item_id=id --train=train_file
#python rnn_qps_cluster.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=worker  --task_index=1 --item=item_name --item_id=id --train=train_file
#python rnn_qps_cluster.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225,127.0.0.1:2226 --job_name=worker  --task_index=2 --item=item_name --item_id=id --train=train_file

class rnn_lstm(object):
    def __init__(self, train_file, test_file):
        #——————————————————导入数据——————————————————————

        if( train_file != ''):
            fd_train = open(train_file)
            df_train = pd.read_csv(fd_train)     #读入训练数据
            self.train_data = df_train.iloc[:,0:2].values  #前闭后开,取第[0,2)列

        if( test_file != ''):
            fd_test = open(test_file)
            df_test = pd.read_csv(fd_test)       #读入测试数据
            self.test_data  = df_test.iloc[:,0:2].values   #前闭后开,取第[0,2)列
        
        '''
        self.TRAIN_END= int(len(self.data)*0.8);
        print 'TRAIN_END=',self.TRAIN_END,'TOTAL=',len(self.data)
        plt.figure()
        plt.plot(data)
        plt.show()
        '''
        #——————————————————定义权重和偏置参数——————————————————
        with tf.name_scope('weights'):
            #输入层、输出层, 权重、偏置
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
    def get_train_data(self, batch_size = 60,time_step = 20):#90w

        batch_index = []
        normalized_train_data = (self.train_data - np.mean(self.train_data,axis = 0))/np.std(self.train_data,axis = 0)  
        print "normalized_train_data:\n",normalized_train_data
    
        print "---------------------------------------------"
        #训练集
        train_x,train_y = [],[]  
        c  =  0;
        for i in range(len(normalized_train_data) - time_step):
           if i % batch_size == 0:
               batch_index.append(i)
           
           #取time_step个时间序列数据, 取前面input_size列数据
           x = normalized_train_data[i:i+time_step,:input_size]
           #取time_step个时间序列数据, 第input_size列数据,np.newaxis增加一个维度
           y = normalized_train_data[i:i+time_step,input_size,np.newaxis]
           train_x.append(x.tolist())
           train_y.append(y.tolist())
    
        batch_index.append((len(normalized_train_data)-time_step))
    
        return batch_index,train_x,train_y
    
    #————————————————————获取测试集——————————————————————
    def get_test_data(self, time_step = 20):
        #取begin后面的数据
    
        mean = np.mean(self.test_data, axis = 0)
        std = np.std(self.test_data, axis = 0)
        normalized_test_data = (self.test_data - mean)/std 
   
        padding = time_step - len(normalized_test_data)%time_step
        normalized_test_data = np.pad(normalized_test_data, (0, padding), 'constant')
        #有size个sample
        #size = (len(normalized_test_data) + time_step)//time_step 
        size = len(normalized_test_data) //time_step 
    
        test_x,test_y = [],[]
  
        i = 0;
        for i in range(size - 1 ):
           #特征列数据
           x = normalized_test_data[ i*time_step:(i+1)*time_step, :input_size]
           #标签列标签
           y = normalized_test_data[ i*time_step:(i+1)*time_step, input_size]
    
           test_x.append(x.tolist())
           #是extennd,不是append
           test_y.extend(y)
    
        test_x.append((normalized_test_data[(i+1)*time_step:,:input_size]).tolist())
        print('len(test_x)=',len(test_x))
        test_y.extend((normalized_test_data[(i+1)*time_step:,input_size]).tolist())
    
        return mean,std,test_x,test_y, padding
    
    #——————————————————实现lstm神经网络——————————————————
    def lstm(self, X):
        
        with tf.name_scope('weights'):
            batch_size = tf.shape(X)[0]
            time_step = tf.shape(X)[1]
            w_in = self.weights['in']
            b_in = self.biases['in']
        
        #将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
        input  =  tf.reshape(X,[-1,input_size])  
        inputs_data  =  tf.matmul(input, w_in) + b_in
    
        #将tensor转成3维，作为lstm cell的输入
        inputs_data  =  tf.reshape(inputs_data, [-1, time_step, lstm_num_units] ) 
    
        #num_units: 输出维度,图一中ht的维数，如果num_units=10,那么ht就是10维行向量, 官方解释:int, The number of units in the LSTM cell.
        #forget_bias：遗忘门的初始化偏置,默认是1.0，都不忘记, 0:都忘记
        #activation: 内部状态的激活函数,默认是: tanh.
        #cell = tf.nn.rnn_cell.LSTMCell( lstm_num_units )
        def attn_cell():
            lstm_cell = tf.nn.rnn_cell.LSTMCell( lstm_num_units )
            with tf.name_scope('lstm_dropout'):
                return tf.contrib.rnn.DropoutWrapper( lstm_cell, output_keep_prob = FLAGS.keep_prob)
   
        enc_cell = []
        for i in range(FLAGS.lstm_layer_num):
            enc_cell.append( attn_cell() )

        with tf.name_scope( 'lstm_cell_layer'):
            mlstm_cell = tf.nn.rnn_cell.MultiRNNCell( enc_cell, state_is_tuple = True)

        #将 LSTM 中的状态初始化为全 0  数组，batch_size 给出一个 batch 的大小
        #返回[batch_size, 2*len(cells)],或者[batch_size, s]
        #这个函数只是用来生成初始化值的
        init_state = mlstm_cell.zero_state(batch_size,dtype = tf.float32)
    
        #https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
        #cell: BasicLSTMCell，BasicRNNCell，GRUCell 的对象实例，自己定义的cell 内容
        #inputs:如果是time_major=True，input的维度是[max_time, batch_size, input_size]，反之就是[batch_size, max_time, input_zise]；
        #return: output_rnn里面，包含了所有时刻的输出H
        #        final_states里面，包含了最后一个时刻的输出 H 和 C；
        output_rnn,final_states = tf.nn.dynamic_rnn(mlstm_cell, inputs_data,initial_state = init_state, dtype = tf.float32)
       
        with tf.name_scope('outputs'):
            #自定义输出层
            output = tf.reshape(output_rnn,[-1,lstm_num_units]) 
            w_out = self.weights['out']
            bias_out = self.biases['out']
            pred = tf.matmul(output, w_out) + bias_out
    
        return pred,final_states
    
    #————————————————训练数据————————————————————
    
    def train_lstm(self, cluster, server, iteration=50, batch_size = 60,time_step = 120):
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" %(FLAGS.task_index), 
            cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            
            with tf.name_scope('inputs'):
                X = tf.placeholder(tf.float32, shape = [None,time_step,input_size])
                tf.summary.histogram('X',X)

            with tf.name_scope('targets'):
                Y = tf.placeholder(tf.float32, shape = [None,time_step,output_size])
                tf.summary.histogram('Y',Y)
    
            with tf.variable_scope("my_lstm"):
                pred,_  =  self.lstm(X)
    
            loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1])- tf.reshape(Y, [-1])))
            
            with tf.name_scope('loss'):
                tf.summary.scalar('Loss', loss)

            with tf.name_scope('train'):
                train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)
            
            #获取训练数据
            batch_index,train_x,train_y = self.get_train_data(batch_size,time_step)
    
        saver = tf.train.Saver(tf.global_variables(),max_to_keep = 15)
    
        summary_op = tf.summary.merge_all()
        #summary_writer = tf.summary.FileWriter('./log', tf.get_default_graph())


        summary_hook = tf.train.SummarySaverHook(save_secs=600,output_dir=FLAGS.log_dir, summary_op=summary_op)
        sess_config = tf.ConfigProto(device_count={"CPU":12},
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=12,
            gpu_options = tf.GPUOptions(allow_growth=True), 
            allow_soft_placement = True, 
            log_device_placement = False)
        #hooks = [tf.train.StopAtStepHook(last_step = FLAGS.train_max_step), summary_hook]

        with tf.train.MonitoredTrainingSession(master=server.target,
                                is_chief=(FLAGS.task_index == 0),
                                checkpoint_dir="./model/"+FLAGS.model_path+"/",
                                config=sess_config,
                                save_checkpoint_secs=60) as sess:
            #初始化全局变量
            #sess.run(tf.global_variables_initializer())
            for i in range(iteration):  
                for step in range(len(batch_index)-1):
                   #summary, _, loss_ = sess.run([summary_op, train_op,loss], feed_dict = {X:train_x[batch_index[step]:batch_index[step+1]], Y:train_y[batch_index[step]:batch_index[step+1]]})
                   _, loss_,step = sess.run([train_op,loss, global_step], feed_dict = {X:train_x[batch_index[step]:batch_index[step+1]], Y:train_y[batch_index[step]:batch_index[step+1]]})
                print("Number of iterations:",i,' step:',step," loss:",loss_)
                #if( i%summary_interval == 0 ):
                #    summary_writer.add_summary(summary, i)
            #保存模型
            #now = time.time() 
            #curDay = time.strftime("%Y-%m-%d", time.localtime(now)) #昨天和今天的数据
            #if( os.path.exists(model_path) ):
            #    os.rename(model_path, model_path+'_'+ curDay)
            #print("model_save: ",saver.save(sess, model_path+'/modle.ckpt'))
            print("The train has finished")
            os._exit(0)
    
    #————————————————预测数据————————————————————
    def inference_lstm(self, time_step = 120):
        X = tf.placeholder(tf.float32, shape = [None, time_step, input_size])
        mean, std, test_x, test_y,padding = self.get_test_data( time_step )
        
        with tf.variable_scope("my_lstm", reuse = tf.AUTO_REUSE):
            pred, _ = self.lstm(X)
    
        saver = tf.train.Saver( tf.global_variables() )
        with tf.Session() as sess:
            #读取刚才训练好的模型参数
            module_file  =  tf.train.latest_checkpoint("./model/"+FLAGS.model_path+"/")
            saver.restore(sess, module_file)
            test_predict = []
    
            for step in range(len(test_x)):
              prob = sess.run(pred, feed_dict = { X:[ test_x[ step ] ] } )
              predict = prob.reshape((-1))
    
              test_predict.extend(predict)
    
            #验证数据
            test_y = np.array(test_y)*std[input_size] + mean[input_size]

            #
            test_predict = test_predict[:-padding]

            #预测数据
            test_predict = np.array(test_predict)*std[input_size] + mean[input_size]
     
            #偏差程度
            rmse = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)]) 
            print('rmse:', rmse)
            print("The accuracy of this predict:",1 - rmse)
    
            #以折线图表示结果
            plt.figure()
            #预测结果用蓝线表示
            plt.plot(list(range(len(test_predict))), test_predict, label='predict',color = 'b',)
            #验证标签用红线表示
            plt.plot(list(range(len(test_y))), test_y, label='test', color = 'r')
            plt.legend(loc='upper right', fontsize=10)
            plt.savefig("./qps_trend.png")
            #终端是不能展示图片
            #plt.show()

            #——————————————————初始化kafka——————————————————
            kk = kafka_producer.qpsKafka();
            kk.connKafka()
            ##写入kafka
            kk.produceQPS(test_predict, FLAGS.item, FLAGS.item_id, )

def my_main(args):
    print('train_file:',FLAGS.train)
    print('test_file:',FLAGS.inference)
    print('worker_hosts:',FLAGS.worker_hosts)
    
    lstm = rnn_lstm(FLAGS.train, FLAGS.inference)
    
    #预测数据
    if( FLAGS.inference != ""):
        print 'predict....'
        lstm.inference_lstm()
        return
    
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    #训练数据
    if FLAGS.job_name == "ps":
      server.join()
    elif FLAGS.job_name == "worker":
        if( FLAGS.train != "" ):
            print 'train....'
            lstm.train_lstm(cluster,server, FLAGS.iteration )

if __name__ == "__main__":
    tf.app.run(my_main)

