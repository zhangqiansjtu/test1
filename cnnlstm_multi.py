import tensorflow as tf
import numpy as np
import cnnlstm_input
import time


my_data = cnnlstm_input.Data_Control(
    './data/lipcontrol/cutdata7/'
    )
n_class = 10
Xlen = my_data.trainlen
X = my_data.traindata
X = X.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2], my_data.traindata.shape[3], 1)
Y = my_data.trainlabel


Keep_p = 0.6
batch_size = 32

#测试数据
Xtestlen = my_data.testlen
Xtest = my_data.testdata
Xtest = Xtest.reshape(-1,my_data.testdata.shape[1],my_data.testdata.shape[2],my_data.traindata.shape[3],1)
Ytest = my_data.testlabel




sess = tf.InteractiveSession()  # 创建session
def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

def bn_layer(inputs, phase_train, scope=None):
    #定义Batch Normalization layer
       return tf.cond(phase_train,
                      lambda: tf.contrib.layers .batch_norm(inputs,decay=0.9,is_training=True, scale=True,
            updates_collections=None, scope=scope),
                      lambda: tf.contrib.layers .batch_norm(inputs, decay=0.9,is_training=False, scale=True,
            updates_collections=None, scope=scope, reuse = True))
    #     return 0
    # return tf.layers.batch_normalization(inputs, training=phase_train, scale=True)



# 三，搭建网络,定义算法公式，也就是forward时的计算
def cnnBN(Iinputs,Qinputs, train,keep_prob, reuse=False, name='cnnBN'):
    with tf.variable_scope(name, reuse=reuse) as scope:


        Iw_conv1 = weight_variable([5, 3, 1, 8])
        Ib_conv1 = bias_variable([8])
        Iconv1 = conv2d(Iinputs, Iw_conv1) + Ib_conv1
        IBN_out1 = bn_layer(Iconv1, train, scope='IBN1')
        Ih_conv1 = tf.nn.relu(IBN_out1)
        Ih_pool1 = max_pool_2x2(Ih_conv1)


        Qw_conv1 = weight_variable([5, 3, 1, 8])
        Qb_conv1 = bias_variable([8])
        Qconv1 = conv2d(Qinputs, Qw_conv1) + Qb_conv1
        QBN_out1 = bn_layer(Qconv1,train,scope='QBN1')
        Qh_conv1 = tf.nn.relu(QBN_out1)
        Qh_pool1 = max_pool_2x2(Qh_conv1)

## 第二层卷积操作 ##
        Iw_conv2 = weight_variable([5, 3, 8, 16])
        Ib_conv2 = bias_variable([16])
        Iconv2 = conv2d(Ih_pool1, Iw_conv2) + Ib_conv2
        IBN_out2 = bn_layer(Iconv2, train, scope='IBN2')
        Ih_conv2 = tf.nn.relu(IBN_out2)
        Ih_pool2 = tf.nn.max_pool(Ih_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

        Qw_conv2 = weight_variable([5, 3, 8, 16])
        Qb_conv2 = bias_variable([16])
        Qconv2 = conv2d(Qh_pool1, Qw_conv2) + Qb_conv2
        QBN_out2 = bn_layer(Qconv2,train,scope='QBN2')
        Qh_conv2 = tf.nn.relu(QBN_out2)
        Qh_pool2 = tf.nn.max_pool(Qh_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        Iw_conv3 = weight_variable([5, 3, 16, 32])
        Ib_conv3 = bias_variable([32])
        Iconv3 = conv2d(Ih_pool2, Iw_conv3) + Ib_conv3
        IBN_out3 = bn_layer(Iconv3, train, scope='IBN3')
        Ih_conv3 = tf.nn.relu(IBN_out3)
        # Ih_pool3 = Ih_conv3
        Ih_pool3 = tf.nn.max_pool(Ih_conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        Qw_conv3 = weight_variable([5, 3, 16, 32])
        Qb_conv3 = bias_variable([32])
        Qconv3=conv2d(Qh_pool2, Qw_conv3) + Qb_conv3
        QBN_out3 = bn_layer(Qconv3, train, scope='QBN3')
        Qh_conv3 = tf.nn.relu(QBN_out3)
        # Qh_pool3 = Qh_conv3
        Qh_pool3 = tf.nn.max_pool(Qh_conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')



        Ih_pool3 = tf.expand_dims(Ih_pool3, 3)
        Qh_pool3 = tf.expand_dims(Qh_pool3, 3)
        sensor_conv = tf.concat([Ih_pool3,Qh_pool3], 3)
        sensorw_conv1 = weight_variable([3, 3, 2, 32, 32])
        sensorb_conv1 = bias_variable([32])
        sensorconv1 = tf.nn.conv3d(sensor_conv, sensorw_conv1, strides=[1, 1, 1, 1, 1], padding='SAME') + sensorb_conv1
        sensorBN_out1 = bn_layer(sensorconv1, train, scope='sensorBN1')
        sensor_conv1 = tf.nn.relu(sensorBN_out1)
        sensor_conv1 = tf.nn.max_pool3d(sensor_conv1, ksize=[1, 2, 2, 1, 1], strides=[1, 2, 2, 1, 1], padding='SAME')
        # sensorw_conv2 = weight_variable([4, 2, 2, 16, 16])
        # sensorb_conv2 = bias_variable([16])
        # sensorconv2 = tf.nn.conv3d(sensor_conv1, sensorw_conv2, strides=[1, 1, 1, 1, 1], padding='SAME') + sensorb_conv2
        # sensorBN_out2 = bn_layer(sensorconv2, istraining, scope='sensorBN2')
        # sensor_conv2 = tf.nn.relu(sensorBN_out2)
        # sensor_conv2 = tf.nn.max_pool3d(sensor_conv2, ksize=[1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='SAME')
        h_conv = sensor_conv1
        h_conv_shape = h_conv.get_shape().as_list()
## 第五层全连接操作 ##
        full_units = 32
        W_fc1 = weight_variable([h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3] * h_conv_shape[4], full_units])
        # 偏移量
        b_fc1 = bias_variable([full_units])
        h_flat = tf.reshape(h_conv, [-1, h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3] * h_conv_shape[4]])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        h_fc1_shape = h_fc1.get_shape().as_list()
# 转换成lstm的输入形式
        h_fc1 = tf.reshape(h_fc1, [-1, my_data.traindata.shape[1], full_units])

        return h_fc1

xs = tf.placeholder(tf.float32, [None, my_data.traindata.shape[1] , my_data.traindata.shape[2],my_data.traindata.shape[3],1])
ys = tf.placeholder(tf.int32, [None])
x_seqlen = tf.placeholder(tf.int32, [None])
yso = tf.one_hot(ys,n_class)
keep_prob = tf.placeholder(tf.float32)
# istraining判断是不是训练，用在batch normalization层
istraining = tf.placeholder(tf.bool)
x_image = tf.reshape(xs, [-1, my_data.traindata.shape[2], my_data.traindata.shape[3], 1])
x_image_split = tf.split(x_image, 2, 2)
x_I = [x_image_split[0]]
x_Q = [x_image_split[1]]
x_I = tf.reshape(x_I, [-1, my_data.traindata.shape[2], 8, 1])
x_Q = tf.reshape(x_Q, [-1, my_data.traindata.shape[2], 8, 1])
hfc1 = cnnBN(x_I,x_Q, istraining,keep_prob,name='cnnBN')

               # time steps
n_hidden = [64, 128]      # neurons in hidden layer              # MNIST classes (0-9 digits)
n_layers = len(n_hidden)

weights = {
    'out': tf.Variable(tf.truncated_normal([2*n_hidden[-1], n_class],stddev=0.5))
}
biases = {
    'out': tf.Variable(tf.constant(0.1, shape=[n_class, ]))
}
def lstm_cell(n_hidden):
    return tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
def dropout_lstm_cell(n_hidden, keep_prob):
    return tf.contrib.rnn.DropoutWrapper(lstm_cell(n_hidden), output_keep_prob=keep_prob)
outputs, final_states1,final_states2= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=[dropout_lstm_cell(n_hidden[i], keep_prob) for i in range(n_layers)],
        cells_bw=[dropout_lstm_cell(n_hidden[i], keep_prob) for i in range(n_layers)],
        inputs=hfc1,
        sequence_length=x_seqlen,
        time_major=False,
        dtype=tf.float32)

outputs = tf.concat([final_states1[-1].h, final_states2[-1].h], 1)
results = tf.matmul(outputs, weights['out']) + biases['out']
# results=tf.clip_by_value(results,1e-10,1.0)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=results, labels=yso))
global_step=tf.Variable(0,trainable=False)
# lr = tf.train.exponential_decay(0.001, global_step,800,0.99, staircase=True)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy, global_step=global_step)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

# 五，开始数据训练以及评测

correct_prediction = tf.equal(tf.argmax(results, 1), tf.argmax(yso, 1))
pred=tf.argmax(results, 1)
yla=tf.argmax(yso, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
def generatebatch(X,step, batch_size):
    start = (step*batch_size)%len(X)
    if start + batch_size >len(X):
        start = ((step+1) * batch_size) % len(X)
    end = min(start + batch_size,len(X))
    return start, end # 生成每一个batch
# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    starttime=time.time()
    lasttime=time.time()
    randomind=range(0,len(X))
    best_step = 0
    best_acc=0
    for i in range(60000):
       start,end=generatebatch(randomind,i, batch_size)
       batch_x= X[start:end]
       batch_y = Y[start:end]
       batch_xs_slen = Xlen[start:end]
       train_step.run(feed_dict={xs: batch_x, ys: batch_y, x_seqlen: batch_xs_slen,keep_prob: Keep_p, istraining: True})
       if i % 100 == 0:
           train_accuracy,loss,yo,lg,fc= sess.run([accuracy,cross_entropy,yso,results,hfc1],
                                             feed_dict={xs: batch_x,
                                                        ys: batch_y,
                                                        x_seqlen: batch_xs_slen,
                                                        keep_prob: Keep_p,
                                                        istraining:True})
           testloss,test_accuracy,yo,lg,fc,pre,yl = sess.run([cross_entropy,accuracy,yso,results,hfc1,pred,yla],
                                             feed_dict={xs: Xtest,
                                                        ys: Ytest,
                                                        x_seqlen: Xtestlen,
                                                        keep_prob: 1,
                                                        istraining:False})
           if test_accuracy > best_acc:
               best_acc = test_accuracy
               best_step = i

           result=np.vstack((pre, yl))
           result=np.transpose(result)
           if test_accuracy>best_acc:
               best_acc = test_accuracy
               best_step = i
               resultbest = result
           nowtime = time.time()
           dur1 = nowtime-starttime
           dur2 = nowtime-lasttime
           lasttime = nowtime
           print("step %d, %0fs, %0fs, loss %g, training accuracy %g, testloss %g,testing accuracy %g" % (i, dur1,dur2,loss,train_accuracy,testloss,test_accuracy))

       if i % 100 == 0:
           randomind = list(range(X.shape[0]))
           np.random.shuffle(randomind)
       if i%2000 == 0:
           print("beststep %d, bestacc %g" %(best_step, best_acc))