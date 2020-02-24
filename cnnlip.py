import tensorflow as tf
import numpy as np
import cnn_input
import time


my_data = cnn_input.Data_Control(
    './data/lipcontrol/cutdata7/'
    )
n_class = 10


X = my_data.traindata
X = X.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2], 1)
Y = my_data.trainlabel

Keep_p = 0.6
batch_size = 32

#测试数据
Xtest = my_data.testdata
Xtest = Xtest.reshape(-1, my_data.testdata.shape[1], my_data.testdata.shape[2], 1)
Ytest = my_data.testlabel

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')

def bn_layer(inputs, phase_train, scope=None):
       return tf.cond(phase_train,
                      lambda: tf.contrib.layers .batch_norm(inputs,decay=0.9, is_training=True, scale=True,
            updates_collections=None, scope=scope),
                      lambda: tf.contrib.layers .batch_norm(inputs, decay=0.9, is_training=False, scale=True,
            updates_collections=None, scope=scope, reuse=True))

def cnnBN(Iinputs,Qinputs, train,keep_prob, reuse=False, name='cnnBN'):
    with tf.variable_scope(name, reuse=reuse) as scope:


        Iw_conv1 = weight_variable([8, 3, 1, 8])
        Ib_conv1 = bias_variable([8])
        Iconv1 = conv2d(Iinputs, Iw_conv1) + Ib_conv1
        IBN_out1 = bn_layer(Iconv1, train, scope='IBN1')
        Ih_conv1 = tf.nn.relu(IBN_out1)
        Ih_pool1 = max_pool_2x2(Ih_conv1)


        Qw_conv1 = weight_variable([8, 3, 1, 8])
        Qb_conv1 = bias_variable([8])
        Qconv1 = conv2d(Qinputs, Qw_conv1) + Qb_conv1
        QBN_out1 = bn_layer(Qconv1,train,scope='QBN1')
        Qh_conv1 = tf.nn.relu(QBN_out1)
        Qh_pool1 = max_pool_2x2(Qh_conv1)

## 第二层卷积操作 ##
        Iw_conv2 = weight_variable([8, 3, 8, 16])
        Ib_conv2 = bias_variable([16])
        Iconv2 = conv2d(Ih_pool1, Iw_conv2) + Ib_conv2
        IBN_out2 = bn_layer(Iconv2, train, scope='IBN2')
        Ih_conv2 = tf.nn.relu(IBN_out2)
        Ih_pool2 = tf.nn.max_pool(Ih_conv2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')

        Qw_conv2 = weight_variable([8, 3, 8, 16])
        Qb_conv2 = bias_variable([16])
        Qconv2 = conv2d(Qh_pool1, Qw_conv2) + Qb_conv2
        QBN_out2 = bn_layer(Qconv2,train,scope='QBN2')
        Qh_conv2 = tf.nn.relu(QBN_out2)
        Qh_pool2 = tf.nn.max_pool(Qh_conv2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')

        Iw_conv3 = weight_variable([5, 3, 16, 32])
        Ib_conv3 = bias_variable([32])
        Iconv3 = conv2d(Ih_pool2, Iw_conv3) + Ib_conv3
        IBN_out3 = bn_layer(Iconv3, train, scope='IBN3')
        Ih_conv3 = tf.nn.relu(IBN_out3)
        # Ih_pool3 = Ih_conv3
        Ih_pool3 = tf.nn.max_pool(Ih_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        Qw_conv3 = weight_variable([5, 3, 16, 32])
        Qb_conv3 = bias_variable([32])
        Qconv3 = conv2d(Qh_pool2, Qw_conv3) + Qb_conv3
        QBN_out3 = bn_layer(Qconv3, train, scope='QBN3')
        Qh_conv3 = tf.nn.relu(QBN_out3)
        # Qh_pool3 = Qh_conv3
        Qh_pool3 = tf.nn.max_pool(Qh_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



        Ih_pool3 = tf.expand_dims(Ih_pool3, 3)
        Qh_pool3 = tf.expand_dims(Qh_pool3, 3)
        sensor_conv = tf.concat([Ih_pool3,Qh_pool3], 3)
        sensorw_conv1 = weight_variable([3, 4, 2, 32, 32])
        sensorb_conv1 = bias_variable([32])
        sensorconv1 = tf.nn.conv3d(sensor_conv, sensorw_conv1, strides=[1, 1, 1, 1, 1], padding='SAME') + sensorb_conv1
        sensorBN_out1 = bn_layer(sensorconv1, train, scope='sensorBN1')
        sensor_conv1 = tf.nn.relu(sensorBN_out1)
        sensor_conv1 = tf.nn.max_pool3d(sensor_conv1, ksize=[1, 3, 1, 1, 1], strides=[1, 3, 1, 1, 1], padding='SAME')
        sensorw_conv2 = weight_variable([3, 4, 2, 32, 32])
        sensorb_conv2 = bias_variable([32])
        sensorconv2 = tf.nn.conv3d(sensor_conv1, sensorw_conv2, strides=[1, 1, 1, 1, 1], padding='SAME') + sensorb_conv2
        sensorBN_out2 = bn_layer(sensorconv2, istraining, scope='sensorBN2')
        sensor_conv2 = tf.nn.relu(sensorBN_out2)
        sensor_conv2 = tf.nn.max_pool3d(sensor_conv2, ksize=[1, 2, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding='SAME')
        h_conv = sensor_conv2
        h_conv_shape=h_conv.get_shape().as_list()
## 第五层全连接操作 ##
        full_unit = 128
        W_fc1 = weight_variable([h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3]* h_conv_shape[4], full_unit])
        b_fc1 = bias_variable([full_unit])
        h_flat = tf.reshape(h_conv, [-1, h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3]* h_conv_shape[4]])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([full_unit, n_class])
        b_fc2 = bias_variable([n_class])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv,h_fc1

xs = tf.placeholder(tf.float32, [None, my_data.traindata.shape[1], my_data.traindata.shape[2],1])
ys = tf.placeholder(tf.int32, [None])
yso=tf.one_hot(ys,n_class)
keep_prob = tf.placeholder(tf.float32)
# istraining判断是不是训练，用在batch normalization层
istraining=tf.placeholder(tf.bool)
x_image = tf.reshape(xs, [-1, my_data.traindata.shape[1], my_data.traindata.shape[2], 1])
x_image_split = tf.split(x_image, 2, 2)
x_I = [x_image_split[0]]
x_Q = [x_image_split[1]]
x_I = tf.reshape(x_I, [-1, my_data.traindata.shape[1], 8, 1])
x_Q = tf.reshape(x_Q, [-1, my_data.traindata.shape[1], 8, 1])
logits,fc1 = cnnBN(x_I, x_Q, istraining,keep_prob,name='cnnBN')
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=yso))
global_step=tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(0.001, global_step,800,0.99, staircase=True)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy, global_step=global_step)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

# 五，开始数据训练以及评测

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(yso, 1))
pred=tf.argmax(logits, 1)
yla=tf.argmax(yso, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
def generatebatch(X,step, batch_size):
    start = (step*batch_size)%len(X)
    if start + batch_size >len(X):
        start = ((step+1) * batch_size) % len(X)
    end = min(start + batch_size,len(X))
    return start, end # 生成每一个batch
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    starttime = time.time()
    lasttime = time.time()
    randomind = range(0, len(X))
    best_step = 0
    best_acc = 0
    resultbest = 0
    for i in range(60001):
       start,end = generatebatch(randomind,i, batch_size)
       batch_index = randomind[start:end]
       batch_x = X[batch_index]
       batch_y = Y[batch_index]
       train_step.run(feed_dict={xs: batch_x, ys: batch_y, keep_prob: Keep_p, istraining: True})
       if i % 400 == 0:
           train_accuracy, loss, yo, lg = sess.run([accuracy, cross_entropy, yso, logits],
                                             feed_dict={xs: batch_x,
                                                        ys: batch_y,
                                                        keep_prob: Keep_p,
                                                        istraining:True})

           testloss,test_accuracy, yo, lg, pre, yl = sess.run([cross_entropy, accuracy, yso, logits, pred, yla],
                                             feed_dict={xs: Xtest,
                                                        ys: Ytest,
                                                        keep_prob: 1,
                                                        istraining:False})
           result = np.vstack((pre, yl))
           result = np.transpose(result)
           if i == 0:
               predbest = pre
           if test_accuracy >= best_acc:
               best_acc = test_accuracy
               best_step = i
               resultbest = result
               predbest = pre
           nowtime = time.time()
           dur1 = nowtime-starttime
           dur2 = nowtime-lasttime
           lasttime = nowtime
           print("step %d, %0fs, %0fs, loss %g, tracc %g, teloss %g,teacc %g" % (i, dur1, dur2, loss, train_accuracy,testloss,test_accuracy))
       if i % 40 == 0:
           randomind = list(range(X.shape[0]))
           np.random.shuffle(randomind)
       if i % 10000 == 0 and i >= 2000:

           # for k in range(0, resultbest.shape[0]):
           #     if Ytest[k] != predbest[k]:
           #          print(Ytest[k], predbest[k], k)
           print("beststep %d, bestacc %g" % (best_step, best_acc))
print()
