import tensorflow as tf
import numpy as np
import cnn_input_phase
import time


my_data = cnn_input_phase.Data_Control(
    './data/lipcontrol/cutdata/'
    )
n_class = 10

X = my_data.traindata
X = X.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2],1)
Y = my_data.trainlabel


Keep_p = 0.6
batch_size = 32

#测试数据
Xtest = my_data.testdata
Xtest = Xtest.reshape(-1,my_data.testdata.shape[1],my_data.testdata.shape[2],1)
Ytest = my_data.testlabel




sess = tf.InteractiveSession()  # 创建session
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
                      lambda: tf.contrib.layers .batch_norm(inputs, decay=0.9, is_training=True, scale=True,
            updates_collections=None, scope=scope),
                      lambda: tf.contrib.layers .batch_norm(inputs, decay=0.9, is_training=False, scale=True,
            updates_collections=None, scope=scope, reuse = True))



def cnnBN(inputs, train,keep_prob, reuse=False, name='cnnBN'):
    with tf.variable_scope(name, reuse=reuse) as scope:
## 第一层卷积操作 ##
        W_conv1 = weight_variable([8, 3, 1, 8])
        b_conv1 = bias_variable([8])
        conv1 = conv2d(inputs, W_conv1) + b_conv1
        BN_out1 = bn_layer(conv1,train,scope='BN1')
        h_conv1 = tf.nn.relu(BN_out1)
        h_pool1 = max_pool_2x2(h_conv1)

        w_conv2 = weight_variable([8, 3, 8, 16])
        b_conv2 = bias_variable([16])
        conv2 = conv2d(h_pool1, w_conv2) + b_conv2
        BN_out2 = bn_layer(conv2,train,scope='BN2')
        h_conv2 = tf.nn.relu(BN_out2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')

        w_conv3 = weight_variable([5, 3, 16, 32])
        b_conv3 = bias_variable([32])
        conv3 = conv2d(h_pool2, w_conv3) + b_conv3
        BN_out3 = bn_layer(conv3, train, scope='BN3')
        h_conv3 = tf.nn.relu(BN_out3)
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 3, 2, 1], strides=[1, 3, 2, 1], padding='SAME')

        w_conv4 = weight_variable([3, 3, 32, 32])
        b_conv4 = bias_variable([32])
        conv4 = conv2d(h_pool3, w_conv4) + b_conv4
        BN_out4 = bn_layer(conv4, train, scope='BN4')
        h_conv4 = tf.nn.relu(BN_out4)
        h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME')


        w_conv5 = weight_variable([3, 3, 32, 32])
        b_conv5 = bias_variable([32])
        conv5 = conv2d(h_pool4, w_conv5) + b_conv5
        BN_out5 = bn_layer(conv5, train, scope='BN5')
        h_conv5 = tf.nn.relu(BN_out5)
        h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        h_conv = h_pool5
        h_conv_shape = h_conv.get_shape().as_list()


        f_units1 = 128
        W_fc1 = weight_variable([h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3], f_units1])
        b_fc1 = bias_variable([f_units1])
        h_flat = tf.reshape(h_conv, [-1, h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3]])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = weight_variable([f_units1, n_class])
        b_fc2 = bias_variable([n_class])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv, h_fc1

xs = tf.placeholder(tf.float32, [None, my_data.traindata.shape[1], my_data.traindata.shape[2], 1])
ys = tf.placeholder(tf.int32, [None])
yso = tf.one_hot(ys, n_class)
keep_prob = tf.placeholder(tf.float32)
istraining = tf.placeholder(tf.bool)
x_image = tf.reshape(xs, [-1, my_data.traindata.shape[1], my_data.traindata.shape[2], 1])
logits, fc1 = cnnBN(x_image, istraining,keep_prob,name='cnnBN')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=yso))
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy, global_step=global_step)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

# 五，开始数据训练以及评测

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(yso, 1))
pred = tf.argmax(logits, 1)
yla = tf.argmax(yso, 1)
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
    randomind = range(0,len(X))
    best_step = 0
    best_acc = 0
    resultbest = 0
    for i in range(60001):
       start,end = generatebatch(randomind,i, batch_size)
       batch_index = randomind[start:end]
       batch_x = X[batch_index]
       batch_y = Y[batch_index]
       train_step.run(feed_dict={xs: batch_x, ys: batch_y, keep_prob: Keep_p, istraining: True})
       if i % 200 == 0:
           train_accuracy, loss, yo, lg = sess.run([accuracy, cross_entropy, yso, logits],
                                             feed_dict={xs: batch_x,
                                                        ys: batch_y,
                                                        keep_prob: Keep_p,
                                                        istraining:True})

           testloss, test_accuracy, yo, lg, pre, yl = sess.run([cross_entropy,accuracy,yso,logits,pred,yla],
                                             feed_dict={xs: Xtest,
                                                        ys: Ytest,
                                                        keep_prob: 1,
                                                        istraining: False})
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
       if i % 10000 == 0 and i > 1:

           # for k in range(0, resultbest.shape[0]):
           #     if Ytest[k] != predbest[k]:
           #          print(Ytest[k], predbest[k], k)
           print("beststep %d, bestacc %g" % (best_step, best_acc))
           print()