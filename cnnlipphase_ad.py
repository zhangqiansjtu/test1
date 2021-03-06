import tensorflow as tf
import numpy as np
import cnn_input_phase
import time


my_data = cnn_input_phase.Data_Control(
    './data/lipcontrol/cutdata/'
    )
n_class = 10
n_user = 8
X = my_data.traindata
X = X.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2],1)
Y = my_data.trainlabel
Yuser = my_data.trainuser


Keep_p = 0.6
batch_size = 32

#测试数据
Xtest = my_data.testdata
Xtest = Xtest.reshape(-1,my_data.testdata.shape[1],my_data.testdata.shape[2],1)
Ytest = my_data.testlabel
Yusertest = my_data.testuser




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

        return h_conv

xs = tf.placeholder(tf.float32, [None, my_data.traindata.shape[1], my_data.traindata.shape[2], 1])
ys = tf.placeholder(tf.int32, [None])
yso = tf.one_hot(ys, n_class)
ysu = tf.placeholder(tf.int32, [None])
ysuser = tf.one_hot(ysu,n_user)
keep_prob = tf.placeholder(tf.float32)
istraining = tf.placeholder(tf.bool)
x_image = tf.reshape(xs, [-1, my_data.traindata.shape[1], my_data.traindata.shape[2], 1])
h_conv = cnnBN(x_image, istraining,keep_prob,name='cnnBN')
h_conv_shape = h_conv.get_shape().as_list()


f_units1 = 128
W_fc1 = weight_variable([h_conv_shape [1] * h_conv_shape [2] * h_conv_shape [3], f_units1])
b_fc1 = bias_variable([f_units1])
h_flat = tf.reshape(h_conv, [-1, h_conv_shape [1] * h_conv_shape [2] * h_conv_shape [3]])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([f_units1, n_class])
b_fc2 = bias_variable([n_class])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
logits1 = y_conv
#ad
# h_flatd = tf.reshape(h_conv, [-1, h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3]])
hdin=tf.concat([h_flat, y_conv], 1)
h_flatd_shape = h_flat.get_shape().as_list()
W_fcd1 = tf.get_variable('d_wfc1', [h_flatd_shape[1] + n_class, f_units1],initializer=tf.truncated_normal_initializer(stddev=0.1))
b_fcd1 = tf.get_variable('d_bfc1', [f_units1],initializer=tf.truncated_normal_initializer(stddev=0.1))
h_fcd1 = tf.nn.relu(tf.matmul(hdin, W_fcd1) + b_fcd1)

W_fcd3 = tf.get_variable('d_wfc3', [f_units1, n_user],initializer=tf.truncated_normal_initializer(stddev=0.1))
b_fcd3 = tf.get_variable('d_bfc3', [n_user],initializer=tf.truncated_normal_initializer(stddev=0.1))
y_convd = tf.nn.softmax(tf.matmul(h_fcd1, W_fcd3) + b_fcd3)
logits2 = tf.clip_by_value(y_convd, 1e-10, 1.0)

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = list(filter(lambda var: var not in d_vars, tvars))
aaa = tf.ones_like(logits2)-logits2
aaa = aaa/(n_user-1)
aaa = tf.clip_by_value(aaa, 1e-10, 1.0)


cross_entropy11=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1,labels=yso))
cross_entropy12=-tf.reduce_mean(ysuser * tf.log(aaa))
lamda= tf.placeholder(tf.float32)
cross_entropy2 = -tf.reduce_mean(ysuser * tf.log(logits2))
cross_entropy1 = cross_entropy11+lamda*cross_entropy12

global_step=tf.Variable(0,trainable=False)
opt =tf.train.AdamOptimizer()
trainerG1 = opt.minimize(cross_entropy11, var_list=g_vars)
trainerG = opt.minimize(cross_entropy1, var_list=g_vars)
trainerD = opt.minimize(cross_entropy2, var_list=d_vars)
result_domain=tf.argmax(logits2, 1)
# 五，开始数据训练以及评测
correct_prediction = tf.equal(tf.argmax(logits1, 1), tf.argmax(yso, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_prediction2 = tf.equal(tf.argmax(logits2, 1), tf.argmax(ysuser, 1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))


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
       start, end = generatebatch(randomind,i, batch_size)
       batch_index = randomind[start:end]
       batch_x = X[batch_index]
       batch_y = Y[batch_index]
       batch_yuser = Yuser[batch_index]
       if i <= 20000:
           kk = 9
       else:
           kk = 9 + (i // 10000 - 1)*1
       la = 0
       if i % kk == 0:
           la = 1
       if la == 0:
           _, train_accuracy, loss, loss1, loss2, user_accuracy = sess.run(
               [trainerG1, accuracy, cross_entropy1, cross_entropy11, cross_entropy12, accuracy2],
               feed_dict={xs: batch_x, ys: batch_y, ysu: batch_yuser, keep_prob: Keep_p, istraining: True, lamda: la})
       else:
           _, train_accuracy, loss, loss1, loss2, user_accuracy= sess.run(
               [trainerG, accuracy, cross_entropy1, cross_entropy11, cross_entropy12, accuracy2],
               feed_dict={xs: batch_x, ys: batch_y, ysu: batch_yuser, keep_prob: Keep_p, istraining: True, lamda: la})
       _, lossD, user_accuracyD = sess.run([trainerD, cross_entropy12, accuracy2],
                                           feed_dict={xs: batch_x, ys: batch_y, ysu: batch_yuser, keep_prob: 1,
                                                      istraining: False})

       if i % 200 == 0:
           train_accuracy, loss = sess.run([accuracy, cross_entropy11],
                                             feed_dict={xs: batch_x,
                                                        ys: batch_y,
                                                        keep_prob: Keep_p,
                                                        istraining:True})

           testloss, test_accuracy= sess.run([cross_entropy11,accuracy],
                                             feed_dict={xs: Xtest,
                                                        ys: Ytest,
                                                        keep_prob: 1,
                                                        istraining: False})
           if test_accuracy >= best_acc:
               best_acc = test_accuracy
               best_step = i
           nowtime = time.time()
           dur1 = nowtime-starttime
           dur2 = nowtime-lasttime
           lasttime = nowtime
           print("step %d, %0fs, %0fs, loss %g,loss1 %g,loss2 %g, tracc %g, teacc %g, uacc %g %g" % (
           i, dur1, dur2, loss, loss1, loss2, train_accuracy, test_accuracy, user_accuracy, la))
       if i % 40 == 0:
           randomind = list(range(X.shape[0]))
           np.random.shuffle(randomind)
       if i % 2000 == 0 and i > 1:
           print("beststep %d, bestacc %g" % (best_step, best_acc))
           print(la)