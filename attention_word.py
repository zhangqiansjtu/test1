import numpy as np
import time
import tensorflow as tf
import attention_input_word
from tensorflow.python.layers.core import Dense

n_classes = 10
my_data = attention_input_word.Data_Control(
    './data/lipcontrol/cutdata7/', n_classes
)
#label集合需要添加三类label，分别是补全类“pad”，开始类“GO”， 结束类“EOS”
X = my_data.traindata
X = X.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2], my_data.traindata.shape[3], 1)
Y = my_data.trainlabel
Y = Y.astype(np.int)
Xlen = my_data.trainlen
Ylen = my_data.trainlabel_len

Xtest = my_data.testdata
Xtest = Xtest.reshape(-1, my_data.traindata.shape[1], my_data.traindata.shape[2], my_data.traindata.shape[3], 1)
Ytest = my_data.testlabel
Ytest = Ytest.astype(np.int)
Xtestlen = my_data.testlen
Ytestlen = my_data.testlabel_len

def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]



def get_inputs():
    '''
    模型输入tensor
    '''
    xs = tf.placeholder(tf.float32,
                        [None, my_data.traindata.shape[1], my_data.traindata.shape[2], my_data.traindata.shape[3], 1])

    inputs = tf.reshape(xs, [-1, my_data.traindata.shape[2], my_data.traindata.shape[3], 1], name='inputs')
    targets = tf.placeholder(tf.int32, [None, my_data.alllabel.shape[1]], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    istraining = tf.placeholder(tf.bool)

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = Y.shape[1]
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return xs, inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length, istraining

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME')


def bn_layer(inputs, phase_train, scope=None):
    # 定义Batch Normalization layer
    return tf.cond(phase_train,
                   lambda: tf.contrib.layers.batch_norm(inputs, decay=0.9, is_training=True, scale=True,
                                                        updates_collections=None, scope=scope),
                   lambda: tf.contrib.layers.batch_norm(inputs, decay=0.9, is_training=False, scale=True,
                                                        updates_collections=None, scope=scope, reuse=True))

def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, istraining, keep_prob):
    '''
    构造Encoder层，包含CNN和LSTM

    参数说明：
    #     - input_data: 输入tensor
    #     - rnn_size: rnn隐层结点数量
    #     - num_layers: 堆叠的rnn cell数量
    #     - source_sequence_length: 源数据的序列长度
    #     - istraining: 是否训练，用于BN
    #     '''

    # CNN

    W_conv1 = weight_variable([5, 6, 1, 8])
    b_conv1 = bias_variable([8])
    conv1 = conv2d(input_data, W_conv1) + b_conv1
    BN_out1 = bn_layer(conv1, istraining, scope='BN1')
    h_conv1 = tf.nn.relu(BN_out1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([5, 3, 8, 16])
    b_conv2 = bias_variable([16])
    conv2 = conv2d(h_pool1, w_conv2) + b_conv2
    BN_out2 = bn_layer(conv2, istraining, scope='BN2')
    h_conv2 = tf.nn.relu(BN_out2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 2, 1], strides=[1, 3, 2, 1], padding='SAME')

    w_conv3 = weight_variable([3, 3, 16, 32])
    b_conv3 = bias_variable([32])
    conv3 = conv2d(h_pool2, w_conv3) + b_conv3
    BN_out3 = bn_layer(conv3, istraining, scope='BN3')
    h_conv3 = tf.nn.relu(BN_out3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv = h_pool3
    h_conv_shape = h_conv.get_shape().as_list()

    full_units = 32
    W_fc1 = weight_variable([h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3], full_units])
    b_fc1 = bias_variable([full_units])
    h_flat = tf.reshape(h_conv, [-1, h_conv_shape[1] * h_conv_shape[2] * h_conv_shape[3]])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)
    convout = tf.reshape(h_fc1, [-1, my_data.traindata.shape[1], full_units])
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    # cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size[i]) for i in range(num_layers)])
    #
    # encoder_output, _ = tf.nn.dynamic_rnn(cell, convout,
    #                                                   sequence_length=source_sequence_length, dtype=tf.float32)

    encoder_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=[get_lstm_cell(rnn_size[i]) for i in range(num_layers)],
        cells_bw=[get_lstm_cell(rnn_size[i]) for i in range(num_layers)],
        inputs=convout,
        sequence_length=source_sequence_length,
        time_major=False,
        dtype=tf.float32)
    return encoder_output

def process_decoder_input(data):
    '''
    补充<GO>，并移除最后一个字符
    '''
    # cut掉最后一个字符
    ending = tf.strided_slice(data, [0, 0], [tf.shape(data)[0], -1], [1, 1])
    decoder_input = tf.concat([tf.fill([tf.shape(data)[0], 1], n_classes+1), ending], 1)

    return decoder_input


def decoding_layer(decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_output, decoder_input, keep_prob, source_sequence_length):
    '''
    构造Decoder层

    参数：
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''
    # 1. Embedding
    decoder_embeddings = tf.Variable(tf.random_uniform([n_classes+3, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 2. 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=keep_prob)



    def get_attention_cell(rnn_size):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=rnn_size, memory=encoder_output, memory_sequence_length=source_sequence_length)
        cell = get_decoder_cell(rnn_size)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=rnn_size,
                                                        output_attention=True)
        return attn_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_attention_cell(rnn_size[i]) for i in range(num_layers)])
    # 3. Output全连接层
    output_layer = Dense(n_classes+3,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4. Training decoder
    with tf.variable_scope("decode"):
        # 得到help对象
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # 构造decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           cell.zero_state(dtype=tf.float32, batch_size=tf.shape(xs)[0]),
                                                           output_layer)
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
        # attention_matrices = final_state.alignment_history.stack(
        #     name="train_attention_matrix")
    # 5. Predicting decoder
    # 与training共享参数
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([n_classes+1], dtype=tf.int32), [tf.shape(xs)[0]],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     n_classes + 2)
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             cell.zero_state(dtype=tf.float32,
                                                                             batch_size=tf.shape(xs)[0]),
                                                             output_layer)
        predicting_decoder_output = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  istraining, decoding_embedding_size,
                  rnn_size, num_layers, keep_prob):
    # 获取encoder的状态输出
    encoder_output = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         istraining,
                                         keep_prob)

    # 预处理后的decoder输入
    decoder_input = process_decoder_input(targets)

    # 将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_output,
                                                                        decoder_input,
                                                                        keep_prob,
                                                                        source_sequence_length)

    return training_decoder_output, predicting_decoder_output


# Batch Size
batch_size = 32
# RNN Size
rnn_size = [128, 128]
# Number of Layers
num_layers = 2
# Embedding Size
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

# 构造graph
train_graph = tf.Graph()

with train_graph.as_default():
    # 获得模型输入
    xs, input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length, istraining = get_inputs()
    keep_prob = tf.placeholder(tf.float32)
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       istraining,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers,
                                                                       keep_prob)

    training_logits = tf.identity(training_decoder_output[0].rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output[0].rnn_output, name='predictions_logits')
    predicting_results = tf.identity(predicting_decoder_output[0].sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    # accuracy = tf.contrib.metrics.accuracy(targets, predicting_logits, weights=masks, name='accuracy')
    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(cost)



def generatebatch(X,step, batch_size):
    start = (step * batch_size) % len(X)
    if start + batch_size >len(X):
        start = ((step+1) * batch_size) % len(X)

    end = min(start + batch_size,len(X))
    batch_xs = X[start:end]
    return batch_xs # 生成每一个batch


def get_accuracy(pre_result, target):
    edit_dis = []
    for i in range(0, target.shape[0]):
        pred = []
        for j in pre_result[i]:
            if j == n_classes or j == n_classes+2:
                break
            pred.append(j)
        tar = [j for j in target[i] if j != n_classes and j != n_classes+2]
        edit_dis.append(edit_distance(pred, tar) / len(tar))
    accuracy = 1 - np.mean(edit_dis)
    return accuracy
# config = tf.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    last_time = start_time
    best_step = 0
    best_acc = 0
    for step in range(0, 60000):
        sources_batch = generatebatch(X, step, batch_size)
        targets_batch = generatebatch(Y, step, batch_size)
        sources_lengths = generatebatch(Xlen, step, batch_size)
        targets_lengths = generatebatch(Ylen, step, batch_size)

        _, loss, pre_result, tra_logits, pre_logits = sess.run(
            [train_op, cost, predicting_results, training_logits, predicting_logits],
            {xs: sources_batch,
             targets: targets_batch,
             lr: learning_rate,
             istraining: True,
             target_sequence_length: targets_lengths,
             source_sequence_length: sources_lengths,
             keep_prob: 0.8})

        if step % 40 == 0:
            # 计算validation loss
            tra_acc = get_accuracy(pre_result, targets_batch)
            test_result, tra_logits, test_logits = sess.run(
                [predicting_results, training_logits, predicting_logits],
                {xs: Xtest,
                 targets: Ytest,
                 lr: learning_rate,
                 istraining: False,
                 target_sequence_length: Ytestlen,
                 source_sequence_length: Xtestlen,
                 keep_prob: 1})
            test_acc = get_accuracy(test_result, Ytest)
            now_time = time.time()
            duration1 = now_time - start_time
            duration2 = now_time - last_time
            last_time = now_time
            print("step %d, %0f (%0f)s, loss %g, tracc %g, teacc %g" % (
                step, duration1, duration2, loss, tra_acc, test_acc))
            if test_acc >= best_acc:
                best_acc = test_acc
                best_step = step
            if step % 400 == 0:
                print("beststep %d, bestacc %g" % (best_step, best_acc))
