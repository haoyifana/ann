#coding=utf-8
import xlrd
import random
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
def readExcel(fname,ishead=True,ind=0):
    bk = xlrd.open_workbook(fname)
    try:
        sh = bk.sheet_by_index(ind)
    except:
        print("no sheet in %s named Sheet1" % fname)
    # 获取行数
    nrows = sh.nrows
    # 获取列数
    ncols = sh.ncols
    print("nrows %d, ncols %d" % (nrows, ncols))
    # 获取第一行第一列数据
    cell_value = sh.cell_value(1, 1)
    # print cell_value
    if ishead:
        startIndex = 1
    else:
        startIndex = 0
    row_list = []
    # 获取各行数据
    for i in range(startIndex, nrows):
        row_data = sh.row_values(i)
        row_list.append(row_data)
    return row_list,nrows-startIndex,ncols-startIndex

def get_batch(data, label, batch_size):
    indlist = []
    data_batch = []
    label_batch = []
    for i in range(batch_size+1):
        r = random.randint(0, len(data)-1)
        if indlist.__contains__(r):
            i -= 1
            continue
        else:
            indlist.append(r)

            data_batch.append(data[r])
            label_batch.append(label[r])
    return data_batch, label_batch

# def get_all_test(data, label):
#     data_batch = []
#     label_batch = []
#     for i in range(len(data)):
#         if i % 5 == 0:
#             data_batch.append(data[i])
#             label_batch.append(label[i])
#     return data_batch, label_batch

def get_all_test(data, label):
    data_batch = []
    label_batch = []
    for i in range(len(data)):
        data_batch.append(data[i])
        label_batch.append(label[i])
    return data_batch, label_batch

def net(input, input_size, output_size, istrain):
    w = tf.get_variable(name="weight1",
                        shape=[input_size, 32],
                        trainable=True,
                        initializer=tf.initializers.random_normal)
    b = tf.get_variable(name="bias1",
                     shape=[32],
                     trainable=True,
                     initializer=tf.initializers.random_normal)
    net = tf.matmul(input, w) + b
    net = tf.nn.sigmoid(net)

    w = tf.get_variable(name="weight2",
                        shape=[32, 64],
                        trainable=True,
                        initializer=tf.initializers.random_normal)
    b = tf.get_variable(name="bias2",
                        shape=[64],
                        trainable=True,
                        initializer=tf.initializers.random_normal)
    net = tf.matmul(net, w) + b
    net = tf.nn.sigmoid(net)


    w = tf.get_variable(name="weight3",
                        shape=[64, output_size],
                        trainable=True,
                        initializer=tf.initializers.random_normal)
    b = tf.get_variable(name="bias3",
                        shape=[output_size],
                        trainable=True,
                        initializer=tf.initializers.random_normal)
    net = tf.matmul(net, w) + b
    net = tf.nn.sigmoid(net)
    #
    # w = tf.get_variable(name="weight3",
    #                     shape=[16, output_size],
    #                     trainable=True,
    #                     initializer=tf.initializers.random_normal)
    # b = tf.get_variable(name="bias3",
    #                     shape=[output_size],
    #                     trainable=True,
    #                     initializer=tf.initializers.random_normal)
    # net = tf.matmul(net, w) + b
    # net = tf.nn.sigmoid(net)

    return net

def get_data_and_label_from_excel(xls_path):
    train_data, _, _ = readExcel(xls_path, False, 2)
    train_label, _, _ = readExcel(xls_path, False, 3)
    test_data, _, _ = readExcel(xls_path, False, 4)
    test_label, _, _ = readExcel(xls_path, False, 5)
    return train_data, train_label, test_data, test_label

def train(input_size, output_size):
    data, label, test_data, test_label= get_data_and_label_from_excel("E:\deeplearning\\net\\23个输入变量.xlsx")

    x = tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, output_size])

    logits = net(x, input_size, output_size, False)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
    loss = tf.reduce_mean(loss)

    curr_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(1e-1, curr_step, decay_steps=1000, decay_rate=0.97)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=curr_step)

    predict_label = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predict_label, tf.argmax(y_, 1))
    accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        test_data, tast_label = get_all_test(test_data, test_label)
        acc = sess.run(accurary, feed_dict={x: test_data, y_: tast_label})
        print("---test:acc=%.4f%%" % (acc * 100))

        init_learning_rate = 0.01
        max_step = 200000
        for i in range(max_step):
            curr_step = i
            batch_xs, batch_ys = get_batch(data, label, 30)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if i % 1000 == 0 or i == max_step-1:
                [acc, los, rate] = sess.run([accurary, loss, learning_rate] ,feed_dict={x:batch_xs, y_:batch_ys})
                print("---step:%d, acc=%.4f%%, loss=%.4f, learning_rate=%.6f" % (i, acc*100, los, rate))
            if i % 1000 == 0 or i == max_step-1:
                test_data, tast_label = get_all_test(test_data, test_label)
                [acc, prediction] = sess.run([accurary, predict_label], feed_dict={x: test_data, y_: tast_label})
                print("---test:acc=%.4f%%" % (acc*100))
                print(prediction)
train(23,3)