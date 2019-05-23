# -*- coding: utf-8 -*-
from utility import *
import sys
import time
import shutil
import os
import h5py
import numpy as np
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def file_name(file_dir, f):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == f:
                L.append(os.path.join(root, file))
    return L


def read_data(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        #print(train_data.shape)
        #print(train_label.shape)
    return train_data, train_label


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = 255.0
    return 10.0 * tf_log10((max_pixel ** 2) / (tf.reduce_mean(tf.squared_difference(y_pred, y_true))))  # tf.square


def EDSR_train(train_data_dir, test_data_file, model_save_path):
    #train_data, train_label = read_data(train_data_file)
    test_data, test_label = read_data(test_data_file)

    print 'Read test data! The shape of test data: ' + str(test_data.shape)
    batch_size = 64

    total_epoch = 40
    per_epoch_save = 1
    start_epoch = 0
    is_load = False
    image_size = 48
    channels = 3
    scale_factor = 2
    residual_factor = 0.1

    images = tf.placeholder(tf.float32, [None, image_size, image_size, channels],
                            name='images')
    labels = tf.placeholder(tf.float32, [None, image_size * scale_factor, image_size * scale_factor, channels]
                            , name='labels')
    learning_rate = tf.placeholder(tf.float32)

    pred = EDSR(images, scale_factor=scale_factor, residual_factor=residual_factor)
    # loss = tf.reduce_mean(tf.square(labels - pred))
    loss = tf.reduce_mean(tf.losses.absolute_difference(labels, pred))
    psnr = PSNR(labels, pred)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # beta1=0.1, beta2=0.1
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if is_load:
            start_epoch = 18
            check_point_path = model_save_path + '/' + str(start_epoch) + '/'  # 保存好模型的文件路径
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        bar_length = 30
        train_file = file_name(train_data_dir, '.h5')
        print 'The number of file:' + str(len(train_file))
        if len(train_file) == 1:
            train_data, train_label = read_data(train_file[0])
            print 'Read train data before training!'
        for ep in range(1 + start_epoch, total_epoch + 1):
            lr = math.pow(0.95, ep - 1) * 0.0004
            if lr < 0.00001:
                lr = 0.00001
            print '*'*60
            print 'epoch %d/%d, lr = %2.5f:' % (ep, total_epoch, lr)
            start_time = time.time()
            for file_number in range(len(train_file)):
                if len(train_file) > 1:
                    train_data, train_label = read_data(train_file[file_number])
                    print 'Read the file ' + str(file_number + 1) + '! The shape of the data: ' + str(train_data.shape)
                indices = np.random.permutation(len(train_data))  # 每次随机打乱数据
                train_data = train_data[indices]
                train_label = train_label[indices]
                iterations = train_data.shape[0] // batch_size
                iterations_all = 0
                pre_index = 0
                train_loss = 0.0
                train_psnr = 0.0
                start_time0 = time.time()
                #iterations = 1
                for it in range(1, iterations + 1):
                    iterations_all = iterations_all + 1
                    batch_x = train_data[pre_index:pre_index + batch_size]
                    batch_y = train_label[pre_index:pre_index + batch_size]
                    _, batch_loss, batch_psnr = sess.run([train_step, loss, psnr],
                                                         feed_dict={images: batch_x, labels: batch_y, learning_rate: lr})
                    train_loss += batch_loss
                    train_psnr += batch_psnr
                    pre_index += batch_size
                    if it == iterations:
                        train_loss /= iterations_all
                        train_psnr /= iterations_all
                        test_loss, test_psnr = sess.run([loss, psnr], feed_dict={images: test_data, labels: test_label})

                        s1 = "\r%d/%d [%s%s] - batch_time = %.2fs - train_loss = %.5f - train_psnr = %.2f" % (
                            it, iterations, ">" * (bar_length * it // iterations),
                            "-" * (bar_length - bar_length * it // iterations),
                            (time.time() - start_time0) / it,
                            train_loss, train_psnr)
                        sys.stdout.write(s1)
                        sys.stdout.flush()
                        print '\ncost_time: %ds, test_loss: %.5f, test_psnr: %.2f' % (int(time.time() - start_time),
                                                                                      test_loss, test_psnr)
                    else:
                        s1 = "\r%d/%d [%s%s] - batch_time = %.2fs - train_loss = %.5f - train_psnr = %.2f" % (
                            it, iterations, ">" * (bar_length * it // iterations),
                            "-" * (bar_length - bar_length * it // iterations),
                            (time.time() - start_time0) / it,
                            train_loss / it, train_psnr / it)  # run_test()
                        sys.stdout.write(s1)
                        sys.stdout.flush()
            if ep % per_epoch_save == 0:
                path = model_save_path + '/save/' + str(ep) + '/'
                save_model = saver.save(sess, path + 'EDSR_model')
                new_path = model_save_path + '/' + str(ep) + '/'
                shutil.move(path, new_path)
                # 模型首先是被保存在save下面的,直接保存的话，前面的epoch对应的文件夹会出现内部文件被删除的情况，原因不明；所以这里用shutil.move把模型所在的文件夹移动了一下
                print '\nModel saved in file: %s' % save_model


def main():
    train_file = 'train_data_div2k'
    test_file = 'test.h5'
    model_save_path = 'EDSR_checkpoint_div2k'

    if os.path.exists(model_save_path) == False:
        print('The ' + '"' + model_save_path + '"' + 'can not find! Create now!')
        os.mkdir(model_save_path)

    if os.path.exists(model_save_path + '/save') == False:
        print('The ' + '"save' + '"' + ' can not find! Create now!')
        os.mkdir(model_save_path + '/save')

    EDSR_train(train_file, test_file, model_save_path)


if __name__ == '__main__':
    main()