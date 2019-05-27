# -*- coding: utf-8 -*-
import cv2
from utility import *
import os
import scipy.io as sio
import numpy as np
import math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
#session = tf.Session(config=config)


def file_name(file_dir, f):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == f:
                L.append(os.path.join(root, file))
    return L


def cal_psnr(target, ref):
    # assume RGB image
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    p = 20 * math.log10(255. / rmse)
    return p


def rgb2y(image, model='RGB'):#image RGB
    image_shape = image.shape
    if image_shape[2] == 3:
        if model == 'RGB':
            B = image[:, :, 2]
            G = image[:, :, 1]
            R = image[:, :, 0]

        elif model == 'BGR':
            B = image[:, :, 0]
            G = image[:, :, 1]
            R = image[:, :, 2]
        Y = R * 0.256789 + G * 0.504129 + B * 0.097906 + 16
    else:
        Y = image
    Y = np.round(Y)
    return Y


if __name__ == '__main__':

    database = "./Set5"
    model_path = './EDSR_checkpoint_div2k/'
    scale_factor = 2
    channels = 3
    #feature_size = 256
    #residual_num = 32
    residual_factor = 0.1

    start_model = 12
    step = 1
    model_num = 1

    data = tf.placeholder(tf.float32, [1, None, None, channels], name='input')
    predit = EDSR(data, scale_factor=scale_factor, residual_factor=residual_factor)

    with tf.Session(config=config) as sess:

        LR_path = file_name("./" + database + "/LRX" + str(scale_factor), ".mat")
        Gnd_path = file_name("./" + database + "/Gnd", ".mat")
        for model in range(1, 1+model_num):
            print '*' * 30
            image_psnr = 0.0
            check_point_path = model_path + str(start_model + step * (model - 1)) + '/'
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(len(LR_path)):
                low_image = sio.loadmat(LR_path[i])['im_low']#255
                original_image = sio.loadmat(Gnd_path[i])['im_original']
                shape = low_image.shape
                img_bic = np.zeros((1, shape[0], shape[1], channels), dtype=float)
                img_bic[0, :, :, :] = low_image
                shape = original_image.shape
                output = sess.run([predit], feed_dict={data: img_bic})
                #输出是一个list
                output = output[0]
                #output[output[:] > 255] = 255
                #output[output[:] < 0] = 0
                high_image = output[0, :, :, :]
                high_image = np.round(high_image)
                high_y = rgb2y(high_image, 'RGB')
                high_y = high_y[scale_factor:shape[0] - scale_factor, scale_factor:shape[1] - scale_factor]

                original_y = rgb2y(original_image, 'RGB')
                original_y = original_y[scale_factor:shape[0] - scale_factor, scale_factor:shape[1] - scale_factor]

                p = cal_psnr(original_y, high_y)
                image_psnr += p
                print str(i) + '.' + str(p)
            print 'average psnr = ' + str(image_psnr/len(LR_path))

