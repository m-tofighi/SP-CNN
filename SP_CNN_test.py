# This code is for testing the trained SP-CNN network. 
# How to run: 
# 	1- Download the trained network models (checkpoint) for each dataset,
# 	2- Modify the directories to the paths containing the trained model and test data,
# 	3- Specify a path to save the outputs
#	4- For assessment we used the same Matlab code provided by Sirinukunwattana et al.
#
# For further information and questions please contact M. Tofighi at tofighi@psu.edu

import numpy as np
import tensorflow as tf
import glob
import cv2
import ntpath
EPOCH_SIZE = 100

TEST_PATH = './PathToDataset/'  # Directory of test data
ALL_MODEL_PATH = './record/model/' # Directory of trained model
ALL_TEST_SAVE_PATH = './output/' # Directory to save test results

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

with tf.Session() as sess:
    w_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 64], stddev=1e-1), name='w_conv1')
    w_conv2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-1), name='w_conv2')
    w_conv3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-1), name='w_conv3')
    w_conv4 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-1), name='w_conv4')
    w_conv5 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-1), name='w_conv5')
    w_conv6 = tf.Variable(tf.random_normal([3, 3, 64, 1], stddev=1e-1), name='w_conv6')

    b_conv1 = tf.Variable(tf.zeros([64]), name='b_conv1')
    b_conv2 = tf.Variable(tf.zeros([64]), name='b_conv2')
    b_conv3 = tf.Variable(tf.zeros([64]), name='b_conv3')
    b_conv4 = tf.Variable(tf.zeros([64]), name='b_conv4')
    b_conv5 = tf.Variable(tf.zeros([64]), name='b_conv5')
    b_conv6 = tf.Variable(tf.zeros([1]), name='b_conv6')
    # declaring inputs
    input_cnn = tf.placeholder(tf.float32)

    # implementing the network
    h_conv1 = tf.nn.relu(tf.add(tf.nn.conv2d(input_cnn, w_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)
    h_conv6 = tf.nn.conv2d(h_conv5, w_conv6, strides = [1,1,1,1], padding = 'SAME') + b_conv6

    # Loading the test input and the model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    for Ep in range(0, EPOCH_SIZE-1):
        MODEL_PATH = ALL_MODEL_PATH + 'tf' + str(Ep) + '.ckpt'
        TEST_SAVE_PATH = ALL_TEST_SAVE_PATH + 'Ep_' + str(Ep) + '/'
        saver.restore(sess, MODEL_PATH)
        print(glob.glob(TEST_PATH + '*.bmp'))
        for testImgName in glob.glob(TEST_PATH + '*.bmp'):
            print('Test Image %s'% path_leaf(testImgName))
            testImg = cv2.imread(testImgName, 0).astype(np.float32)
            testImg_normalized = testImg / 255
            test_input = np.array([testImg_normalized])
            test_elem = np.rollaxis(test_input, 0,3)
            test_data = test_elem[np.newaxis, ...]
            output_data = sess.run([h_conv6], feed_dict={input_cnn:test_data})
            output_image = output_data[0][0,:,:,0]
            output_image = output_image*255
            tst_name = path_leaf(testImgName)
            testedImgName = tst_name[0:-4] + '_Ep' + str(Ep) + '.bmp'
            cv2.imwrite(TEST_SAVE_PATH + testedImgName, output_image)

    print('Testing finished!')