#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception_cifar.py


import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf
import scipy.misc
import time

sys.path.append('../')
import loader as loader
from src.nets.googlenet import GoogLeNet_cifar
from src.helper.trainer import Trainer
from src.helper.evaluator import Evaluator

LOGS_ROOT = '../logs/'
LOG_PATH = LOGS_ROOT + '/log/'
WEIGHT_PATH = LOGS_ROOT + '/weight/'

DATA_ROOT = "../data/"
TR_DATA_ROOT = DATA_ROOT + "/trdir/"
VAL_DATA_ROOT = "../valdir/"
PRED_DATA_ROOT = "../predictdir/"

if os.path.exists(LOGS_ROOT) == False:
    os.mkdir(LOGS_ROOT)
if os.path.exists(LOG_PATH) == False:
    os.mkdir(LOG_PATH)
if os.path.exists(WEIGHT_PATH) == False:
    os.mkdir(WEIGHT_PATH)


# DATA_PATH = '../data/dataset/cifar/'
# PRETRINED_PATH = '../data/pretrain/inception/googlenet.npy'
# IM_PATH = '../data/cifar/'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--predict', action='store_true',
                        help='Get prediction result')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    parser.add_argument('--load', type=int, default=99,
                        help='Epoch id of pre-trained model')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.4,
                        help='Keep probability for dropout')
    parser.add_argument('--maxepoch', type=int, default=100,
                        help='Max number of epochs for training')

    parser.add_argument('--im_name', type=str, default='.png',
                        help='Part of image name')

    return parser.parse_args()

def train():
    FLAGS = get_args()
    # Create Dataflow object for training and testing set
    # train_data, valid_data = loader.load_cifar(
    #     cifar_path=DATA_PATH, batch_size=FLAGS.bsize, subtract_mean=True)

    pre_trained_path=None
    if FLAGS.finetune:
        # Load the pre-trained model (on ImageNet)
        # for convolutional layers if fine tuning
        pre_trained_path = PRETRINED_PATH

    # Create a training model
    train_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, pre_trained_path=pre_trained_path,
        bn=True, wd=0, sub_imagenet_mean=False,
        conv_trainable=True, fc_trainable=True)
    train_model.create_train_model()
    # Create a validation model
    valid_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    valid_model.create_test_model()

    # create a Trainer object for training control
    trainer = Trainer(train_model, valid_model, init_lr=FLAGS.lr)
    
    start_time = time.time()
    loss = []
    acc = []
    j=1
    for i in os.listdir(DATA_ROOT):
        print("***\n***\nThe {}th Training\n***\n***".format(j))
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(LOG_PATH)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
            # if os.path.exists('{}checkpoint'.format(WEIGHT_PATH)) == True:
            #     saver.restore(sess, '{}chartclassification-epoch{}-batchsize{}-lr{}'.format(WEIGHT_PATH, FLAGS.maxepoch, FLAGS.bsize, FLAGS.lr))
            #     print("[*]load success")
            # for epoch_id in range(FLAGS.maxepoch):
                # train one epoch
            print("***\n***\nThe {}th Training\n***\n***".format(j))
            tr_dir = DATA_ROOT+ i +"/preprocessdata1/"
            val_dir = DATA_ROOT + i +"/preprocessdata2/"    
            trainer.train_epoch(sess, tr_dir, val_dir, FLAGS.maxepoch, FLAGS.bsize, keep_prob=FLAGS.keep_prob, summary_writer=writer)
            loss_e, acc_e = trainer.valid_epoch(sess, val_dir, FLAGS.bsize)
            loss.append(loss_e)
            acc.append(acc_e)
            # test the model on validation set after each epoch
            # trainer.valid_epoch(sess, dataflow=valid_data, summary_writer=writer)
            # saver.save(sess, '{}inception-cifar-epoch-{}'.format(WEIGHT_PATH, FLAGS.maxepoch))
            saver.save(sess, '{}chartclassification-epoch{}-batchsize{}-lr{}'.format(WEIGHT_PATH, FLAGS.maxepoch, FLAGS.bsize, FLAGS.lr))
            writer.close()
        j+=1
    print("===\nTotal Time:{:.2f}, Average Loss:{:.4f}, Average Accuracy:{:.4f}\n===".format(time.time()-start_time, sum(loss)*1./5,sum(acc)*1./5))

def evaluate():
    FLAGS = get_args()

    train_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, pre_trained_path=None,
        bn=True, wd=0, sub_imagenet_mean=False,
        conv_trainable=True, fc_trainable=True)
    train_model.create_train_model()
    # Create Dataflow object for training and testing set
    # train_data, valid_data = loader.load_cifar(
    #     cifar_path=DATA_PATH, batch_size=FLAGS.bsize, subtract_mean=True)
    # Create a validation model
    valid_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    valid_model.create_test_model()

    trainer = Trainer(train_model, valid_model, init_lr=FLAGS.lr)

    # create a Evaluator object for evaluation
    # evaluator = Evaluator(valid_model)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # load pre-trained model cifar
        saver.restore(sess, '{}chartclassification-epoch{}-batchsize{}-lr{}'.format(WEIGHT_PATH, FLAGS.maxepoch, FLAGS.bsize, FLAGS.lr))
        # print('training set:', end='')
        # trainer.valid_epoch(sess,TR_DATA_ROOT, FLAGS.bsize)
        print('Validation set:', end='')
        trainer.valid_epoch(sess,VAL_DATA_ROOT,FLAGS.bsize)

def predict():
    FLAGS = get_args()
    # Read Cifar label into a dictionary
    # label_dict = loader.load_label_dict(dataset='cifar')
    # # Create a Dataflow object for test images
    # image_data = loader.read_image(
    #     im_name=FLAGS.im_name, n_channel=3,
    #     data_dir=IM_PATH, batch_size=1, rescale=False)

    # Create a testing GoogLeNet model
    # label_dict = {}
    # i = 0
    # for file in os.listdir(TR_DATA_ROOT):
    #     label_dict[i] = file 
    #     i+=i 
    # for file in os.listdir(PRED_DATA_ROOT):
    #     img=scipy.misc.imread(PRED_DATA_ROOT+file).astype(np.float)


    test_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    test_model.create_test_model()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}chartclassification-epoch{}-batchsize{}-lr{}'.format(WEIGHT_PATH, FLAGS.maxepoch, FLAGS.bsize, FLAGS.lr))
        
        label_dict = {}
        i = 0
        for file in os.listdir(DATA_ROOT+"1/preprocessdata1"):
            label_dict[i] = file 
            i=i+1


        for file in os.listdir(PRED_DATA_ROOT):
            img=[scipy.misc.imread(PRED_DATA_ROOT+file).astype(np.float)]
            # read batch files

            # get batch file names
            # get prediction results
            pred = sess.run(test_model.layers['top_5'],
                            feed_dict={test_model.image: img})
            # display results
            print('===============================')
            print('[image]: {}'.format(file))
            print('---')
            for j in range(3): 
                re_prob = pred[0][j]
                re_label = pred[1][j]
                print('{}: probability: {:.4f}, label: {}'.format(j+1, re_prob, label_dict[re_label]),'\n---')

if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.eval:
        evaluate()
    if FLAGS.predict:
        predict()
