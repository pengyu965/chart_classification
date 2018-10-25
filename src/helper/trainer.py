#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: trainer.py


import os
import sys
import time
import numpy as np
import tensorflow as tf
import scipy.misc


def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            time,
            summary_val=None,
            summary_writer=None
            ):
    print('[step: {}] [time: {:.2f}]'.format(global_step, time), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. ), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. )
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)

class Trainer(object):
    # def __init__(self, train_model, valid_model, train_data, init_lr=1e-3):
    def __init__(self, train_model, valid_model, init_lr=1e-3):
        self._t_model = train_model
        self._v_model = valid_model
        # self._train_data = train_data
        self._init_lr = init_lr

        self._train_op = train_model.get_train_op()
        self._train_loss_op = train_model.get_loss()
        self._train_accuracy_op = train_model.get_accuracy()

        self._valid_loss_op = valid_model.get_loss()
        self._valid_accuracy_op = valid_model.get_accuracy()
        # self._train_summary_op = train_model.get_train_summary()
        # self._valid_summary_op = train_model.get_valid_summary()

        self.global_step = 0
        self.epoch_id = 0

    def train_epoch(self, sess, tr_dir, val_dir, epoch, batch_size, keep_prob=1., summary_writer=None):
        # if self.epoch_id < 35:
        #     self._lr = self._init_lr
        # elif self.epoch_id < 50:
        #     self._lr = self._init_lr / 10.
        # else:
        #     self._lr = self._init_lr / 100.
        # self._t_model.set_is_training(True)
        display_name_list = ['loss', 'accuracy']
        cur_summary = None
        start_time = time.time()

        # cur_epoch = self._train_data.epochs_completed

        step = 0
        loss_sum = 0
        acc_sum = 0
        # self.epoch_id += 1
        rank = 0
        
        each_num = int(batch_size/10)
        lens=[]
        labeldic = {}
        i=0
        batch_label = []
        for file in os.listdir(tr_dir):
            imglist = os.listdir(tr_dir + file)
            lens.append(len(imglist))
            
            labeldic[i] = file
            
            label = [i]*each_num
            batch_label = batch_label + label
            i=i+1
        

        idx=min(lens)//each_num

        for ep in range(epoch):
            step = 0
            if ep < epoch//3:
                self._lr = self._init_lr
            elif ep < epoch*2//3:
                self._lr = self._init_lr/10
            else:
                self._lr = self._init_lr/100
            # if ep< epoch//3:
            #     self._lr = self._init_lr
            # elif ep< epoch*2//3:
            #     self._lr = self._init_lr/10
            # else:
            #     self._lr = self._init_lr/100

            for idi in range(idx):
                batch_img = []
                for file in os.listdir(tr_dir):
                    all_img = os.listdir(tr_dir+file)
                    img_list = all_img[idi*each_num:(idi+1)*each_num]
                    newbatchimg = [
                        scipy.misc.imread(tr_dir+file+"/"+img).astype(np.float) for img in img_list
                        ]
                    batch_img = batch_img + newbatchimg

        # for ep in range(epoch):
        #     if ep<35:
        #         self._lr = self._init_lr
        #     elif ep<50:
        #         self._lr = self._init_lr/10
        #     else:
        #         self._lr = self._init_lr/100

        #     batch_img = []
        #     idx = 0
        #     i = 0
        #     for file in os.listdir(tr_dir):
        #         all_img = os.listdir(tr_dir+file)
        #         batch_label = [0] * 10
        #         batch_label[i] = 1
        #         i+=1
        #         while (idx+1)*batch_size < len(all_img):
        #             img_list = all_img[idx*batch_size:(idx+1)*batch_size]
        #             batch_img = [
        #                 scipy.misc.imread(tr_dir+file+"/"+img).astype(np.float) for img in img_list
        #                 ]

        # while cur_epoch == 100:
                self.global_step += 1
                step += 1

                # batch_data = self._train_data.next_batch_dict()
                im = batch_img
                label = batch_label
                _, loss, acc = sess.run(
                    [self._train_op, self._train_loss_op, self._train_accuracy_op], 
                    feed_dict={self._t_model.image: im,
                            self._t_model.label: label,
                            self._t_model.lr: self._lr,
                            self._t_model.keep_prob: keep_prob})

                loss_sum += loss
                acc_sum += acc

                # if step % 100 == 0:
                #     display(self.global_step,
                #         step,
                #         [loss, acc],
                #         display_name_list,
                #         'train',
                #         summary_val=cur_summary,
                #         summary_writer=summary_writer)

                print('==== epoch: {}, [{}/{}], lr:{} ===='.format(ep, idi, idx, self._lr))
                display(self.global_step,
                        step,
                        [loss, acc],
                        display_name_list,
                        'train',
                        time.time()-start_time,
                        summary_val=cur_summary,
                        summary_writer=summary_writer)

            # Validation
            val_display_name_list = ['loss', 'accuracy']
            cur_val_summary = None
            # dataflow.reset_epoch()

            val_step = 0
            loss_val_sum = 0
            acc_val_sum = 0


            
            val_lens=[]
            for val_file in os.listdir(val_dir):
                imglist = os.listdir(val_dir + val_file)
                val_lens.append(len(imglist))
                
            val_idx=min(val_lens)//each_num

            for val_idi in range(val_idx):
                val_batch_img = []
                for val_file in os.listdir(val_dir):
                    val_all_img = os.listdir(val_dir+val_file)
                    val_img_list = val_all_img[val_idi*each_num:(val_idi+1)*each_num]
                    val_newbatchimg = [
                        scipy.misc.imread(val_dir+val_file+"/"+val_img).astype(np.float) for val_img in val_img_list
                        ]
                    val_batch_img = val_batch_img + val_newbatchimg


                val_step += 1

                im = val_batch_img
                label = batch_label
                loss_val, acc_val = sess.run(
                    [self._valid_loss_op, self._valid_accuracy_op], 
                    feed_dict={self._v_model.image: im,
                            self._v_model.label: label})
                loss_val_sum += loss_val
                acc_val_sum += acc_val

            print('[Valid]: [{}/{}]'.format(val_idi,val_idx), end='')
            display(self.global_step,
                    val_step,
                    [loss_val_sum * 1./val_step, acc_val_sum * 1./val_step],
                    val_display_name_list,
                    'valid',
                    time.time()-start_time,
                    summary_val=cur_val_summary,
                    summary_writer=summary_writer)

            
                       
            
            

    def valid_epoch(self, sess, dir, batch_size, summary_writer=None):
        # display_name_list = ['loss', 'accuracy']
        # cur_summary = None

        # step = 0
        loss_sum = 0
        acc_sum = 0
        # self.epoch_id += 1

        each_num = int(batch_size/10)
        lens=[]
        labeldic = {}
        i=0
        batch_label = []
        for file in os.listdir(dir):
            imglist = os.listdir(dir + file)
            lens.append(len(imglist))
            
            labeldic[i] = file
            
            label = [i]*each_num
            batch_label = batch_label + label
            i=i+1
        

        idx=min(lens)//each_num
        if idx<1:
            print("\n***The number of Validation dataset is too small***\n Stoped")
            sys.exit()
        

        

        for idi in range(idx):
            batch_img = []
            for file in os.listdir(dir):
                all_img = os.listdir(dir+file)
                img_list = all_img[idi*each_num:(idi+1)*each_num]
                newbatchimg = [
                    scipy.misc.imread(dir+file+"/"+img).astype(np.float) for img in img_list
                    ]
                batch_img = batch_img + newbatchimg
            
            im = batch_img
            label = batch_label
            loss, acc = sess.run(
                [self._valid_loss_op, self._valid_accuracy_op], 
                feed_dict={self._v_model.image: im,
                           self._v_model.label: label})

            loss_sum += loss
            acc_sum += acc

        print("loss:{:.4f} accuracy:{:.4f}".format(loss_sum * 1./idx, acc_sum * 1./idx))
        return loss_sum * 1./idx, acc_sum * 1./idx
        # display(None,
        #         idx,
        #         [loss_sum *1./idx, acc_sum*1./idx],
        #         display_name_list,
        #         'Evaluate',
        #         summary_val=cur_eva_summary,
        #         summary_writer=None)
