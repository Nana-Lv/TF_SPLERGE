#-*-coding:utf-8-*-
import os, tqdm
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.python.platform import gfile

from config import split_config
from config import merge_config


class GenGroundTruth:
    def __init__(self, mode='split'):
        if mode not in ['split', 'merge']:
            raise ValueError("GenGroundTruth's mode must be 'split' or 'merge'")
        
        self.mode = mode
        self.current_config = split_config if mode == 'split' else merge_config
        
        self.input_path = self.current_config.original_data
        self.output_path = self.current_config.tfrecords_data
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        
        self.merge_flag = {}
        self.merge_data_num = 0
        self.find_need_merge()
        print('There are {} tables should be merged.'.format(self.merge_data_num))
        self.generate_tfrecords()
        
        return

    def gen_split_gt(self, img_path, json_path):
        """
        Generate ground truth for split stage.
        :param img_path:
        :param json_path:
        :return:
        """
        img = cv2.imread(img_path)
        with open(json_path, 'r') as f:
            cells = f.readlines()
        h, w, c = img.shape
        mask_r = np.zeros((h, w)).astype(np.uint8)
        mask_c = np.zeros((h, w)).astype(np.uint8)
        mask_char = np.zeros((h, w)).astype(np.uint8)
        for cell in cells:
            cell = eval(cell[:-1])
            sr, er, sc, ec = cell['merge_info']
            x1, y1, x2, y2 = cell['bbox_info']
            mask_char[y1: y2, x1: x2] = 1
            if er == -1 and ec == -1:
                mask_r[y1: y2, x1: x2] = 1
                mask_c[y1: y2, x1: x2] = 1
            elif er == -1 and ec != -1:
                mask_r[y1: y2, x1: x2] = 1
            elif er != -1 and ec == -1:
                mask_c[y1: y2, x1: x2] = 1
        
        # cv2.imshow('img', img * 255)
        # cv2.imshow('maskr', mask_r * 255)
        # cv2.imshow('maskc', mask_c * 255)
        # cv2.imshow('mask_char', mask_char * 255)
        # cv2.waitKey()
        
        has_char_row = np.nonzero(np.sum(mask_r, axis=1))[0]
        
        gt_row = np.ones((h)).astype(np.int8)# * 255
        gt_row[has_char_row] = 0
    
        has_char_col = np.nonzero(np.sum(mask_c, axis=0))[0]
        gt_col = np.ones((w)).astype(np.int8)# * 255
        gt_col[has_char_col] = 0
        return gt_row, gt_col, mask_char

    def check_need_merge(self, json_path):
        """
        Check the table's grids whether need to be merged.
        :param json_path:
        :return:
        """
        with open(json_path, 'r') as f:
            cells = f.readlines()
        self.merge_flag[json_path] = 0
        for cell in cells:
            cell = eval(cell[:-1])
            sr, er, sc, ec = cell['merge_info']
            if er == -1 and ec != -1:
                self.merge_flag[json_path] = 1
            elif er != -1 and ec == -1:
                self.merge_flag[json_path] = 1
        if self.merge_flag[json_path] == 1:
            self.merge_data_num += 1
        return
    
    def find_need_merge(self):
        """
        Find all tables which contain grids need merged.
        :return:
        """
        file_list = os.listdir(self.input_path)
        for i, img_name in tqdm.tqdm(enumerate(file_list)):
            if img_name.split('.')[-1] != 'png':
                continue
            img_path = os.path.join(self.input_path, img_name)
            json_path = img_path.replace('.png', '.json')
            self.check_need_merge(json_path)
        return
        
    def generate_tfrecords(self, valid_rate=0.1, test_rate=0.1):
        """
        Generate tfrecords for training.
        :param valid_rate:
        :param test_rate:
        :return:
        """
        train_filepath = os.path.join(self.output_path, self.current_config.train_data_name)
        valid_filepath = os.path.join(self.output_path, self.current_config.valid_data_name)
        test_filepath = os.path.join(self.output_path, self.current_config.test_data_name)
        train_writer = tf.io.TFRecordWriter(train_filepath)
        valid_writer = tf.io.TFRecordWriter(valid_filepath)
        test_writer = tf.io.TFRecordWriter(test_filepath)
        train_cnt, valid_cnt, test_cnt = 0, 0, 0
        train_cnt_nomerge, valid_cnt_nomerge, test_cnt_nomerge = 0, 0, 0
 
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        file_list = os.listdir(self.input_path)
        
        prob = np.random.rand(len(file_list))
        for i, img_name in tqdm.tqdm(enumerate(file_list)):
            if img_name.split('.')[-1] != 'png':
                continue
            img_path = os.path.join(self.input_path, img_name)
            image_shape = cv2.imread(img_path).shape
            image_raw_data = gfile.FastGFile(img_path, 'rb').read()
            
            json_path = img_path.replace('.png', '.json')
            gt_row, gt_col, mask_char = self.gen_split_gt(img_path, json_path)
            
            example = tf.train.Example(
                features=tf.train.Features(feature={'image': _bytes_feature(image_raw_data),
                                                    'mask_char': _bytes_feature(mask_char.tostring()),
                                                    'split_gt_row': _bytes_feature(gt_row.tostring()),
                                                    'split_gt_col': _bytes_feature(gt_col.tostring()),
                                                    'imgH': _int64_feature(image_shape[0]),
                                                    'imgW': _int64_feature(image_shape[1]),
                                                    'imgC': _int64_feature(image_shape[2])}))
            chance = prob[i]
            if self.mode == 'merge':
                if self.merge_flag[json_path]:
                    if chance < valid_rate:
                        valid_cnt += 1
                        valid_writer.write(example.SerializeToString())
                        train_cnt += 1
                        train_writer.write(example.SerializeToString())
                    elif chance < valid_rate + test_rate:
                        test_cnt += 1
                        test_writer.write(example.SerializeToString())
                    else:
                        train_cnt += 1
                        train_writer.write(example.SerializeToString())
                else:
                    if train_cnt_nomerge + test_cnt_nomerge > self.merge_data_num:
                        continue
                    if chance < valid_rate:
                        valid_cnt_nomerge += 1
                        valid_writer.write(example.SerializeToString())
                        train_cnt_nomerge += 1
                        train_writer.write(example.SerializeToString())
                    elif chance < valid_rate + test_rate:
                        test_cnt_nomerge += 1
                        test_writer.write(example.SerializeToString())
                    else:
                        train_cnt_nomerge += 1
                        train_writer.write(example.SerializeToString())

            elif self.mode == 'split':
                if chance < valid_rate:
                    valid_cnt += 1
                    valid_writer.write(example.SerializeToString())
                    train_cnt += 1
                    train_writer.write(example.SerializeToString())
                elif chance < valid_rate + test_rate:
                    test_cnt += 1
                    test_writer.write(example.SerializeToString())
                else:
                    train_cnt += 1
                    train_writer.write(example.SerializeToString())

        # close files
        train_writer.close()
        valid_writer.close()
        test_writer.close()
        print('Train image: {}, Valid image: {}, Test image: {}'.format(train_cnt, valid_cnt, test_cnt))
        print(train_cnt_nomerge, valid_cnt_nomerge, test_cnt_nomerge)
        
        return
    
    @classmethod
    def parse(cls, record):
        """
        Post processing cann't be put in this part.
        :param record:
        :return:
        """
        features = tf.io.parse_single_example(record, features={'image': tf.io.FixedLenFeature([], tf.string),
                                                                'mask_char': tf.io.FixedLenFeature([], tf.string),
                                                                'split_gt_row': tf.io.FixedLenFeature([], tf.string),
                                                                'split_gt_col': tf.io.FixedLenFeature([], tf.string),
                                                                'imgH': tf.io.FixedLenFeature([], tf.int64),
                                                                'imgW': tf.io.FixedLenFeature([], tf.int64),
                                                                'imgC': tf.io.FixedLenFeature([], tf.int64)
                                                                })
        images = tf.io.decode_jpeg(features['image'])
        mask_char = features['mask_char']
        gt_row = features['split_gt_row']
        gt_col = features['split_gt_col']
        imgH, imgW, imgC = features['imgH'], features['imgW'], features['imgC']
        return images, [gt_row, gt_col], mask_char, [imgH, imgW, imgC]

    def check_image(self, num=5):
        """
        Check the data wrote into the tfrd files.
        :param num:
        :return:
        """
        tfrecords = os.path.join(self.output_path, self.current_config.train_data_name)
        dataset = tf.data.TFRecordDataset(tfrecords)
    
        dataset = dataset.map(self.parse).shuffle(1000).batch(1)
        cnt = 0
        for data in dataset:
            if cnt == num:
                break
            image_batch, label_batch, mask_char, img_shape = data
            img_norm = image_batch[0]
            gt_row, gt_col = label_batch[0]
            h, w, c = img_shape[0]

            mask_char1 = np.reshape(np.frombuffer(mask_char[0].numpy(), dtype=np.uint8), (h, w))
            cv2.imshow('img', cv2.cvtColor(img_norm.numpy(), code=cv2.COLOR_RGB2BGR))
            cv2.imshow('mask_char', 255 * mask_char1)
            cv2.waitKey()
            gt_row = np.frombuffer(gt_row.numpy(), dtype=np.uint8)
            gt_col = np.frombuffer(gt_col.numpy(), dtype=np.uint8)
            cv2.imshow('row', gt_row * 255)
            cv2.imshow('col', gt_col * 255)
            cv2.waitKey()
            cnt += 1
        return


if __name__ == '__main__':
    GGT = GenGroundTruth(mode='mere')  # or mode='merge'
    GGT.check_image()



