#-*-coding:utf-8-*-
import cv2
import tensorflow as tf
import numpy as np
import os
from config import split_config
from config import merge_config

from model import split
from model import merge
from merge_train import parse
from model.loss import gen_merge_inputs, cal_D, cal_R

if __name__ == '__main__':
    
    Split = split.Split()
    Split.load_weights(split_config.saved_models + 'split_499')
    
    Merge = merge.Merge()
    Merge.load_weights(merge_config.saved_models + 'merge_1092')
    
    valid_data = tf.data.TFRecordDataset(os.path.join(merge_config.tfrecords_data, merge_config.valid_data_name)).map(parse).batch(merge_config.batch_size)

    for batch, (_image_batch, *_) in enumerate(valid_data):
        if batch != 4:
            continue
        image_batch = tf.image.convert_image_dtype(_image_batch, tf.float32)

        inputs, grid_loc = gen_merge_inputs(image_batch, Split)
        matrix_u2, matrix_u3, matrix_d2, matrix_d3, matrix_l2, matrix_l3, matrix_r2, matrix_r3 = Merge(inputs, grid_loc)
        D3, R3 = cal_D(matrix_u3, matrix_d3), cal_R(matrix_l3, matrix_r3)
        D3, R3 = D3.numpy(), R3.numpy()
        print(np.max(D3), np.max(R3))
        
        grid_loc_row, grid_loc_col = grid_loc
        print(image_batch.shape, grid_loc_row, grid_loc_col)

        image = image_batch[0].numpy()
        
        h, w, c = image.shape
        for row in grid_loc_row:
            cv2.line(image, (0, row), (w, row), thickness=2, color=(0, 0, 255))
        for col in grid_loc_col:
            cv2.line(image, (col, 0), (col, h), thickness=2, color=(0, 255, 128))

        cv2.namedWindow('ori', 0)
        cv2.imshow('ori', image)
        cv2.waitKey()
        cv2.imshow('ori', image)
        cv2.waitKey()
    
        
        
        
    