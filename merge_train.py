#-*-coding:utf-8-*-
import os
import tensorflow as tf

from config import merge_config
from model.split import Split
from model.merge import Merge
from model.loss import loss_merge, gen_merge_inputs

tf.config.experimental_run_functions_eagerly(True)


def parse(record):
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


class DataLoader:
    def __init__(self, input_path):
        train_path = os.path.join(input_path, 'train.tfrecords')
        valid_path = os.path.join(input_path, 'valid.tfrecords')
        test_path = os.path.join(input_path, 'test.tfrecords')
        self.train_data = tf.data.TFRecordDataset(train_path).map(parse)
        self.valid_data = tf.data.TFRecordDataset(valid_path).map(parse).batch(merge_config.batch_size)
        self.test_data = tf.data.TFRecordDataset(test_path).map(parse).batch(merge_config.batch_size)
        return
    
    
class MergeTrain:
    def __init__(self, input_path):
        # 先载入 split 模型
        self.split_model = Split()
        print('Loading split model...')
        self.split_model.load_weights(merge_config.Split_Model_Path)
        print('Done')
        self.merge_model = Merge()
        if merge_config.con_train:
            self.merge_model.load_weights(os.path.join(merge_config.saved_models, 'merge_' + str(merge_config.start_epoch)))
            print('Loaded merge model from {}'.format(merge_config.saved_models + 'merge_' + str(merge_config.start_epoch)))
        
        DL = DataLoader(input_path)
        self.lr = merge_config.lr
        self.train_data = DL.train_data
        self.valid_data = DL.valid_data
        self.test_data = DL.test_data
        self.train()
        return
    
    def train(self):
        optimizer = tf.keras.optimizers.Adam(self.lr)
        summary_writer = tf.summary.create_file_writer(logdir=merge_config.log_dir)
        to_continue = []
        
        for epoch in range(merge_config.start_epoch, merge_config.train_epoch):
            # train
            dataset = self.train_data.shuffle(merge_config.shuffle_buffer).batch(merge_config.batch_size)
            for batch, (_image_batch, label_batch, char_mask, *_) in enumerate(dataset):
                image_batch = tf.image.convert_image_dtype(_image_batch, tf.float32)    # 线性缩放到0-1
                if len(to_continue):
                    if image_batch.shape == to_continue[0]:
                        continue
                if epoch == merge_config.start_epoch and batch == 0:
                    to_continue.append(image_batch.shape)
                try:
                    inputs, grid_loc = gen_merge_inputs(image_batch, self.split_model)
                except:
                    continue
                with tf.GradientTape() as tape:
                    suc, loss, accr, accc = loss_merge(inputs, grid_loc, char_mask, self.merge_model)    #
                # print(self.merge_model.variables)
                if not suc:
                    print('Continue...')
                    continue
                grads = tape.gradient(loss, self.merge_model.variables)

                if batch % merge_config.log_freq == 0:
                    print('Epoch :{}, batch: {}, loss = {}, accr = {}, accc = {}'.format(epoch, batch, tf.reduce_mean(loss), accr, accc))
                    with summary_writer.as_default():
                        tf.summary.scalar("loss", tf.reduce_mean(loss), step=epoch * merge_config.train_data_size + batch)
                        tf.summary.scalar("accr", accr, step=epoch * merge_config.train_data_size + batch)
                        tf.summary.scalar("accc", accc, step=epoch * merge_config.train_data_size + batch)
                        
                optimizer.apply_gradients(grads_and_vars=zip(grads, self.merge_model.variables))
            
                
            # save
            if epoch != merge_config.start_epoch and epoch % merge_config.save_freq == 0:  #
                self.merge_model.save_weights(os.path.join(merge_config.saved_models, 'merge_' + str(epoch)))    # yes
    
            # # valid
            # if epoch != merge_config.start_epoch and epoch % merge_config.valid_freq == 0:
            #     for batch, (_image_batch, label_batch, char_mask, *_) in enumerate(self.valid_data):
            #         image_batch = tf.image.convert_image_dtype(_image_batch, tf.float32)  # 线性缩放到0-1
            #         try:
            #             inputs, grid_loc = gen_merge_inputs(image_batch, self.split_model)
            #         except:
            #             continue
            #         suc, loss, accr, accc = loss_merge(inputs, grid_loc, char_mask, self.merge_model)
            #         if not suc:
            #             continue
            #         with summary_writer.as_default():
            #             tf.summary.scalar("accr_valid", accr, step=epoch * merge_config.valid_data_size + batch)
            #             tf.summary.scalar("accc_valid", accc, step=epoch * merge_config.valid_data_size + batch)
            #
            # # test
            # if epoch != merge_config.start_epoch and epoch % merge_config.test_freq == 0:
            #     for batch, (_image_batch, label_batch, char_mask, *_) in enumerate(self.test_data):
            #         image_batch = tf.image.convert_image_dtype(_image_batch, tf.float32)  # 线性缩放到0-1
            #         try:
            #             inputs, grid_loc = gen_merge_inputs(image_batch, self.split_model)
            #         except:
            #             continue
            #         suc, loss, accr, accc = loss_merge(inputs, grid_loc, char_mask, self.merge_model)
            #         if not suc:
            #             continue
            #         with summary_writer.as_default():
            #             tf.summary.scalar("accr_test", accr, step=epoch * merge_config.test_data_size + batch)
            #             tf.summary.scalar("accc_test", accc, step=epoch * merge_config.test_data_size + batch)
            #


if __name__ == '__main__':
    input_path = merge_config.tfrecords_data
    ST = MergeTrain(input_path)
