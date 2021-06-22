#-*-coding:utf-8-*-
import os
import tensorflow as tf
from config import split_config
from gen_ground_truth.gen_gt import GenGroundTruth as GGT
from model.split import Split
from model.loss import loss_split
tf.config.experimental_run_functions_eagerly(True)


class DataLoader:
    def __init__(self, input_path):
        train_path = os.path.join(input_path, 'train.tfrecords')
        valid_path = os.path.join(input_path, 'valid.tfrecords')
        test_path = os.path.join(input_path, 'test.tfrecords')
        self.train_data = tf.data.TFRecordDataset(train_path).map(GGT.parse)
        self.valid_data = tf.data.TFRecordDataset(valid_path).map(GGT.parse).batch(split_config.batch_size)
        self.test_data = tf.data.TFRecordDataset(test_path).map(GGT.parse).batch(split_config.batch_size)
        return
    
    
class SplitTrain:
    def __init__(self, input_path):
        self.split_model = Split()
        DL = DataLoader(input_path)
        self.lr = split_config.lr
        self.train_data = DL.train_data
        self.valid_data = DL.valid_data
        self.test_data = DL.test_data
        self.train()
        return
    
    def train(self):
        optimizer = tf.keras.optimizers.Adam(self.lr)
        summary_writer = tf.summary.create_file_writer(logdir=split_config.log_dir)

        if split_config.con_train:
            self.split_model.load_weights(split_config.saved_models + 'split_' + str(split_config.start_epoch))
        to_continue = []
        
        for epoch in range(split_config.start_epoch, split_config.train_epoch):
            # train
            with summary_writer.as_default():
                tf.summary.trace_on(graph=True, profiler=True)
            dataset = self.train_data.shuffle(split_config.shuffle_buffer).batch(split_config.batch_size)
            for batch, (_image_batch, label_batch, *_) in enumerate(dataset):
                image_batch = tf.image.convert_image_dtype(_image_batch, tf.float32)    # 线性缩放到0-1
                if len(to_continue):
                    if image_batch.shape == to_continue[0]:
                        continue
                if epoch == split_config.start_epoch and batch == 0:
                    to_continue.append(image_batch.shape)
                    
                with tf.GradientTape() as tape:
                    loss, accr, accc = loss_split(image_batch, label_batch, self.split_model)

                grads = tape.gradient(loss, self.split_model.variables)
                if batch % split_config.log_freq == 0:
                    print('Epoch :{}, batch: {}, loss = {}, accr = {}, accc = {}'.format(epoch, batch, tf.reduce_mean(loss), accr, accc))
                    with summary_writer.as_default():
                        tf.summary.scalar("loss", tf.reduce_mean(loss), step=epoch * split_config.train_data_size + batch)
                        tf.summary.scalar("accr", accr, step=epoch * split_config.train_data_size + batch)
                        tf.summary.scalar("accc", accc, step=epoch * split_config.train_data_size + batch)
                optimizer.apply_gradients(grads_and_vars=zip(grads, self.split_model.variables))
            with summary_writer.as_default():
                tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=split_config.log_dir)
            
            # save
            if epoch != split_config.start_epoch and epoch % split_config.save_freq == 0:  #
                self.split_model.save_weights(os.path.join(split_config.saved_models, 'split_' + str(epoch)))
    
            # valid
            if epoch != split_config.start_epoch and epoch % split_config.valid_freq == 0:
                for batch, (_image_batch, label_batch, *_) in enumerate(self.valid_data):
                    image_batch = tf.image.convert_image_dtype(_image_batch, tf.float32)
                    loss, accr, accc = loss_split(image_batch, label_batch, self.split_model)
                    with summary_writer.as_default():
                        tf.summary.scalar("accr_valid", accr, step=epoch * split_config.valid_data_size + batch)
                        tf.summary.scalar("accc_valid", accc, step=epoch * split_config.valid_data_size + batch)

            # test
            if epoch != split_config.start_epoch and epoch % split_config.test_freq == 0:
                for batch, (_image_batch, label_batch, *_) in enumerate(self.test_data):
                    image_batch = tf.image.convert_image_dtype(_image_batch, tf.float32)
                    loss, accr, accc = loss_split(image_batch, label_batch, self.split_model)
                    with summary_writer.as_default():
                        tf.summary.scalar("accr_test", accr, step=epoch * split_config.test_data_size + batch)
                        tf.summary.scalar("accc_test", accc, step=epoch * split_config.test_data_size + batch)
            

if __name__ == '__main__':
    input_path = split_config.tfrecords_data
    ST = SplitTrain(input_path)