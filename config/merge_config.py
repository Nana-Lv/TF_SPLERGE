#-*-coding:utf-8-*-
import os
root_dir = '/home/yx-lxd/PycharmProjects/TF_SPLERGE'
# prepare tfrecords
original_data = os.path.join(root_dir, 'splerge_data')
tfrecords_data = os.path.join(root_dir, 'tfrd_data_merge')
if not os.path.exists(tfrecords_data):
    os.mkdir(tfrecords_data)
train_data_name = 'train.tfrecords'
valid_data_name = 'valid.tfrecords'
test_data_name = 'test.tfrecords'


# about training
Split_Model_Path = os.path.join(root_dir, 'final_models/split_499')
log_dir = os.path.join(root_dir, 'merge_tensorboard')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
saved_models = os.path.join(root_dir, 'merge_saved_models')
if not os.path.exists(saved_models):
    os.mkdir(saved_models)
lr = 0.00075 * 0.7 * 0.8 # 0.00075

start_epoch = 1092
con_train = True if start_epoch != 0 else False

train_epoch = 2000
log_freq = 1

# about batch
train_data_size = 161  # need checked if you change the training data
shuffle_buffer = train_data_size
batch_size = 1
batch_num = train_data_size // batch_size

valid_data_size = 9
test_data_size = 22


# display
save_freq = 1
valid_freq = 2
test_freq = 2
