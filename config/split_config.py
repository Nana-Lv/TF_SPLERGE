#-*-coding:utf-8-*-
import os
root_dir = '/home/yx-lxd/PycharmProjects/TF_SPLERGE'
# prepare tfrecords

original_data = os.path.join(root_dir, 'splerge_data')
tfrecords_data = os.path.join(root_dir, 'tfrd_data_split')
train_data_name = 'train.tfrecords'
valid_data_name = 'valid.tfrecords'
test_data_name = 'test.tfrecords'

# about training
log_dir = os.path.join(root_dir, 'split_tensorboard')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
saved_models = os.path.join(root_dir, 'split_saved_models')
if not os.path.exists(saved_models):
    os.mkdir(saved_models)
lr = 0.00075  # 0.00075

start_epoch = 0
con_train = True if start_epoch != 0 else False

train_epoch = 500
log_freq = 1

# about batch
train_data_size = 224  # need checked if you change the training data
shuffle_buffer = train_data_size
batch_size = 1
batch_num = train_data_size // batch_size

valid_data_size = 22
test_data_size = 22


# display
save_freq = 1
valid_freq = 2
test_freq = 2
