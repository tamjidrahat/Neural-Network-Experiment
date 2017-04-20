import tflearn
import numpy as np
import tarfile

from tflearn.datasets import oxflower17
from tflearn.data_utils import shuffle
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

X, Y = oxflower17.load_data('./data/oxflower17',one_hot=True,resize_pics=(227,227))



#build alexnet

net = input_data(shape=[None,227,227,3])
net = conv_2d(net,96,11,strides=4,activation='relu')
net = max_pool_2d(net,3,strides=2)
net = local_response_normalization(net)

net = conv_2d(net,256,5,activation='relu')
net = max_pool_2d(net,3,strides=2)
net = local_response_normalization(net)

net = conv_2d(net,384,3,activation='relu')

net = conv_2d(net,384,3,activation='relu')

net = conv_2d(net,256,3,activation='relu')
net = max_pool_2d(net,3,strides=2)
net = local_response_normalization(net)

net = fully_connected(net,4096, activation='tanh')
net = dropout(net,0.5)

net = fully_connected(net,4096, activation='tanh')
net = dropout(net,0.5)

net = fully_connected(net,17,activation='softmax')

net = regression(net,optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)


model = tflearn.DNN(net,tensorboard_verbose=2,checkpoint_path='./checkpoints/oxflowers17',max_checkpoints=1)
model.fit(X,Y,n_epoch=1000,validation_set=0.1,show_metric=True,batch_size=64,shuffle=True,snapshot_epoch=False
          ,snapshot_step=200,run_id='oxflowers17_alexnet')

model.save('./checkpoints/oxflowers17/oxflowers17.tf1')

