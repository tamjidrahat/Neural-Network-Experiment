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

X, Y = oxflower17.load_data('./data/oxflower17',one_hot=True)

#build VGG-16 net

net = input_data(shape=[None,227,227,3])

net = conv_2d(net, 64, 3, activation='relu')
net = conv_2d(net, 64, 3, activation='relu')
net = max_pool_2d(net, 2, strides=2)

net = conv_2d(net, 128,3, activation='relu')
net = conv_2d(net, 128,3, activation='relu')
net = max_pool_2d(net, 2, strides=2)

net = conv_2d(net, 256,3, activation='relu')
net = conv_2d(net, 256,3, activation='relu')
net = conv_2d(net, 256,3, activation='relu')
net = max_pool_2d(net, 2, strides=2)

net = conv_2d(net, 512,3, activation='relu')
net = conv_2d(net, 512,3, activation='relu')
net = conv_2d(net, 512,3, activation='relu')
net = max_pool_2d(net, 2, strides=2)

net = conv_2d(net, 512,3, activation='relu')
net = conv_2d(net, 512,3, activation='relu')
net = conv_2d(net, 512,3, activation='relu')
net = max_pool_2d(net, 2, strides=2)

net = fully_connected(net,4096,activation='relu')
net = dropout(net,0.5)

net = fully_connected(net,4096,activation='relu')
net = dropout(net,0.5)

net = fully_connected(net,17,activation='softmax')
net = dropout(net,0.5)

net = regression(net, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0001)

model = tflearn.DNN(net,tensorboard_verbose=2,checkpoint_path='./checkpoints/oxflowers17_vgg16/oxflowers17.tf1',max_checkpoints=1)

#model.load('./checkpoints/oxflowers17_vgg16/oxflowers17.tf1-40')

model.fit(X,Y,n_epoch=500,show_metric=True,batch_size=32,shuffle=True,snapshot_epoch=False
          ,snapshot_step=10,run_id='oxflowers17_vgg16')