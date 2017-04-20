import tflearn
import numpy as np
import tarfile

from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression

#load data
(X_train,Y_train),(X_test,Y_test) = cifar10.load_data(dirname="./data/cifar10/cifar-10-batches-py", one_hot=True)
X_train,Y_train = shuffle(X_train,Y_train)

#data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_flip_leftright()

#build neural net
net = input_data(shape=[None,32,32,3],data_preprocessing=img_prep, data_augmentation=img_aug)

#layer 1
net = conv_2d(net,32,3,activation='relu')
net = max_pool_2d(net,2)

#layer 2
net = conv_2d(net,64,3,activation='relu')

#layer 3
net = conv_2d(net,64,3,activation='relu')
net = max_pool_2d(net,2)

#layer 4
net = fully_connected(net,512,activation='relu')
net = dropout(net,0.5)

net = fully_connected(net,10,activation='softmax')
net = regression(net,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001,batch_size=64)


#create model
model = tflearn.DNN(net)

#now train the model
#model.fit(X_train,Y_train,n_epoch=50, validation_set=(X_test,Y_test),show_metric=True, batch_size=96,run_id='cifar10')

#save the progress
#model.save('cifar10.tf1')

#print "Cifar 10 network is complete! trained model is saved as cifar10.tf1"

#load the trained model
model.load('./checkpoints/cifar10/cifar10.tf1')

correct = 0.
total_testset = len(X_test)

for i in range(total_testset):
    predictedY = np.argmax(model.predict([X_test[i]]))
    actualY = np.argmax(Y_test[i])
    if(predictedY == actualY):
        correct += 1

print 'accuracy: '+str(correct/total_testset)
#accuracy: 0.8124