import tflearn
import numpy as np
import scipy.io as sio

from tflearn.data_utils import shuffle,to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression

#load data

def load_data():
    dict_train = sio.loadmat('./data/svhn/train_32x32.mat')
    dict_test = sio.loadmat('./data/svhn/test_32x32.mat')

    X = np.array(dict_train['X'])
    X_train = []
    for i in xrange(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.array(X_train,dtype=np.float32)

    Y_train = dict_train['y']
    for i in xrange(len(Y_train)):
        if(Y_train[i] == 10):
            Y_train[i] = 0
    Y_train = to_categorical(Y_train,10)

    X = np.array(dict_test['X'])
    X_test = []
    for i in xrange(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.array(X_test,dtype=np.float32)

    Y_test = dict_test['y']
    for i in xrange(len(Y_test)):
        if (Y_test[i] == 10):
            Y_test[i] = 0
    Y_test = to_categorical(Y_test, 10)

    return (X_train,Y_train),(X_test,Y_test)






(X_train,Y_train),(X_test,Y_test) = load_data()
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
#model.fit(X_train,Y_train,n_epoch=20, validation_set=(X_test,Y_test),show_metric=True, batch_size=96,run_id='svhn')

#save the progress
#model.save('svhn_cnn')

#print "SVHN network is complete! trained model is saved as svhn_cnn"


#load the trained model
model.load('./checkpoints/svhn/svhn_cnn.tf1')
print "trained model is loaded"
correct = 0.
total_testset = len(X_test)

for i in range(total_testset):
    predictedY = np.argmax(model.predict([X_test[i]]))
    actualY = np.argmax(Y_test[i])
    if(predictedY == actualY):
        correct += 1

print 'accuracy: '+str(correct/total_testset)
#accuracy: 0.901236939152