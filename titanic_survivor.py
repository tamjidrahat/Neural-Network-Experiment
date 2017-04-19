import tensorflow as tf
import numpy as np
import tflearn
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv
#titanic.download_dataset('titanic_dataset.csv')



data,labels = load_csv('./data/titanic/titanic_dataset.csv',target_column=0,categorical_labels=True,n_classes=2)


def data_preprocess(data, col_to_ignore):
    for col in sorted(col_to_ignore,reverse=True):
        [row.pop(col) for row in data]
    for i in range(len(data)):
        data[i][1] = 1.0 if data[i][1] == 'female' else 0.0
    return np.array(data,dtype=np.float32)

col_to_ignore = [1,6]
data = data_preprocess(data,col_to_ignore)



#build network
network = tflearn.input_data(shape=[None,6])
network = tflearn.fully_connected(network,32)
network = tflearn.fully_connected(network,32)
network = tflearn.fully_connected(network,2,activation='softmax')
network = tflearn.regression(network)

model = tflearn.DNN(network)
model.fit(data,labels,n_epoch=10,batch_size=16,show_metric=True)


# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = data_preprocess([dicaprio, winslet], col_to_ignore)

pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", str(np.argmax(pred[0])))
print("Winslet Surviving Rate:", str(np.argmax(pred[1])))
