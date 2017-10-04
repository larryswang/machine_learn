import cv2
import numpy as np
import scipy.io as sio

K=1
# load the data
mnist_data = sio.loadmat('mnist_data.mat')

#seperate training data and test data
train_array = np.array(mnist_data['train'])
test_array = np.array(mnist_data['test'])
train_data = train_array[:,1:785].astype(np.float32)
train_label = train_array[:,0].astype(np.float32)

#create random data
random_numbers = np.random.choice(10000,100)
random_data = test_array[random_numbers].astype(np.float32)
test_data = random_data[:, 1:785]
test_label = random_data[:,0]

#knn
knn=cv2.KNearest()
knn.train(train_data, train_label)

ret, results, neighbors, dist = knn.find_nearest(test_data, K)

#calculate accuracy
count = 0
for i in range(0,100):
    if results[i]==test_label[i]:
        count=count+1

print count,"%"
