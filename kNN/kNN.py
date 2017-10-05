import numpy as np
import scipy.io as sio
import operator
import math

K=13  #for running in different parameters, modify this K

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1,length+1):
        #distance+=pow((instance1[x]-instance2[x]),2)
        distance += abs(instance1[x]-instance2[x])
    #return math.sqrt(distance)
    return distance
 
def getNeighbors(trainingSet, testInstance, k):
    distance=[]
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distance.append((trainingSet[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distance[x][0])
        return neighbors

def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1),
            reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][0]==predictions[x]:
            correct+=1
    return (correct/float(len(testSet))) *100

def main():
    # load the data
    mnist_data = sio.loadmat('mnist_data.mat')

    #seperate training data and test data
    train_array = np.array(mnist_data['train'])
    test_array = np.array(mnist_data['test'])
  
    #create random data
    random_numbers = np.random.choice(10000,100)
    test_data = test_array[random_numbers]
    
    predictions = []
    
    for x in range(len(test_data)):
        neighbors = getNeighbors(train_array, test_data[x], K)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted ='+repr(result) + ',actual=' +repr(test_data[x][0]))

    accuracy = getAccuracy(test_data, predictions)
    print('Accuracy:'+repr(accuracy)+'%')


if __name__ == "__main__":
    main()
