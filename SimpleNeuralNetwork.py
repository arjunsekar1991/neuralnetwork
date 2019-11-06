import random
from builtins import print
from math import exp
from random import seed
import numpy
import pandas
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def buildNetwork(inputNeurons, hiddenLayers, outputNeurons):
    neuralNetwork = list()
    inputLayer = [{'thetas':[random.random() for thetaCounter in range(inputNeurons + 1)]} for i in range(inputNeurons)]
    neuralNetwork.append(inputLayer)
    for index,hiddenNeurons in enumerate(hiddenLayers):
        #print("important logic",index,hiddenNeurons)
        if len(hiddenLayers) == 1:
            #print("this must not run")
            hiddenLayer = [{'thetas':[random.random() for thetaCounter in range(inputNeurons + 1)]} for i in range(hiddenNeurons)]
            neuralNetwork.append(hiddenLayer)
        else:
            if index==0:
                #print("index is zero")
                hiddenLayer = [{'thetas':[random.random() for thetaCounter in range(inputNeurons + 1)]} for i in range(hiddenNeurons)]
                neuralNetwork.append(hiddenLayer)
            else:
                hiddenLayer = [{'thetas':[random.random() for thetaCounter in range(hiddenLayers[index - 1] + 1)]} for i in range(hiddenNeurons)]
                neuralNetwork.append(hiddenLayer)
    outputLayer = [{'thetas':[random.random() for thetaCounter in range(hiddenLayers[-1] + 1)]} for i in range(outputNeurons)]
    neuralNetwork.append(outputLayer)

    return neuralNetwork





def feedForwardPropagation(neuralNetwork, X):
    for eachLayer in neuralNetwork:
        outputs = []
        for neurons in eachLayer:
            activation = calculateZ(neurons['thetas'], X)
            neurons['output'] = calculateSigmoid(activation)
            outputs.append(neurons['output'])
        X = outputs
    return X



def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        #print(layer)
        errors = list()

        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['thetas'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoidDerivative(neuron['output'])



def updateThetas(neuralNetwork, X, alpha):
    for layerCounter in range(len(neuralNetwork)):
        if layerCounter != 0:
            X = [neuron['output'] for neuron in neuralNetwork[layerCounter - 1]]
        for eachNeurons in neuralNetwork[layerCounter]:
            for eachInput in range(len(X)):
                eachNeurons['thetas'][eachInput] += alpha * eachNeurons['delta'] * X[eachInput]
            eachNeurons['thetas'][-1] += alpha * eachNeurons['delta']



def fit(neuralNetwork, Xtrain, YTrain, alpha, MAX_ITER, numberOfClassesToPredict):

    for iterationCounter in range(MAX_ITER):
        costDuringTraining = 0
        for Xtrain_,YTrain_ in zip(Xtrain,YTrain):
            outputs = feedForwardPropagation(neuralNetwork, Xtrain_)
            oneHotEncodedOutput = oneHotEncoding(YTrain_,numberOfClassesToPredict)
            costDuringTraining += sum([(oneHotEncodedOutput[encodedOutputCounter]-outputs[encodedOutputCounter])**2 for encodedOutputCounter in range(len(oneHotEncodedOutput))])
            backward_propagate_error(neuralNetwork, oneHotEncodedOutput)
            updateThetas(neuralNetwork, Xtrain_, alpha)


def oneHotEncoding(YTrain_,numberOfClassesToPredict):
    oneHotEncodedOutput = [0 for _ in range(numberOfClassesToPredict)]
    oneHotEncodedOutput[YTrain_] = 1
    return oneHotEncodedOutput











def calculateZ(thetas, inputs):
    z = thetas[-1]
    for i in range(len(thetas)-1):
        z += thetas[i] * inputs[i]
    return z

def calculateSigmoid(z):
    return 1.0 / (1.0 + exp(-z))


def sigmoidDerivative(output):
    return output * (1.0 - output)

def predict(neuralNetwork, X):
    softmaxPredictions = feedForwardPropagation(neuralNetwork, X)
    return softmaxPredictions.index(max(softmaxPredictions))
seed(2)









rawData = pandas.read_csv('BSOM_DataSet_for_HW3.csv')
dataWithColumnsRequired = rawData[['all_mcqs_avg_n20', 'all_NBME_avg_n4', 'CBSE_01', 'CBSE_02','LEVEL']]
dataWithColumnsRequiredWithoutNull = dataWithColumnsRequired.dropna(axis = 0, how ='any')


x = dataWithColumnsRequiredWithoutNull.drop('LEVEL',axis=1).values
x = (x- x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
ynonfactor = dataWithColumnsRequiredWithoutNull.LEVEL

y= dataWithColumnsRequiredWithoutNull.LEVEL.replace(to_replace=['A', 'B','C','D'], value=[0,1,2,3])


XTrain,XTest,YTrain,YTest = train_test_split(x,y,test_size=0.2,shuffle=False)
numberofInputs,numberofInputfeatures = XTrain.shape
numberofoutputFeatures =len(numpy.unique(y))

alpha = 0.1
maxIteration =1000
hiddenLayers = [5,7]







neuralNetwork = buildNetwork(numberofInputfeatures, hiddenLayers, numberofoutputFeatures)
fit(neuralNetwork, XTrain, YTrain, alpha, maxIteration, numberofoutputFeatures)



predictionList = []
for XTest_,YTest_ in zip(XTest,YTest):
    prediction = predict(neuralNetwork, XTest_)
    predictionList.append(prediction)

#print(YTest.values.tolist())
#print(predictionList)
#print(numpy.unique(YTrain))
print(classification_report(YTest.values.tolist(), predictionList, labels=numpy.unique(YTrain)))

sns.heatmap(confusion_matrix(YTest.values.tolist(), predictionList),annot=True)
plt.show()
