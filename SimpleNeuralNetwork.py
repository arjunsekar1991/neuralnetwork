import random
from sklearn.preprocessing import LabelBinarizer
from math import exp
from random import seed
import warnings
import numpy
import pandas
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from builtins import print

import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn import metrics
class NeuralNetwork:
    def __init__(self,inputNeurons, hiddenLayers, outputNeurons):
        self.neuralNetwork = self.buildNetwork(inputNeurons, hiddenLayers, outputNeurons)
        random.seed(2)

    def buildNetwork(self,inputNeurons, hiddenLayers, outputNeurons):
        neuralNetwork = list()
        inputLayer = [{'thetas':[random.random()   for thetaCounter in range(inputNeurons + 1)]} for i in range(inputNeurons)]
        neuralNetwork.append(inputLayer)
        for index,hiddenNeurons in enumerate(hiddenLayers):
            #print("important logic",index,hiddenNeurons)
            if len(hiddenLayers) == 1:
                #print("this must not run")
                hiddenLayer = [{'thetas':[random.random()   for thetaCounter in range(inputNeurons + 1)]} for i in range(hiddenNeurons)]
                neuralNetwork.append(hiddenLayer)
            else:
                if index==0:
                    #print("index is zero")
                    hiddenLayer = [{'thetas':[random.random()   for thetaCounter in range(inputNeurons + 1)]} for i in range(hiddenNeurons)]
                    neuralNetwork.append(hiddenLayer)
                else:
                    hiddenLayer = [{'thetas':[random.random()  for thetaCounter in range(hiddenLayers[index - 1] + 1)]} for i in range(hiddenNeurons)]
                    neuralNetwork.append(hiddenLayer)
        outputLayer = [{'thetas':[random.random()   for thetaCounter in range(hiddenLayers[-1] + 1)]} for i in range(outputNeurons)]
        neuralNetwork.append(outputLayer)

        return neuralNetwork





    def feedForwardPropagation(self,neuralNetwork, X):
        for eachLayer in neuralNetwork:
            outputs = []
            for neurons in eachLayer:
                Z = self.calculateZ (neurons['thetas'], X)
                neurons['output'] = self.calculateSigmoid(Z)
                outputs.append(neurons['output'])
            X = outputs
        return X



    def backPropagation(self, network, expected):
        for layerCounter in reversed(range(len(network))):
            layer = network[layerCounter]

            cost = list()

            if layerCounter != len(network)-1:
                for layerIterator in range(len(layer)):
                    smallDelta = 0
                    for neuron in network[layerCounter + 1]:
                        smallDelta += (neuron['thetas'][layerIterator] * neuron['delta'])
                    cost.append(smallDelta)
            else:
                for layerIterator in range(len(layer)):
                    neuron = layer[layerIterator]
                    cost.append(expected[layerIterator] - neuron['output'])

            for layerIterator in range(len(layer)):
                neuron = layer[layerIterator]
                neuron['delta'] = cost[layerIterator] * self.sigmoidDerivative(neuron['output'])



    def updateThetas(self,neuralNetwork, X, alpha):
        for layerCounter in range(len(neuralNetwork)):
            if layerCounter != 0:
                X = [neuron['output'] for neuron in neuralNetwork[layerCounter - 1]]
            for eachNeurons in neuralNetwork[layerCounter]:
                for eachInput in range(len(X)):
                    eachNeurons['thetas'][eachInput] = eachNeurons['thetas'][eachInput]+alpha * eachNeurons['delta'] * X[eachInput]
                eachNeurons['thetas'][-1] = eachNeurons['thetas'][-1] + alpha * eachNeurons['delta']



    def fit(self, Xtrain, YTrain, alpha, MAX_ITER, numberOfClassesToPredict):
        costList=[]

        for iterationCounter in range(MAX_ITER):
            cost=0.0
            for Xtrain_,YTrain_ in zip(Xtrain,YTrain):
                self.feedForwardPropagation(self.neuralNetwork, Xtrain_)
                oneHotEncodedOutput = self.oneHotEncoding(YTrain_,numberOfClassesToPredict)
                self.backPropagation(self.neuralNetwork, oneHotEncodedOutput)
                self.updateThetas(self.neuralNetwork, Xtrain_, alpha)
            #print("index",iterationCounter)
            #print("cost",cost)
            costList.append(cost)
            #print("current",costList[iterationCounter])

            #costList[iterationCounter]=cost
            #if iterationCounter != 0:
                #print("previous",costList[iterationCounter-1])
                #print("difference",costList[iterationCounter-1]-costList[iterationCounter])
            #    if costList[iterationCounter]>costList[iterationCounter-1] or costList[iterationCounter-1]-costList[iterationCounter] <0.001 :
                    #print("difference2",costList[iterationCounter]-costList[iterationCounter-1])
            #        break

    def oneHotEncoding(self,YTrain_,numberOfClassesToPredict):
        oneHotEncodedOutput = [0 for _ in range(numberOfClassesToPredict)]
        oneHotEncodedOutput[YTrain_] = 1
        return oneHotEncodedOutput

    def calculateZ(self,thetas, inputs):
        z = thetas[-1]
        for i in range(len(thetas)-1):
            z += thetas[i] * inputs[i]
        return z

    def calculateSigmoid(self,z):
        return 1.0 / (1.0 + exp(-z))


    def sigmoidDerivative(self,output):
        return output * (1.0 - output)

    def predict(self, X):
        softmaxPredictions = self.feedForwardPropagation(self.neuralNetwork, X)
        return softmaxPredictions.index(max(softmaxPredictions))

    #below code is copied from internet
    def multiclass_roc_auc_score(self,y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)










