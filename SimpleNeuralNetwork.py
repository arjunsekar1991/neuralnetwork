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
from sklearn.preprocessing import LabelBinarizer

warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn import metrics
class NeuralNetwork:
    def __init__(self,inputNeurons, hiddenLayers, outputNeurons):
        self.neuralNetwork = self.buildNetwork(inputNeurons, hiddenLayers, outputNeurons)

    def buildNetwork(self,inputNeurons, hiddenLayers, outputNeurons):
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

        for iterationCounter in range(MAX_ITER):
            for Xtrain_,YTrain_ in zip(Xtrain,YTrain):
                self.feedForwardPropagation(self.neuralNetwork, Xtrain_)
                oneHotEncodedOutput = self.oneHotEncoding(YTrain_,numberOfClassesToPredict)
                self.backPropagation(self.neuralNetwork, oneHotEncodedOutput)
                self.updateThetas(self.neuralNetwork, Xtrain_, alpha)


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
    seed(2)
    #below code is copied from internet
    def multiclass_roc_auc_score(self,y_test, y_pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)








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
hiddenLayers = [5]






clf = NeuralNetwork(numberofInputfeatures, hiddenLayers, numberofoutputFeatures)

clf.fit( XTrain, YTrain, alpha, maxIteration, numberofoutputFeatures)



predictionList = []
for XTest_,YTest_ in zip(XTest,YTest):
    prediction = clf.predict(XTest_)
    predictionList.append(prediction)

#print(YTest.values.tolist())
#print(predictionList)
#print(numpy.unique(YTrain))
print(classification_report(YTest.values.tolist(), predictionList, labels=numpy.unique(YTrain)))

sns.heatmap(confusion_matrix(YTest.values.tolist(), predictionList),annot=True)
plt.show()
#roc_auc_score(YTest.values.tolist(), predictionList)

#fpr, tpr, thresholds = metrics.roc_curve()
print(clf.multiclass_roc_auc_score(YTest.values.tolist(), predictionList))



