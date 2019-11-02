import random
import numpy
import math
class NeuralNetwork:
    def __init__(self, inputNeurons, hiddenLayers, outputNeurons,lamda):
        self.inputNeurons = inputNeurons
        self.hiddenLayers=hiddenLayers
        self.outputNeurons=outputNeurons
        self.multiLayerPercepton = []
        self.buildMultiLayerPercepton()
        self.numberOfLayers = len(self.multiLayerPercepton)
        self.lamda =lamda
        self.m=None
    def buildMultiLayerPercepton(self):
        print("hi")

        if len(self.hiddenLayers) == 0:
            self.multiLayerPercepton.append(self.layer(self.inputNeurons, self.outputNeurons))
        else:
            self.multiLayerPercepton.append(self.layer(self.inputNeurons, self.hiddenLayers[0]))
            print(len(self.hiddenLayers))
            for hiddenLayerCounter in range(1, len(self.hiddenLayers)):
                self.multiLayerPercepton.append(self.layer(self.hiddenLayers[hiddenLayerCounter-1], self.hiddenLayers[hiddenLayerCounter]))
            self.multiLayerPercepton.append(self.layer(self.hiddenLayers[-1], self.outputNeurons))

    def dotProduct(self, thetas, input):
        return sum([thetas_ * input_ for thetas_, input_ in zip(thetas, input)])
    def layer(self,numberOfInputs,numberOfOutputs):
        layer=[]
        for outputCounter in range(numberOfOutputs):
            thetas = [random.random() for i in range(numberOfInputs)]
            neuron = {"thetas":thetas,"output":None,"delta":None}
            layer.append(neuron)
        return layer
    def feedForward(self, inputData):


        for layer in self.multiLayerPercepton:
            output = []
            for node in layer:
                node['output'] = self.sigmoid(self.dotProduct(node['thetas'], inputData))
                output.append(node['output'])
            inputData = output # set output as next input
        return inputData

    def fit(self,XTrain,YTrain,learningRate=0.5,maxIteration=1):
        self.m=len(XTrain)
        for maxIterationCounter in range(maxIteration):
            for XTrain_,YTrain_ in zip(XTrain,YTrain):
                self.feedForward(XTrain_)
                self.backPropagate(self.oneHotEncoding(YTrain_,self.outputNeurons))
                self.updateThetas(XTrain_, learningRate)
    def sigmoid(self, input):
        input = float(input)
        return 1/(1+numpy.exp(-input))

    def sigmoidDerivative(self, input):
        return input*(1.0-input)

    def oneHotEncoding(self, trueClassLabel, output):
        oneHotEncodedOutput = numpy.zeros(output, dtype=numpy.int)
        #setting only the true class labels as one rest all are zero
        oneHotEncodedOutput[trueClassLabel] = 1
        return oneHotEncodedOutput
    def backPropagate(self,oneHotEncodedClassLabel):


        for layerCounter in reversed(range(self.numberOfLayers)):
            if layerCounter == self.numberOfLayers - 1:

                for index, neuron in enumerate(self.multiLayerPercepton[layerCounter]):
                    cost = neuron['output'] - oneHotEncodedClassLabel[index]
                    neuron['delta'] = cost * self.sigmoidDerivative(neuron['output'])
            else:

                for index, neuron in enumerate(self.multiLayerPercepton[layerCounter]):
                    cost = sum([neurons_['thetas'][index] * neurons_['delta'] for neurons_ in self.multiLayerPercepton[layerCounter+1]])+(self.lamda/self.m)*neuron['thetas'][index]
                    neuron['delta'] = cost * self.sigmoidDerivative(neuron['output'])


    def updateThetas(self, x, eta):
        for index, neurons in enumerate(self.multiLayerPercepton):

            if index == 0:
                inputs = x
            else:
                inputs = [neurons_['output'] for neurons_ in self.multiLayerPercepton[index-1]]

            for neuron in neurons:
                for inputCounter, input in enumerate(inputs):

                     neuron['thetas'][inputCounter] += - eta * neuron['delta'] * input
    def predict(self, yActual):
        ypred = numpy.array([numpy.argmax(self.feedForward(input_)) for input_ in yActual], dtype=numpy.int)
        return ypred
#cls = NeuralNetwork(2,[3],2)
#print(cls.multiLayerPercepton)


