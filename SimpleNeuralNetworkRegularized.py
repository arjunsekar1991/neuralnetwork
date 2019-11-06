import random
from builtins import print

from random import seed
import numpy
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


rawData = pandas.read_csv('BSOM_DataSet_for_HW3.csv')
dataWithColumnsRequired = rawData[['all_mcqs_avg_n20', 'all_NBME_avg_n4', 'CBSE_01', 'CBSE_02','LEVEL']]
dataWithColumnsRequiredWithoutNull = dataWithColumnsRequired.dropna(axis = 0, how ='any')


x = dataWithColumnsRequiredWithoutNull.drop('LEVEL',axis=1).values
x = (x- x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
ynonfactor = dataWithColumnsRequiredWithoutNull.LEVEL

y= dataWithColumnsRequiredWithoutNull.LEVEL.replace(to_replace=['A', 'B','C','D'], value=[0,1,2,3])

#print()
XTrain,XTest,YTrain,YTest = train_test_split(x,y,test_size=0.2,shuffle=False)
print(XTrain.shape,XTest.shape)


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    input_layer = [{'weights':[random.uniform(-0.5,0.5) for i in range(n_inputs + 1)]} for i in range(n_inputs)]
    network.append(input_layer)
    for index,x in enumerate(n_hidden):
        print("important logic",index,x)
        if len(n_hidden) == 1:
            print("this must not run")
            hidden_layer = [{'weights':[random.uniform(-0.5,0.5) for i in range(n_inputs + 1)]} for i in range(x)]
            network.append(hidden_layer)
        else:
            if index==0:
                print("index is zero")
                hidden_layer = [{'weights':[random.uniform(-0.5,0.5) for i in range(n_inputs + 1)]} for i in range(x)]
                network.append(hidden_layer)
            else:
                hidden_layer = [{'weights':[random.uniform(-0.5,0.5) for i in range(n_hidden[index-1] + 1)]} for i in range(x)]
                network.append(hidden_layer)
    output_layer = [{'weights':[random.uniform(-0.5,0.5) for i in range(n_hidden[-1] + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    i= 1
    print("\n The initialised Neural Network:\n")
    for layer in network:
        j=1
        for sub in layer:
            print("\n Layer[%d] Node[%d]:\n" %(i,j),sub)
            j=j+1
        i=i+1
    return network

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected,n):
    for i in reversed(range(len(network))):
        layer = network[i]
        #print(layer)
        errors = list()

        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

            if i == len(network)-1 :
                if regulerization == 'L2':
                    rg = np.sum(network[i][j]['weights']) / n
                if regulerization == 'L1':
                    rg = np.sum(np.where(np.array(network[i][j]['weights'])<0,-1,1)) / n
                neuron['delta'] += lamda * rg


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, Xtrain,YTrain, l_rate, n_epoch, n_outputs):

    print("\n Network Training Begins:\n")

    for epoch in range(n_epoch):
        sum_error = 0
        n = len(Xtrain)
        for Xtrain_,YTrain_ in zip(Xtrain,YTrain):
            outputs = forward_propagate(network, Xtrain_)
            expected = [0 for i in range(n_outputs)]
            expected[YTrain_] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected,n)
            update_weights(network, Xtrain_, l_rate)
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

    print("\n Network Training Ends:\n")




from math import exp

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for i,neuron in enumerate(layer):
            neuron = layer[i]
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    inputs = np.array(inputs)
    #inputs = np.exp(inputs)/sum(np.exp(inputs))
    return inputs

# Make a prediction with a network
def predict(network, XTest):
    outputs = forward_propagate(network, XTest)
    return list(outputs).index(max(outputs))



#Test training backprop algorithm
seed(2)

numberofInputs,numberofInputfeatures = XTrain.shape
numberofoutputs =4
lamda = 1
regulerization = 'L2'
n_inputs = numberofInputfeatures
print("\n Number of Inputs :\n",n_inputs)
n_outputs = numberofoutputs
print("\n Number of Outputs :\n",n_outputs)

#Network Initialization
network = initialize_network(n_inputs, [5,6], n_outputs)

# Training the Network
train_network(network, XTrain,YTrain, 0.1, 1000, n_outputs)

print("\n Final Neural Network :")

i= 1
for layer in network:
    j=1
    for sub in layer:
        print("\n Layer[%d] Node[%d]:\n" %(i,j),sub)
        j=j+1
    i=i+1
    

predictionList = []
for XTest_,YTest_ in zip(XTest,YTest):
    prediction = predict(network, XTest_)
    predictionList.append(prediction)
    print('Expected=%d, Got=%d' % (YTest_, prediction))
print(YTest.values.tolist())
print(predictionList)
print(numpy.unique(YTrain))
print(classification_report(YTest.values.tolist(), predictionList, labels=numpy.unique(YTrain)))
print()
print(sns.heatmap(confusion_matrix(YTest.values.tolist(), predictionList),annot=True));
plt.show()
