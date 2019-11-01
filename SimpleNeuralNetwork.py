import random
class NeuralNetwork:
    def __init__(self, inputNeurons, hiddenLayers, outputNeurons):
        self.inputNeurons = inputNeurons
        self.hiddenLayers=hiddenLayers
        self.outputNeurons=outputNeurons
        self.multiLayerPercepton = []
        self.buildMultiLayerPercepton()

    def buildMultiLayerPercepton(self):
        print("hi")

        if len(self.hiddenLayers) == 0:
            self.multiLayerPercepton.append(self.layer(self.inputNeurons, self.outputNeurons))
        else:
            self.multiLayerPercepton.append(self.layer(self.inputNeurons, self.hiddenLayers[0]))
            print(len(self.hiddenLayers))
            for i in range(1, len(self.hiddenLayers)):
                self.multiLayerPercepton.append(self.layer(self.hiddenLayers[i-1], self.hiddenLayers[i]))
            self.multiLayerPercepton.append(self.layer(self.hiddenLayers[-1], self.outputNeurons))


    def layer(self,numberOfInputs,numberOfOutputs):
        layer=[]
        for outputCounter in range(numberOfOutputs):
            thetas = [random.random() for i in range(numberOfInputs)]
            neuron = {"thetas":thetas,"output":None,"delta":None}
            layer.append(neuron)
        return layer


cls = NeuralNetwork(2,[3],2)
print(cls.multiLayerPercepton)


