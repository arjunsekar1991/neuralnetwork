class NeuralNetwork:
    def __init__(self, inputNeurons, hiddenLayers, outputNeurons):
        self.inputNeurons = inputNeurons
        self.hiddenLayers=hiddenLayers
        self.outputNeurons=outputNeurons
        self.multiLayerPercepton = multiLayerPercepton()


