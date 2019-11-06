import numpy
import pandas
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from builtins import print
from random import seed
from SimpleNeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,f1_score
rawData = pandas.read_csv('BSOM_DataSet_for_HW3.csv')
dataWithColumnsRequired = rawData[['all_mcqs_avg_n20', 'all_NBME_avg_n4', 'CBSE_01', 'CBSE_02','LEVEL']]
dataWithColumnsRequiredWithoutNull = dataWithColumnsRequired.dropna(axis = 0, how ='any')
from random import seed
#seed(2)
x = dataWithColumnsRequiredWithoutNull.drop('LEVEL',axis=1).values
x = (x- x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
ynonfactor = dataWithColumnsRequiredWithoutNull.LEVEL

y= dataWithColumnsRequiredWithoutNull.LEVEL.replace(to_replace=['A', 'B','C','D'], value=[0,1,2,3])


XTrain,XTest,YTrain,YTest = train_test_split(x,y,test_size=0.2,shuffle=False)
numberofInputs,numberofInputfeatures = XTrain.shape
numberofoutputFeatures =len(numpy.unique(y))

alpha = 0.1
maxIteration =1000
hiddenLayers = [[3],[5],[17],[32]]





for differentConfigurations in hiddenLayers:
    seed(2)
    print(differentConfigurations)
    clf = NeuralNetwork(numberofInputfeatures, differentConfigurations, numberofoutputFeatures)

    clf.fit( XTrain, YTrain, alpha, maxIteration, numberofoutputFeatures)



    predictionList = []
    for XTest_,YTest_ in zip(XTest,YTest):
        prediction = clf.predict(XTest_)
        predictionList.append(prediction)

    #print(YTest.values.tolist())
    #print(predictionList)
    #print(numpy.unique(YTrain))
    #clf.multiclassROCPlot(predictionList,YTest)
    #print('F1 score :',f1_score(yt,yp,average='weighted'))
    print('F1',f1_score(YTest,predictionList,average='weighted'))
    print('Precision',precision_score(YTest,predictionList,average='weighted'))
    print('Recall:',recall_score(YTest,predictionList,average='weighted'))
    print(classification_report(YTest.values.tolist(), predictionList, labels=numpy.unique(YTrain)))

    fig, ax = plt.subplots(figsize=(8,8))
    ax = fig.add_axes([0.4,0.2,0.5,0.6])
    ax2=sns.heatmap(confusion_matrix(YTest, predictionList), annot=True, fmt='g', yticklabels=numpy.unique(YTrain), xticklabels=numpy.unique(YTrain), ax=ax, linewidths=0.1, square=True);
    bottom, top = ax2.get_ylim()
    ax2.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    #plt.show()
    #roc_auc_score(YTest.values.tolist(), predictionList)

    #fpr, tpr, thresholds = metrics.roc_curve()
    print(clf.multiclass_roc_auc_score(YTest.values.tolist(), predictionList))
