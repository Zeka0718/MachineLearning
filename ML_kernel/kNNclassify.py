from numpy import *
import operator

def classify (inX, dataset, lables, k):
    datasetsize = dataset.shape[0]
    dis=tile(inX, (datasetsize, 1))-dataset
    dis=dis**2
    a=dis.sum(axis=1)
    a=a**0.5
    sortarray=a.argsort()
    classcount={}
    for i in range(k):
        votelable=lables[sortarray[i]]
        classcount[votelable]=classcount.get(votelable,0)+1
        sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]