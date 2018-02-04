from numpy import *
import csv
from ML_kernel.LogisticRegression import gradient

list1 = []
list2 = []

with open('/Users/luoyu/Downloads/train.csv','rt') as myFile:
    lines = csv.reader(myFile)
    for line in lines:
        if line[0] != 'label':
            list1.append(line[0])
            list2.append(line[1:len(line)])

testLable = array(list1[0:8400]).astype('Float64')
testInput = array(list2[0:8400]).astype('Float64')
trainLable = array(list1[8400:42000]).astype('Float64')
trainInput = array(list2[8400:42000]).astype('Float64')

weights = gradient(trainInput, trainLable, 0.1, 500)
