# -*- coding: UTF-8 -*-

import csv

from numpy import *

from ML_kernel.kNNclassify import classify

with open('/Users/luoyu/Downloads/train.csv','rt') as myFile:
    lines = csv.reader(myFile)
    list1 = []
    list2 = []
    for line in lines:
        if line[0] != 'label':
            list1.append(line[0])
            list2.append(line[1:len(line)])


    arrayLable = array(list1[0:8400]).astype('Float64')
    arrayLocation = array(list2[0:8400]).astype('Float64')
    arraytrain1 = array(list1[8400:42000]).astype('Float64')
    arraytrain2 = array(list2[8400:42000]).astype('Float64')


    yes=0
    for i in range(8400):
        y = classify(arrayLocation[i], arraytrain2, arraytrain1, 5)
        str='The indicated number is (%d). The real number is (%d)' %(y,arrayLable[i])
        print(str)
        if(y==arrayLable[i]):
            print('correct')
            yes+=1
        else:
            print('wrong!!!!')
    print(yes/8400)





