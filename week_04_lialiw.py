# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 18:38:07 2021

@author: Fik
"""
"""
Sources:
    https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html
    https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
    https://www.w3schools.com/python/python_dictionaries_change.asp
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    
    
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from math import sqrt
   
    
def separate_X_y(df_xValues, y_name, attrDrop, flag_to_numpy):

    df_xValues.drop(columns=attrDrop, inplace=True) #Dropping columns irrelevant to dataset's analysis
    df_yValue = df_xValues[y_name]
    df_xValues.drop(labels=y_name, axis=1, inplace=True)
    #print(df_yValue.name)
    #print(df_xValues.columns)
    if flag_to_numpy==1:
        return df_xValues.to_numpy(), df_yValue.to_numpy()
    else:
        #print(flag_to_numpy)
        return df_xValues, df_yValue

    
def divide_dataframe_onAttr(df, attr, value):
    df_valueOff = df.drop(df[ (df[attr]==value) ].index)
    df_valueOn = df.drop(df_valueOff.index)
    return df_valueOn, df_valueOff


def createAttrSample_train_test(df, attr_name, attr_value, cutPerc):
    '''
    Distributive property is used to create training and testing sets
    '''
    df_Yes, df_No = divide_dataframe_onAttr(df, attr_name,  attr_value)
    #print(len(df_Yes), len(df_No))
    
    No_train, No_test = train_test_split(df_No, test_size = (1-cutPerc), train_size = cutPerc)
    Yes_train, Yes_test = train_test_split(df_Yes, test_size = (1-cutPerc), train_size = cutPerc)
    
    trainSet = pd.concat([Yes_train, No_train])
    testSet=pd.concat([Yes_test, No_test])
    
    return trainSet, testSet


def applying_knnClassifier(k, aMetric, x_trainSet, y_trainSet, x_testSet):
    # Create k-nearest neighbors classifer object
    model = KNeighborsClassifier(n_neighbors=k, metric=aMetric)
     
    # Train the model using the training sets
    '''
    y_trainSet = y_trainSet.astype('int')
    '''
    model.fit(x_trainSet, y_trainSet)

    #Predict the response for test dataset
    y_pred = model.predict(x_testSet)
    
    return y_pred

    
def evaluatingClassifier_model(y_pred, y_testSet, accuFlag, cmFlag): 
    '''
    y_testSet = y_testSet.astype('int')
    '''
    # Accuracy
    score = metrics.accuracy_score(y_testSet, y_pred)
    s="Accuracy: "+ str(score)+"\r"; print(s)
    # Confusion Matrix (Digits Dataset)
    cm = metrics.confusion_matrix(y_testSet, y_pred) #; print(cm[0][1],cm[1][0])
    s="Confusion matrix: \r\n"+str(cm)+"\r\n\n"; print(s);
    #print(type(cm))
    if accuFlag and cmFlag:
        return score, cm
    elif accuFlag:
        return score
    elif cmFlag:
        return cm
    

    
def comparingModels_single_testSet_categoricalOutput(cm1, cm2, lenTestSet, flag):
    E1 = (cm1[0][1] + cm1[1][0])/lenTestSet
    E2 = (cm2[0][1] + cm2[1][0])/lenTestSet
    
    var1=E1*(1-E1); var2=E2*(1-E2)
    std1 = sqrt(var1/lenTestSet)
    std2 = sqrt(var2/lenTestSet)
    P=abs(E1-E2)/ sqrt( (std1+std2) / lenTestSet)
    s="E1: "+str(E1)+", E2: "+str(E2)+", P: "+str(P)#; print(s)
    if flag:
        if P>=2:
            return True, s
        return False, s
    else:
        if P>=2:
            return True
        return False


def main():
    
    k_list=[5,7]
    metrics_list=['euclidean', 'manhattan', 'chebyshev']
    
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    
    #read the data from .xlsx
    fileName = 'diabetes_data.csv'
    df = pd.read_csv(fileName) #; print(df); print(df.dtypes)
    
    
    yName = 'Outcome'
    df_trainSet, df_testSet = createAttrSample_train_test(df, yName,  attr_value = 0, cutPerc=0.8) #; print(df); print(trainSet)#; print(df_trainSet)
    x_trainSet, y_trainSet =separate_X_y(df_trainSet, yName, attrDrop=[], flag_to_numpy=False)
    x_testSet, y_testSet =separate_X_y(df_testSet, yName, attrDrop=[], flag_to_numpy=False)
   
    len_yTest=len(y_testSet)
    
    dict_cmModels={}
    for k in k_list:
        for aMetric in metrics_list:
            y_pred = applying_knnClassifier(k, aMetric, x_trainSet, y_trainSet, x_testSet) #; print(y_pred)
            cm = evaluatingClassifier_model(y_pred, y_testSet, False, True)
            dict_cmModels.update({(k,aMetric):cm})
    
    txtName = fileName.strip('csv')+'txt'
    f = open(txtName, 'a')

    keysExamined=[]
    for primeKey in dict_cmModels.keys():
        for key in dict_cmModels.keys():
            if key not in keysExamined and primeKey!=key:
                #print(primeKey, key)
                result, s =comparingModels_single_testSet_categoricalOutput(dict_cmModels[primeKey], dict_cmModels[key], len_yTest, True)
                f.write(str(primeKey)+' '+str(key)+': '+s+' => '+str(result)+'\n')
        keysExamined.append(primeKey) #; print(keysExamined)
    f.close()


    
if __name__ == "__main__":     
    main()