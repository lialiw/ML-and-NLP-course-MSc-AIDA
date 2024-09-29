# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 03:43:01 2021

@author: Fik
"""
"""
Sources:
    https://www.geeksforgeeks.org/convert-excel-to-csv-in-python/
    https://www.datacamp.com/community/tutorials/decision-tree-classification-python
    https://www.section.io/engineering-education/entropy-information-gain-machine-learning/
    https://blog.clairvoyantsoft.com/entropy-information-gain-and-gini-index-the-crux-of-a-decision-tree-99d0cdc699f4
    
    https://www.statology.org/pandas-unique-values-in-column/
    https://www.geeksforgeeks.org/how-to-drop-rows-in-pandas-dataframe-by-index-labels/
    https://www.kite.com/python/answers/how-to-replace-column-values-in-a-pandas-dataframe-in-python
    https://stackoverflow.com/questions/19900202/how-to-determine-whether-a-column-variable-is-numeric-or-not-in-pandas-numpy
    
    Attention to:
        https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
    
    Code used from:
        https://www.datacamp.com/community/tutorials/decision-tree-classification-python
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



def preProcessing(df, indexRowsToRemove_list, attrDrop_list, attrForBinaryReplace, attrForBinaryReplace_reversed, attrCategorical_dictOfDict):  
    
    df.drop(indexRowsToRemove_list, inplace=True)
    
    for attr in attrForBinaryReplace:
        #print(df[attr].unique()) 
        '''CHECK print above!!!'''
        uniqueVals_list = list(df[attr].unique())
        change_binary_categorical_attr_values(df, attr, uniqueVals_list[1], uniqueVals_list[0])
    
    for attr in attrForBinaryReplace_reversed:
        #print(df[attr].unique()) 
        '''CHECK print above!!!'''
        uniqueVals_list = list(df[attr].unique())
        change_binary_categorical_attr_values(df, attr, uniqueVals_list[0], uniqueVals_list[1])
    
    for attr in attrCategorical_dictOfDict.keys():
        df[attr].replace(attrCategorical_dictOfDict[attr], inplace=True)
        
        
def change_binary_categorical_attr_values(df, attr, value_to_0, value_to_1):
    df.loc[(df[attr] == value_to_0) , attr]= 0 
    df.loc[(df[attr] == value_to_1) , attr]= 1
    
    
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


def applying_DTClassifier(x_trainSet, y_trainSet, x_testSet):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    
    # Train Decision Tree Classifer
    y_trainSet = y_trainSet.astype('int')
    clf = clf.fit(x_trainSet, y_trainSet)

    #Predict the response for test dataset
    y_pred = clf.predict(x_testSet)
    
    return y_pred

    
def evaluatingClassifier_model(y_pred, y_testSet, accuFlag, cmFlag): 
    y_testSet = y_testSet.astype('int')
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
    

def main():
    
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    
    #read the data from .xlsx
    fileName = 'Cardiology.xlsx'
    df = pd.DataFrame(pd.read_excel(fileName))#; print(df)
    
    attrCategorical_dictOfDict = { 'chest pain type' : {' Asymptomatic':1, 'NoTang':0, 'Abnormal Angina':3, 'Angina':2}, 'resting ecg': {'Normal':0, 'Hyp':1, 'Abnormal':2}, 
                                  'slope':{'Down':0, 'Flat':1, 'Up':2}, 'thal' : {'Fix':1, 'Rev':2, 'Normal':0}}
    attrForBinaryReplace = ['sex','angina', 'class']
    attrForBinaryReplace_reversed = ['Fasting blood sugar <120']
    

    preProcessing(df, [], [], attrForBinaryReplace, attrForBinaryReplace_reversed, attrCategorical_dictOfDict)#; print(df)

    yName = 'class'
    df_trainSet, df_testSet = createAttrSample_train_test(df, yName,  attr_value = 0, cutPerc=0.75) #; print(df); print(trainSet)#; print(df_trainSet)
    x_trainSet, y_trainSet =separate_X_y(df_trainSet, yName, attrDrop=[], flag_to_numpy=False)
    x_testSet, y_testSet =separate_X_y(df_testSet, yName, attrDrop=[], flag_to_numpy=False)
    
    y_pred = applying_DTClassifier(x_trainSet, y_trainSet, x_testSet)  
    
    evaluatingClassifier_model(y_pred, y_testSet, False, False)
    
    
    
if __name__ == "__main__":     
    main()