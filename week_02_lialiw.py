# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 17:37:53 2021

@author: Fik
"""

'''
Sources:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    https://datacarpentry.org/python-ecology-lesson/03-index-slice-subset/
    https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html
    https://www.marsja.se/pandas-count-occurrences-in-column-unique-values/
    https://www.geeksforgeeks.org/how-to-concatenate-two-or-more-pandas-dataframes/
    https://stackoverflow.com/questions/20297332/how-do-i-retrieve-the-number-of-columns-in-a-pandas-data-frame?rq=1
    https://stackoverflow.com/questions/11346283/renaming-column-names-in-pandas
    https://moonbooks.org/Articles/How-to-copy-a-dataframe-with-pandas-in-python-/
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    https://medium.com/analytics-vidhya/calculation-of-bias-variance-in-python-8f96463c8942
'''


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
'''
from sklearn.metrics import mean_absolute_error
from mlxtend.evaluate import bias_variance_decomp
'''
from math import sqrt

def change_binary_categorical_attr_values(df, attr, value_to_0, value_to_1): # i
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
 

def standarizeData(x):
    '''
    Takes as input a df -x- and scales its features (columns)
    onto unit scale (mean = 0 and variance = 1)
    '''
    return StandardScaler().fit_transform(x)    

   
def divide_dataframe_onAttr(df, attr, value):
    df_valueOff = df.drop(df[ (df[attr]==value) ].index)
    df_valueOn = df.drop(df_valueOff.index)
    return df_valueOn, df_valueOff


def sampleSize(initialSize, percentCut, percentDistr):
    sizeYes = round(initialSize*percentCut*percentDistr)
    return sizeYes, round(initialSize*percentCut)-sizeYes


def createAttrSample_train_test(df, attr_name, attr_value, cutPerc, attrPerc, replace):
    
    df_Yes, df_No = divide_dataframe_onAttr(df, attr_name,  attr_value)
    #print(len(df_Yes), len(df_No))
    sampleSize_Yes, sampleSize_No = sampleSize(len(df), cutPerc, attrPerc)
    
    if replace:
        sample_Yes = df_Yes.sample(n=sampleSize_Yes, replace=True)
        sample_No = df_No.sample(n=sampleSize_No,replace=True)
    else:
        sample_Yes = df_Yes.sample(n=sampleSize_Yes)
        sample_No = df_No.sample(n=sampleSize_No)



    sample_attr_train = pd.concat([sample_Yes, sample_No],sort=False).sort_index()
    '''
    #Equivalently:
    sample_attr_train = sample_No.append(sample_Yes, ignore_index=True)
    '''
    #print(type(sample_attr_train))
    #print(len(sample_attr_train), len(sample_Yes))  
    sample_attr_test = df.drop(sample_attr_train.index)

    return sample_attr_train, sample_attr_test

    
def pca(x, explVar, pca095):
    '''
    Takes as input: 
        a df -x-, standarizes it and applies PCA with parameter explVar
        a float, i.e. explVar, which expresses the minimum percentage of total variance wished to be explained
        a boolean, i.e. pca095
    Returns as output: 
        a df with principal components whose variance (cumulatively) explains at least the percentage given as input 
        a list with PCA column's names    
        if pca095==True, it also returns a df with 'all' its principal components. --> The 'default' option, so PCA 'chooses' 
        the minimum number of principal components such that 95% of the variance is retained.
    '''
    x = standarizeData(x)   #Scaling the data's features (- PCA is effected by scale)     
    principalComponents_explVar = PCA(explVar).fit_transform(x)
    principalDf_explVar = pd.DataFrame(data = principalComponents_explVar) #; print(principalDf_explVar)
    
    columns_names=["pc"+str(i) for i in range(principalDf_explVar.shape[1])] #number of columns of principalDf_explVar    
    #print(columns_names)
    principalDf_explVar.columns = columns_names
        
    if pca095:
        principalComponents = PCA().fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents) ; print(principalDf)
        columns_names=["pc"+str(i) for i in range(principalDf.shape[1])] #number of columns of principalDf
        return principalDf, principalDf_explVar
    
    #print(principalDf_explVar)
    return principalDf_explVar, columns_names
 

def logReg(solv, x_train, y_train, x_test, y_test):

    # Make an instance of the Model
    logisticRegr = LogisticRegression(solver=solv)
    
    # Training the model on the data, storing the information learned from the data
    logisticRegr.fit(x_train, y_train)

    # Evaluation:
    '''
    Other ways of measuring model performance: precision, recall, F1 Score, ROC Curve, etc.
    accuracy = correct predictions / total number of data points
    ''' 
    # Accuracy
    score = logisticRegr.score(x_test, y_test)
    s="Accuracy: "+ str(score)+"\r"; print(s)
  
    #Make predictions on entire test data
    predictions = logisticRegr.predict(x_test)
    # Confusion Matrix (Digits Dataset)
    cm = metrics.confusion_matrix(y_test, predictions) #; print(cm[0][1],cm[1][0])
    #confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
    s="Confusion matrix: \r\n"+str(cm)+"\r\n\n"; print(s);
    
    #print(type(y_test), type(predictions))
    #print( pd.Series(predictions))
    return cm
       

def comparingModels_single_testSet_categoricalOutput(cm1, cm2, lenTestSet, flag):
    E1 = (cm1[0][1] + cm1[1][0])/lenTestSet
    E2 = (cm2[0][1] + cm2[1][0])/lenTestSet
    
    var1=E1*(1-E1); var2=E2*(1-E2)
    std1 = sqrt(var1/lenTestSet)
    std2 = sqrt(var2/lenTestSet)
    P=abs(E1-E2)/ sqrt( (std1+std2) / lenTestSet)
    s="E1: "+str(E1)+", E2: "+str(E2)+", P: "+str(P); print(s)
    if flag:
        if P>=2:
            return True, s
        return False, s
    else:
        if P>=2:
            return True
        return False
    

def main():
    '''
    Dataset:
        Breast_Cancer_Wisconsin_Diagnostic_Data_Set
    '''

    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)

    
    #Creating names for dataset's columns 
    cols_names=["id","diag"]
    elements = ['a','b','c','d','e','f','g','h','i','j']
    for i in range(3):
        for j in range(len(elements)):
            cols_names.append(str(j + 2 + 10*i)+elements[j])
   
    #Reading .csv, naming its columns, assigning it to a df    
    df = pd.read_csv('wdbc.csv', delimiter = ',', header=None,  names=cols_names) #; print(df) ; print(len(df.index))

    #Changing values of column from categorical to binary
    change_binary_categorical_attr_values(df, 'diag', 'B', 'M') #; print(df)
    
    
    attrDrop= ['id'] #Columns to be dropped, considered irrelevant to dataset's analysis
    target_var ='diag'
    
    #Applying PCA 0.75
    X,y = separate_X_y(df, target_var, attrDrop, flag_to_numpy=0)
    principalDf_explVar, pca_cols_names = pca(X, explVar=0.75, pca095=False)
    
    
    #Combining initial dataset with principal components produced
    df_y_prExpvar = pd.concat([y, X, principalDf_explVar],axis=1)#; print(df_y_prExpvar)
    

    #Creating training and testing sets

    y_percentages = df_y_prExpvar[target_var].value_counts(normalize=True) #target variable's "distribution"
    #print(y_percentages)

    attr_value = 0 #attr_value one of the values of target variable
    cutPerc = 0.85 #percentage of records of the initial dtaset to be in the training set
    
    trainSet, testSet = createAttrSample_train_test(df_y_prExpvar, target_var, attr_value , cutPerc, y_percentages[attr_value], replace=0) #;print(trainSet)

    x_trainSet, y_trainSet = separate_X_y(trainSet.copy(), target_var, attrDrop = pca_cols_names, flag_to_numpy=0)
    x_testSet, y_testSet = separate_X_y(testSet.copy(), target_var, attrDrop = pca_cols_names, flag_to_numpy=0)
    y_trainSet, y_testSet = y_trainSet.astype('int'), y_testSet.astype('int')
    # https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
    #print(x_trainSet, y_trainSet)#; print(trainSet)
    
    new_cols_names =list(set(cols_names)-set(['id','diag']))#; print(new_cols_names)
    x_pca_trainSet, y_pca_trainSet = separate_X_y(trainSet.copy(), target_var, attrDrop=new_cols_names , flag_to_numpy=0)
    x_pca_testSet, y_pca_testSet = separate_X_y(testSet.copy(), target_var, attrDrop=new_cols_names , flag_to_numpy=0)
    y_pca_trainSet, y_pca_testSet = y_pca_trainSet.astype('int'), y_pca_testSet.astype('int')
    #print(x_pca_trainSet, y_pca_trainSet)#; print(trainSet)    
    
    #Applying Logistic Regression
    solv='liblinear'
    
    cm = logReg(solv, x_trainSet, y_trainSet, x_testSet, y_testSet)#; print(predictions)
    cm_pca = logReg(solv, x_pca_trainSet, y_pca_trainSet, x_pca_testSet, y_pca_testSet)    
    
    #Comparing the two models
    signDifference = comparingModels_single_testSet_categoricalOutput(cm, cm_pca, len(y_pca_testSet),False)
    print(signDifference)
    
    
    
if __name__ == "__main__":     
    main()