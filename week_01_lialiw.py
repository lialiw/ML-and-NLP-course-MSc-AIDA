# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:45:08 2021

@author: Fik
"""

'''
Sources:
    
    https://www.geeksforgeeks.org/add-column-names-to-dataframe-in-pandas/
    https://stackoverflow.com/questions/31037298/pandas-get-column-average-mean
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html
    https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown
    https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests
    https://pythontic.com/pandas/series-plotting/bar%20chart
'''

import pandas as pd
import statistics as st
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
    
    
def change_binary_categorical_attr_values(df, attr, value_to_0, value_to_1): # i
    df.loc[(df[attr] == value_to_0) , attr]= 0 
    df.loc[(df[attr] == value_to_1) , attr]= 1     


def mean_std_median_lowAvg_highAvg_of_attr(df, attr_name): # ii
    #print(df[attr_name].describe())
    mean, var, std, median, median_lo, median_hi = df[attr_name].mean(), df[attr_name].var(), df[attr_name].std(), df[attr_name].median(), st.median_low(df[attr_name]), st.median_high(df[attr_name])
    print('Mean: ' + str(mean), 'Variance: '+ str(var), 'Std: '+ str(std), 'Median: '+ str(median), 'Median Low: '+ str(median_lo), 'Median High: '+ str(median_hi) )
    
    return mean, var, std, median, median_lo, median_hi


def plot_attr_meanVar(df_attr, mean, var): # iii
    
    #print(type(df_attr.value_counts()), df_attr.value_counts())
    df_attr.value_counts().plot.bar(rot=0, title="Diagnosis: Benign/0, Malignant/1")
    # https://pythontic.com/pandas/series-plotting/bar%20chart
    df_mean_var = pd.DataFrame( {'Diagnosis: Benign/0, Malignant/1':['Mean', 'Variance'], 'val':[mean, var]} )
    ax = df_mean_var.plot.bar(x='Diagnosis: Benign/0, Malignant/1', y='val', rot=0)
    
    
def separate_X_y(df_xValues, y_name, attrDrop, flag_to_numpy):
   
    df_xValues.drop(columns=attrDrop, inplace=True)
    df_yValue = df_xValues[y_name]
    df_xValues.drop(labels=y_name, axis=1, inplace=True)
    #print(df_yValue.name)
    #print(df_xValues.columns)
    if flag_to_numpy==1:
        return df_xValues.to_numpy(), df_yValue.to_numpy()
    else:
        #print(flag_to_numpy)
        return df_xValues, df_yValue
  
    
def logReg(solv, x_train, y_train, dirty):

    # Make an instance of the Model
    logisticRegr = LogisticRegression(solver=solv) #; print(x_train); print(y_train)

    if dirty:
        x_train = x_train + 0.01*np.random.rand(569, 30) #DataFrame, shape must be (568, 30)
        # https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests
        
    y_train=y_train.astype('int')
    # https://stackoverflow.com/questions/45346550/valueerror-unknown-label-type-unknown

    # Training the model on the data, storing the information learned from the data
    result = logisticRegr.fit(x_train, y_train)
    print(result.score(x_train, y_train))

    '''
    logit_model=sm.Logit(y_train,x_train)
    result=logit_model.fit() #LinAlgError: Singular matrix
    s = result.summary()
    print(s);
    #'''

    
def main():

    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)

    dataset='Breast_Cancer_Wisconsin_Diagnostic_Data_Set'
    #if dataset == 'Breast_Cancer_Wisconsin_Diagnostic_Data_Set':
    
    cols_names=["id","diag"]
    elements = ['a','b','c','d','e','f','g','h','i','j']
    for i in range(3):
        for j in range(len(elements)):
            cols_names.append(str(j + 2 + 10*i)+elements[j])
        
    df = pd.read_csv('wdbc.csv', delimiter = ',', header=None,  names=cols_names) #; print(df) ; print(len(df.index))
    #'''
    change_binary_categorical_attr_values(df, 'diag', 'B', 'M') #; print(df)
    
    mean, var, std, median, median_lo, median_hi = mean_std_median_lowAvg_highAvg_of_attr(df, 'diag')
    
    plot_attr_meanVar(df['diag'], mean, var)
    
    attrDrop = ['id']
    X,y = separate_X_y( df, 'diag', attrDrop, 0) #; print(X,y)
    solv='liblinear'
    logReg(solv, X, y, 1)
    #'''   
    

if __name__ == "__main__":     
    main()
