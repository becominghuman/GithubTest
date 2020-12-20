#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 18:12:12 2020

@author: fadouabadr
"""
####### we have seen how to reduce dimensionality of our feature matrix by vreating new features with 
# ideally similar ability to train quality modles but with fewer dimension: this is called FEATURE EXTRACTION
# here, we'll cover an alternative approach by selecting high-quality, iformative features and dropping less useful features
# ===> FEATURE SELECTION

#3 types of feature selection models : 
    # - Filter : select the best features by examining their statistical properties
    # - Wrapper : use trial and error to find the subset of features that produce models with the highest quality prediction
    # - Embedded : select the best feature subset as part of an extension of a learning algo' training process
    
'''
# ======================================================================
# Thresolding Numerical Feature Variance #########
# Problem : you have a set of numerical features and want to remove those with low variance (=> the one containing little information)
# Solution : select a subset of features with variance above a given thresold
# ======================================================================`
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

iris = load_iris()

features = iris.data
target = iris.target

#create thresolder
thresolder = VarianceThreshold(threshold=0.5)

# create high variance feature matrix
features_high_variance = thresolder.fit_transform(features)

#view high vatiance feature matrix
print(features_high_variance[0:3])

#Variance Thresolding is motivated by the idea that features with low variances are likely less interesting than features with high variance
# VT calculates the variance of each feature, and frops all features whose variance does not meet that thresold
# Rq : the variance is not centered => VT will not work when features sets contain different units (ex, features in year and feature in price)
# Second, the variance thresold is selected manually so we have to use our own judgement for a good value to select 

#view variances
print(thresolder.fit(features).variances_)

#Finally, if the features have been standardized (mean 0 and var =1) then VT will not work
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

#calculate vairance of each feature
print("Variance matrix for original set of features:", features.var(axis=0))
selector = VarianceThreshold()
print("Variance matrix for standardized set of features:", selector.fit(features_std).variances_)
'''
'''
# ======================================================================
# Thresolding binary feature variance
# Problem : You have a set of binary categorical  features and want to remove those with low variance
# Solution : select a subset of features with a bernoulli random variable variance above a given thresold
# ======================================================================
from sklearn.feature_selection import VarianceThreshold

#Create feature matrix with 
# feature 0 : 80% class 0
# feature 1 : 80% class 1 
# feature 2  : 60% class 0 and 40% class 1
features = [[0,1,0],[0,1,1],[0,1,0],[0,1,1],[1,0,0]]

#Run thresold by variance
thresolder = VarianceThreshold(threshold=(0.75 * (1-0.75)))
print(thresolder.fit_transform(features))

#One strategy for selecting highly informative categorical features is to examine their variance. In binary features
# their variance is calculated as var(x) = p(1-p)
# p = proportion of observations of class 1
'''

# ======================================================================
# Handling highly correlated features
# Problem : You have a feature matrix and suspect some features are highly correlated
# Solution : use a correlation matrix to check fr highly correlated features. 
# ======================================================================
import pandas as pd
import numpy as np

features = np.array([[1,1,1],[2,2,0],[3,3,1],[4,4,0], [5,5,1],[6,6,0], [7,7,1], [8,7,0], [9,7,1]])

#convert array to dataframe
df = pd.DataFrame(features)

corr = df.corr()

#absolute 
corr_abs = corr.abs()

#select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

#find index of feature columns with correlation great than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



           











