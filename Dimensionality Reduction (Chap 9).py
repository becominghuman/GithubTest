#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:17:20 2020

@author: fadouabadr
"""
'''
### Reducing Features using PCA
# ------------------------------------------
#Problem: given a set of features you want to reduce the number of features while retaining the variance in the data
#Solution : PCA
# ------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
features = StandardScaler().fit_transform(digits.data)

#create a PCA that will retain 99% of variance
pca = PCA(n_components=0.99, whiten=True)
features_pca = pca.fit_transform(features)

print("Original number of features:", features.shape)
print("Reduced number of features:", features_pca.shape)

# PCA: linear dimensionality reduction technique. PCA
# projects observations onto the PC of the feature matrix that retain most variance
# pca unsupervised technique meaning that it does not use the ingo
# from the target vector and instead only considers the fgeature matrix
'''


'''
#### Reducing Features when data is linearly inseparable
# ------------------------------------------
#Problem: you suspect you have linearly inspearable data and want to reduce dimensions
#Solution : use an extension of PCA that uses Kernel to allow for non linear dimensionality reduction
# ------------------------------------------
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

features, y = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)

plt.scatter(features[:,0], features[:,1])

#apply kernel pca with radiuss basis function (rbf) kernel
kpca = KernelPCA(kernel='rbf', gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])

#PCA is able to reduce dimensinality of our feature matrix 
# standard pca uses linear projection to reduce the features. if the data is linearly sperable (can draw a straight line or hyperplane beween differnt classes)
# then pca works well.
# but when non linearily separable (can only separate classes using a curved decision boundary), the linear transformation
# will not work as well
# kernel allow us to project the linearity inseparable daya into higher domension where it is linearly separable (kernel trick)
# kernel = different ways of projecting data
'''

'''
#### Reducing features by maximising class separability
# ------------------------------------------
#Problem: you suspect you have linearly inspearable data and want to reduce dimensions
#Solution : Linear discriminant analysis (LDA) to project features onto components axes that maximize the sepatation of classes
# ------------------------------------------
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris.data
target = iris.target

#plt.scatter(features, y)

#create and run an LDA then use it to transform the features
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)

print("Original number of features:", features.shape[1])
print("Reduced number of features after LDA:", features_lda.shape[1])

#we can use explained_variance_ratio to view the amount of variance explained by each component. In our 
#example, the single component explain 99% of the total variance
print(lda.explained_variance_ratio_)

#LDA is a clasification that is used for dimensionality reduction. LDA works similarly to PCA in that it projects our feature space onto a lower dimensional spaec
#however, in PCA we were only interested in the component axes that max the variance in the data while LDA WE HAVE 
#the additional goal of maximizin g the diff btw 2 classes. 

#Specifically, we can run LinearDiscriminationAnalysis with n_components set to None to return the ratio 
#of variance explained by every component feature, then caluclate how many components are required to get above some thresold of variance
#explained (0.95 or 0.99)
lda = LinearDiscriminantAnalysis(n_components=None)
features_lda = lda.fit(features, target)

#create array of explained variance ratios
lda_variance_ratios = lda.explained_variance_ratio_

def select_n_components(var_ratio, goal_var:float ) -> int: #####
    total_variance = 0.0
    n_components = 0
    
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        
        #if we rech our goal level of explained variance
        if total_variance >= goal_var:
            #end the look
            break
        
    return n_components #the function calculates how many component are required to get above some thresold of variance explained
#run function
print(select_n_components(lda_variance_ratios, 0.95))
'''


#### Reducing features using Matrix Factorization
# ------------------------------------------
#Problem: you have a feature matrix of non negative values and want to reduce dimensionality
#Solution : Use non-negative matrix factorization (NMF) 
# ------------------------------------------
from sklearn.decomposition import NMF
from sklearn.datasets import load_digits

digits = load_digits()
features = digits.data

nmf = NMF(n_components=10, random_state=1, max_iter=1000)
features_nmf = nmf.fit_transform(features)

print("Oroginal number of features", format(features.shape[1]))
print("Reduced number of features with NMF", format(features_nmf.shape[1]))

#NMF: unsupervised technique for linear reduction that factorises (breaks up into multiple matrix whose product approx the original matrix)
#the feature matrix into matrices representing the latent relationship btw observation and their features. 























