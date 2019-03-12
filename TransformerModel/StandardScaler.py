'''
Created on Mar 12, 2019

@author: lhadhazy
'''
from sklearn.base import TransformerMixin
from sklearn import preprocessing

class StandardScaler(TransformerMixin):  # 5

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        scaler = preprocessing.StandardScaler()
        return scaler.fit_transform(X)
    