'''
Created on Mar 12, 2019

@author: lhadhazy
'''
from sklearn.base import TransformerMixin

class ColumnCleanse(TransformerMixin):  

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna(axis='columns') 