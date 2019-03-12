'''
Created on Mar 12, 2019

@author: lhadhazy
'''
from sklearn.base import TransformerMixin

class ColumnSelector(TransformerMixin): 

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.columns]