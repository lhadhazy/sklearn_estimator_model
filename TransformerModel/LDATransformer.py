'''
Linear Discriminant Analysis Transformer
Created on Mar 12, 2019

@author: lhadhazy
'''
from sklearn.base import TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDA(TransformerMixin):

    def __init__(self, n_features):
        self.n_features = n_features
        self.lda = LinearDiscriminantAnalysis(n_components=n_features)

    def fit(self, X, y=None):
        return self.lda.fit(X, y)

    def transform(self, X):
        return X.transform(X)
