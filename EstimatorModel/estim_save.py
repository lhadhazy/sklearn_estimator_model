import numpy as np

from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing.tests.test_data import n_features
from TransformerModel.StandardScaler import StandardScaler


# Estimator decorator class
class CompositeDTEstimator(BaseEstimator, ClassifierMixin):  # 1

    def __init__(self, estimator=DecisionTreeClassifier(random_state=0)):  # 2
        self.decorated_estimator = estimator
        self.transform_steps = {  # 3
            'group_size': np.log2
        }

        self.transform_steps = [
            ColumnCleanse(),
            StandardScaler(),
            LDA(3),
            FunctionTransformer(self._handle_nulls, validate=False),
            FunctionTransformer(self._transform_columns, validate=False),
            ]
    
    def fit(self, X, y):
        # forging of the data pipeline steps under the fit method
        self.trained_estimator_ = make_pipeline(# 4
            ColumnSelector([
                'average_speed', 'distance', 'group_size'  # etc.
            ]),
            self.transform_steps,
            self.decorated_estimator
        ).fit(X, y)

        return self

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return(sum(self.predict(X))) 
    
    def predict(self, X):
        return self.trained_estimator_.predict(X)


# A stateful transformer 
class ColumnSelector(TransformerMixin):  # 5

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.columns]

    
class ColumnCleanse(TransformerMixin):  # 5

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.dropna(axis='columns') 

    
class LDA(TransformerMixin):  # 5

    def __init__(self, n_features):
        self.n_features = n_features
        self.lda = LinearDiscriminantAnalysis(n_components=n_features)
    
    def fit(self, X, y=None):
        return self.lda.fit(X, y)
    
    def transform(self, X):
        return X.transform(X)
    
