import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from TransformerModel.StandardScaler import StandardScaler
from TransformerModel.ColumnSelector import ColumnSelector
from TransformerModel.ColumnCleanse import ColumnCleanse


# Estimator decorator class
class CompositeDTEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator=DecisionTreeClassifier(random_state=0)):
        self.decorated_estimator = estimator
        self.transform_steps = {}

        self.transform_steps = [
            ColumnCleanse(),
            StandardScaler(),
            LDA(3)
            ]

    def fit(self, X, y):
        # forging of the data pipeline steps under the fit method
        self.trained_estimator_ = make_pipeline(
            ColumnSelector([
                'average_speed', 'distance', 'group_size'
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


class LDA(TransformerMixin):  # 5

    def __init__(self, n_features):
        self.n_features = n_features
        self.lda = LinearDiscriminantAnalysis(n_components=n_features)

    def fit(self, X, y=None):
        return self.lda.fit(X, y)

    def transform(self, X):
        return X.transform(X)
