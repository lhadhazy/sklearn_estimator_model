from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from TransformerModel.StandardScaler import StandardScaler
from TransformerModel.NullColumnCleanse import NullColumnCleanse
from TransformerModel.LDATransformer import LDA


# Random Forest classifier decorator class
class CompositeRFEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=100):
        self.decorated_estimator = RandomForestClassifier(
            n_estimators=n_estimators, random_state=0)
        self.trained_estimator_ = None
        self.transform_steps = [
            NullColumnCleanse(),
            StandardScaler(),
            LDA(3)
            ]

    def fit(self, X, y):
        # forging of the data pipeline steps inside the fit method
        self.trained_estimator_ = make_pipeline(
            # self.transform_steps,
            self.decorated_estimator
        ).fit(X, y)

        print(self.decorated_estimator)
        return self

    def score(self, X, y=None):
        return(sum(self.predict(X)))

    def predict(self, X):
        return self.trained_estimator_.predict(X)
