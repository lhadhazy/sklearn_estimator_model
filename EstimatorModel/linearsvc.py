from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from TransformerModel.StandardScaler import StandardScaler
from TransformerModel.NullColumnCleanse import NullColumnCleanse
from TransformerModel.LDATransformer import LDA


# LinearSVC classifier decorator class
class CompositeSVCEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, class_weight='balanced', max_iter=10000):
        self.decorated_estimator = LinearSVC(class_weight=class_weight,
                                             max_iter=1)

        self.class_weight = class_weight
        self.max_iter = max_iter
        self.trained_estimator_ = None
        self.transform_steps = [
            NullColumnCleanse(),
            StandardScaler(),
            LDA(1)
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
