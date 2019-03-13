from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from TransformerModel.StandardScaler import StandardScaler
from TransformerModel.NullColumnCleanse import NullColumnCleanse
from TransformerModel.LDATransformer import LDA


# Random Forest classifier decorator class
class CompositeSVCEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1, kernel='rbf', gamma='auto',
                 class_weight='balanced', cache_size=800, probability=False):
        self.decorated_estimator = SVC(C=C, kernel=kernel, gamma=gamma,
                                       class_weight=class_weight,
                                       cache_size=cache_size, max_iter=1,
                                       probability=probability)
        self.C = C
        self.kernel = kernel
        self.class_weight = class_weight
        self.gamma = gamma
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
