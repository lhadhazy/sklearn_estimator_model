from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn import svm
from sklearn.pipeline import make_pipeline
from TransformerModel.StandardScaler import StandardScaler
from TransformerModel.NullColumnCleanse import NullColumnCleanse
from TransformerModel.LDATransformer import LDA


# Decision tree classifier decorator class
class CompositeOneClassSVMEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, nu=0.1, kernel="rbf", cache_size=800, gamma='auto'):
        self.decorated_estimator = svm.OneClassSVM(nu=nu, kernel=kernel,
                                                   gamma=gamma,
                                                   cache_size=cache_size)

        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
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
