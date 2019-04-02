from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from TransformerModel.StandardScaler import StandardScaler
from TransformerModel.NullColumnCleanse import NullColumnCleanse
from TransformerModel.LDATransformer import LDA


# MLP classifier decorator class
class CompositeMLPEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, max_iter=200, solver='adam', activation='relu',
                 alpha=0.0001, learning_rate_init=0.001, batch_size='auto'):
        self.decorated_estimator = MLPClassifier(max_iter=max_iter,
                                                 activation=activation,
                                                 solver=solver,
                                                 alpha=alpha,
                                                 learning_rate_init=learning_rate_init,
                                                 batch_size=batch_size)

        self.max_iter = max_iter
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init,
        self.batch_size = batch_size,
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
