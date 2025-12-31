from sklearn.mixture import GaussianMixture
from .base import BaseTrainer

class GMMTrainer(BaseTrainer):
    def __init__(self, n_clusters=2, random_state=0):
        self.params = dict(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto"
        )
        self.model = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def build(self):
        self.model = GaussianMixture(**self.params)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)
