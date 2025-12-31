from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def build(self):
        ...

    @abstractmethod
    def fit(self, X):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    def train_and_predict(self, train, test, val):
        self.build()
        self.fit(train)
        return (
            self.predict(train),
            self.predict(test),
            self.predict(val),
        )