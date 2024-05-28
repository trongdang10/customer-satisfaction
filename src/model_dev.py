import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        args:
            X_train:training data
            y_traing training labels
        
        returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    linear regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        trains the model
        args:
            X_train: training data
            y_train: training labels
        Returns:
            none
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model: {}", format(e))
            raise e
        
