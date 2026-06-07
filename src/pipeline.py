import logging
from src.models import FusionTransformer
from src.utils import generate_synthetic_data

class Pipeline:
    def __init__(self):
        self.model = FusionTransformer()

    def train(self):
        logging.info('Generating synthetic data')
        X_tabular, X_text, y = generate_synthetic_data()
        logging.info('Training the model')
        self.model.train(X_tabular, X_text, y)
        logging.info('Saving the trained model')
        self.model.save()

    def evaluate(self):
        logging.info('Loading the trained model')
        self.model.load()
        logging.info('Generating synthetic data')
        X_tabular, X_text, y = generate_synthetic_data()
        logging.info('Evaluating the model')
        metrics = self.model.evaluate(X_tabular, X_text, y)
        logging.info(f'Accuracy: {metrics[0]}')
        logging.info(f'F1 Score: {metrics[1]}')