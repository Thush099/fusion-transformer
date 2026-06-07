import logging
import numpy as np
import torch
from transformers import BertTokenizer

def setup_logger():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def generate_synthetic_data():
    np.random.seed(42)
    X_tabular = np.random.rand(100, 10)
    X_text = ['This is a sample text'] * 100
    y = np.random.randint(0, 2, 100)
    return X_tabular, X_text, y