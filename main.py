import argparse
import logging
from src.pipeline import Pipeline
from src.utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Fusion Transformer')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    args = parser.parse_args()

    setup_logger()
    pipeline = Pipeline()

    if args.train:
        pipeline.train()
    elif args.evaluate:
        pipeline.evaluate()
    else:
        logging.error('Invalid arguments')

if __name__ == '__main__':
    main()