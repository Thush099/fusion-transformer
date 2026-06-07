# Fusion Transformer

Combines tabular and text data using FT-Transformer and cross-attention.

## Problem Statement
In many real-world applications, we have both tabular and text data. However, most machine learning models are designed to handle only one type of data. This project aims to fill this gap by providing a multimodal fusion model that can handle both tabular and text data.

## Architecture
```
+---------------+
|  Tabular Data  |
+---------------+
         |
         |
         v
+---------------+
|  FT-Transformer  |
+---------------+
         |
         |
         v
+---------------+
|  Cross-Attention  |
+---------------+
         |
         |
         v
+---------------+
|  Output Layer    |
+---------------+
```

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To train the model, run the following command:
```bash
python main.py --train
```
This will generate synthetic data, train the model, and save the trained model to a file.

To evaluate the model, run the following command:
```bash
python main.py --evaluate
```
This will load the trained model, generate synthetic data, and evaluate the model on the generated data.

## Sample Output
The model will output the predicted probabilities for each class.

## Design Decisions
The model uses a FT-Transformer to combine the tabular and text data. The FT-Transformer is a type of transformer that is designed to handle multiple input types. The cross-attention mechanism is used to attend to the relevant parts of the input data. The output layer is a simple linear layer with a softmax activation function.

The model is trained using a cross-entropy loss function and the Adam optimizer. The model is evaluated using accuracy and F1 score.