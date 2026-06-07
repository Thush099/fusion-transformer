import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class FusionTransformer(nn.Module):
    def __init__(self):
        super(FusionTransformer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tabular_encoder = nn.Linear(10, 128)
        self.text_encoder = nn.Linear(128, 128)
        self.cross_attention = nn.MultiHeadAttention(128, 8)
        self.output_layer = nn.Linear(128, 2)

    def train(self, X_tabular, X_text, y):
        logging.info('Training the model')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        X_tabular = torch.tensor(X_tabular).to(device)
        X_text = torch.tensor(X_text).to(device)
        y = torch.tensor(y).to(device)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(10):
            optimizer.zero_grad()
            tabular_embeddings = self.tabular_encoder(X_tabular)
            text_embeddings = self.text_encoder(X_text)
            cross_attention_embeddings = self.cross_attention(tabular_embeddings, text_embeddings)
            output = self.output_layer(cross_attention_embeddings)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def evaluate(self, X_tabular, X_text, y):
        logging.info('Evaluating the model')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        X_tabular = torch.tensor(X_tabular).to(device)
        X_text = torch.tensor(X_text).to(device)
        y = torch.tensor(y).to(device)
        tabular_embeddings = self.tabular_encoder(X_tabular)
        text_embeddings = self.text_encoder(X_text)
        cross_attention_embeddings = self.cross_attention(tabular_embeddings, text_embeddings)
        output = self.output_layer(cross_attention_embeddings)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        f1_score = 2 * (accuracy * (predicted == y).sum().item() / len(y)) / (accuracy + (predicted == y).sum().item() / len(y))
        return accuracy, f1_score

    def save(self):
        logging.info('Saving the trained model')
        torch.save(self.state_dict(), 'model.pth')

    def load(self):
        logging.info('Loading the trained model')
        self.load_state_dict(torch.load('model.pth'))