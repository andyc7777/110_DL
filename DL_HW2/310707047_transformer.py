# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 05:10:36 2022

@author: Andy
"""

from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import spacy
from spacy.lang.en import English
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.data import get_tokenizer


def get_dataloader():

    spacy_en = English() #  spacy_en.tokenizer
    tokenize = lambda x: x.split()

    def tokenizer(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    tokenizer_1 = get_tokenizer("basic_english")

    category = Field(sequential=False, use_vocab=False, is_target=True)
    title = Field(sequential=True, use_vocab=True, tokenize=tokenizer_1,
                  lower=True, init_token='<sos>',
                  eos_token='<eos>', is_target=False)
    description = Field(sequential=True, use_vocab=True, tokenize=tokenizer_1,
                        lower=True, init_token='<sos>',
                        eos_token='<eos>', is_target=False)

    train_fields = {'Category': ('c', category),
                    'Title': ('t', title), 'Description': ('d', description)}
    test_fields = {'Title': ('t', title), 'Description': ('d', description)}

    train_data = TabularDataset(
                        path='./news_data/train.csv',
                        format='csv',
                        fields=train_fields)

    valid_data = TabularDataset(
                        path='./news_data/valid.csv',
                        format='csv',
                        fields=train_fields)

    test_data = TabularDataset(
                        path='./news_data/test.csv',
                        format='csv',
                        fields=test_fields)

    title.build_vocab(train_data, max_size=10000, min_freq=1)
    description.build_vocab(train_data,
                            max_size=10000,
                            min_freq=1,
                            )

    ntokens = len(description.vocab)  # size of vocabulary

    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), sort=False, batch_size=64, device="cuda", shuffle=(True, True))
    test_iterator = BucketIterator(test_data, sort=False,
                                   batch_size=1, device="cuda", shuffle=False)

    return train_iterator, valid_iterator, test_iterator, ntokens


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):

        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, 4)

        # print(self.decoder)

        self.init_weights()

    def init_weights(self) -> None:

        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:

        src = self.encoder(src)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        output = self.decoder(output)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def draw_learning_curve(training_loss_record, val_loss_record):

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss_record, label="Train")
    plt.plot(val_loss_record, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, '\n')

    train_iterator, valid_iterator, test_iterator, n_token = get_dataloader()

    emsize = 3000  # embedding dimension
    d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability

    model = TransformerModel(n_token, emsize, nhead, d_hid, nlayers, dropout).to(device)

    # Load the model checkpoint if it exist
    model_PATH = './310707047_transformer.pth'

    if os.path.exists(model_PATH):
        print('Model checkpoint exists, jump to testing phase.')
        model = torch.load(model_PATH)

    else:
        print('Model checkpoint does not exists, jump to training phase.')
        criterion = nn.CrossEntropyLoss()
        lr = 0.005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

        epochs = 50
        training_loss_record = []
        val_loss_record = []
        best_valid_loss = 1e10

        print('Training Phase...')  # Training
        for epoch in range(1, epochs + 1):
            training_loss = 0.0
            model.train()
            for idx, batch in tqdm(enumerate(train_iterator)):

                optimizer.zero_grad()
                y = batch.c - 1

                preds = model(batch.d)

                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()

                training_loss += loss.item() * batch.d.size(0)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx, batch in tqdm(enumerate(valid_iterator)):
                    y = batch.c - 1

                    preds = model(batch.d)

                    loss = criterion(preds, y)
                    val_loss += loss.item() * batch.d.size(0)

            lr = scheduler.get_last_lr()[0]
            epoch_loss = training_loss / 1800
            val_loss = val_loss / 200

            training_loss_record.append(epoch_loss)
            val_loss_record.append(val_loss)
            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
            if val_loss <= best_valid_loss:
                print('Model checkpoint saved')
                torch.save(model, '310707047_transformer.pth')
                best_valid_loss = val_loss

        draw_learning_curve(training_loss_record, val_loss_record)

    print('Testing Phase...')  # Testing

    model = torch.load('310707047_transformer.pth')
    model.eval()
    result = []
    with torch.no_grad():
        for idx, batch in enumerate(test_iterator):

            output = model(batch.d)
            pred = output.argmax(dim=1, keepdim=True) + 1
            result.append(pred.item())

    # Output the submission
    df = pd.read_csv('./news_data/test.csv')
    df_submission = df.drop(columns=['Title', 'Description'])
    df_submission['Category'] = result
    df_submission.to_csv('310707047_submission.csv', index=False)
    print('310707047_submission.csv file saved')


if __name__ == '__main__':
    main()
