from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

import torch
import torch.nn as nn
import torch.optim as optim

#from torchnlp.samplers import BucketBatchSampler
from torchnlp.datasets import snli_dataset
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.encoders import LabelEncoder
from torchnlp import word_to_vector

import itertools
from tqdm import tqdm

from layers import Classifier, Encoder

# load dataset
train, dev, test = snli_dataset(train= True, dev= True, test= True)

# preprocess
for row in itertools.chain(train, dev, test):
    row['premise'] = row['premise'].lower()
    row['hypothesis'] = row['hypothesis'].lower()

# make encoders
sentence_corpus = [row['premise'] for row in itertools.chain(train, dev, test)]
sentence_corpus += [row['hypothesis'] for row in itertools.chain(train, dev, test)]
sentence_encoder = WhitespaceEncoder(sentence_corpus)
label_corpus = [row['label'] for row in itertools.chain(train, dev, test)]
label_encoder = LabelEncoder(label_corpus)

# encode 
for row in itertools.chain(train, dev, test):
    row['premise'] = sentence_encoder.encode(row['premise'])
    row['hypothesis'] = sentence_encoder.encode(row['hypothesis'])
    row['label'] = label_encoder.encode(row['label'])

d_model = sentence_encoder.vocab_size
learning_rate = 0.01
print('model')
model = Encoder(d_model, 256, 0, 1)
print('model1')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

num_epochs = 1

for epoch in range(num_epochs):
    print('epoch 1')
    n_correct, n_total = 0, 0
    #train_sampler = SequentialSampler(train)
    train_iterator = DataLoader(train, batch_size = 32)

    for batch_idx, (premise_batch, hypothesis_batch, label_batch) in enumerate(tqdm(train_iterator)):
        print('in enumerate')
        answer = model(premise_batch, hypothesis_batch)

        loss = criterion(answer, label_batch)
        loss.backward()
        optimizer.step()

        print(f'loss: {loss}')





























