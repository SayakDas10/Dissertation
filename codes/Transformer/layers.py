import torch
import torch.nn as nn
from torchsummary import summary

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_k):
        super(SelfAttentionLayer, self).__init__()
        self.Q = nn.Linear(d_k, d_k)
        self.K = nn.Linear(d_k, d_k)
        self.V = nn.Linear(d_k, d_k)
        self.softmax = nn.Softmax(dim= -1)
        self.d_k = d_k

    def forward(self, x):
        attention = self.softmax(((self.Q(x) @ self.K(x).mT)/self.d_k**0.5) @ self.V(x))
        return attention

class EncoderBlock(nn.Module):
    def __init__(self, d_k, d_hidden, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attention_layer = SelfAttentionLayer(d_k)
        self.layer_norm_1 = nn.LayerNorm(d_k)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_k, d_hidden),
            nn.Linear(d_hidden, d_k)
        )
        self.layer_norm_2 = nn.LayerNorm(d_k)

    def forward(self, x):
        out = self.layer_norm_1(x + self.self_attention_layer(x))
        out = self.layer_norm_2(out + self.feed_forward(out))

        return out

class Encoder(nn.Module):
    def __init__(self, d_k, d_hidden, dropout, num_encoder_layers):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock(d_k, d_hidden, dropout) for _ in range(num_encoder_layers)]
        )

    def forward(self, x):

        return self.encoder_blocks(x)


class Classifier(nn.Module):
    def __init__(self, d_model, d_hidden, dropout, num_encoder_layers, num_classes):
        super().__init__()
        self.encoder = Encoder(d_model, d_hidden, dropout, num_encoder_layers)
        self.l1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(d_hidden, num_classes)

    def forward(self, premise, hypothesis):
        out = torch.cat([premise, hypothesis], 1)
        out = self.encoder(out)
        out = self.l1(out)
        out = self.relu(out)
        out = self.l2(out)

        return out



dims_embd = 10
num_data_points = 100
batch_size = 5
num_hidden_nodes_ffnn = 512
dropout_prob = 0.2
num_layers_encoder = 2

x = torch.rand(batch_size, num_data_points, dims_embd)
y = torch.rand(batch_size, num_data_points, dims_embd)

# Test Attention layer and its input output size  
print('='*70)
model_attention_layer = SelfAttentionLayer(dims_embd)
print('Attention layer models is: \n{}' .format(model_attention_layer))
print('-'*70)
y = model_attention_layer(x)
print('Attention layer input size: {}' .format(x.shape))
print('Attention layer output size: {}' .format(y.shape))
print('-'*70)
        
# Test Transformer block input output size 
print('='*70)
model_transformer_block = EncoderBlock(dims_embd, num_hidden_nodes_ffnn, dropout_prob)
print('Transformer block models is: \n{}' .format(model_transformer_block))
print('-'*70)
print('Transformer block models summary:')
print('-'*70)
summary(model_transformer_block, (num_data_points, dims_embd, ), device=str("cpu"))
print('-'*70)

y = model_transformer_block(x)
print('Transformer block input size: {}' .format(x.shape))
print('Transformer block output size: {}' .format(y.shape))  
print('-'*70)

# Test Transformer encoder input output size 
print('='*70)
model_transformer_encoder = Encoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob, num_layers_encoder)
print('Transformer encoder models is: \n{}' .format(model_transformer_encoder))
print('-'*70)
print('Transformer encoder models summary:')
print('-'*70)
summary(model_transformer_encoder, (num_data_points, dims_embd, ), device=str("cpu"))
print('-'*70)

y = model_transformer_encoder(x)
print('Transformer encoder input size: {}' .format(x.shape))
print('Transformer encoder output size: {}' .format(y.shape))  
print('-'*70)

# Test classifier input output size 
print('='*70)
model_classifier= Classifier(dims_embd, num_hidden_nodes_ffnn, dropout_prob, num_layers_encoder, 2)
print('Transformer encoder models is: \n{}' .format(model_classifier))
print('-'*70)
print('Transformer encoder models summary:')
print('-'*70)
#summary(model_classifier, ((num_data_points, num_data_points), dims_embd, ), device=str("cpu"))
print('-'*70)

y = model_classifier(x, y)
print('Transformer encoder input size: {}' .format(x.shape))
print('Transformer encoder output size: {}' .format(y.shape))  
print('-'*70)



