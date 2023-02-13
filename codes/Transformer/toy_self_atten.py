import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.Q = nn.Linear(d_k, d_model)
        self.K = nn.Linear(d_k, d_model)
        self.V = nn.Linear(d_v, d_model)
        self.d_k = d_k
        self.softmax = nn.Softmax(dim= -1)

    def forward(self, x):
        out = self.Q(x) @ self.K(x)
        out = self.softmax(out / (self.d_k ** 0.5))
        out = out @ self.V(x)
        return out

class LayerNormalization(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm((d_model, d_model))

    def forward(self, x):
        return self.ln(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_hidden):
        super().__init__()
        self.self_attention = SelfAttention(d_model, d_k, d_v)
        self.layer_norm = LayerNormalization(d_model)
        self.ffnn = FeedForward(d_model, d_hidden)

    def forward(self, x):
        out = self.self_attention(x)
        res = out
        out = self.layer_norm(out+res)
        res = out
        out = self.ffnn(out)
        out = self.layer_norm(out+res)
        return out

class Classifier(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_hidden, num_classes):
        super().__init__()
        self.encoder = EncoderBlock(d_model, d_k, d_v, d_hidden)
        self.neural_net = nn.Linear(d_model, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.encoder(x)
        out = self.neural_net(out)
        out = self.sigmoid(out)
        return out


d_model = 8
h = 2
d_k = d_v = int(d_model/h)
d_hidden = 16
num_classes = 1
learning_rate = 0.01

X = torch.randn(d_model, d_k)
Y = torch.tensor([1, 0, 0, 1, 0, 1, 0, 1])
print(X)
model = EncoderBlock(d_model, d_k, d_v, d_hidden)
model = Classifier(d_model, d_k, d_v, d_hidden, num_classes)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

num_epochs = 2

for epoch in range(num_epochs):
    outputs = model(X)
    print(outputs.shape)
    loss = criterion(outputs.reshape(-1),Y.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'epoch {epoch+1}/{num_epochs}, loss= {loss.item():.4f}')







