import torch
import torch.nn as nn
import sys
import os

class Latent_decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 10):
        super(Latent_decoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class Latent_decoder_linear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 10):
        super(Latent_decoder_linear, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(AdaptiveMLP, self).__init__()
        self.layers = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x












