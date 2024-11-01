import numpy as np
import torch
import torch.nn as nn

class neural_network(nn.Module):
    def __init__(self, feature_lists, n_out, activation, device):
        super(neural_network, self).__init__()
        self.device = device
        layers = []

        assert activation in ['relu', 'sigmoid', 'tanh', 'gelu', 'selu']
        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'gelu':
            activation = nn.GELU()
        elif activation == 'selu':
            activation = nn.SELU()

        for in_, out_ in zip(feature_lists[:-1], feature_lists[1:]):
            layers.append(nn.Linear(in_, out_))
            layers.append(activation)      
        layers.append(nn.Linear(feature_lists[-1], n_out))

        if n_out == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        
        self.layers = nn.Sequential(*layers)
        # self.layers.apply(self.init)

    def init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.layers(x)
    
    def infer(self, x):
        self.eval()
        with torch.no_grad():
            if type(x) is np.ndarray: # only pass np.ndarray if model is on cpu
                x = torch.from_numpy(x.astype(np.float32)).to(self.device)
            x = self.out(self.forward(x)).cpu().detach()
            return x