import math
import numpy as np

import torch_geometric
import torch_geometric.utils as utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GraphTransformer_LLPE(nn.Module):
    '''
    GT with Learnable Laplacian Position Encodings
    '''
    def __init__(self, in_dim, pos_dim=8, h_dim=32, out_dim=1, K=64, num_layers=2, num_heads=4, dropout=0, layer_norm_eps=0):
        super(GraphTransformer_LLPE, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers 
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.feat_enc = nn.Linear(in_dim, h_dim)

        # Chebyshev polynomial parameters
        self.K = K # order of Chebyshev polynomial
        self.alpha = torch.nn.parameter.Parameter(torch.empty((self.K, pos_dim), requires_grad=True), requires_grad=True) # coefficients of Chebyshev Polynomials
        nn.init.uniform_(self.alpha, -1/math.sqrt(self.K), 1/math.sqrt(self.K))

        self.hidden_encs = nn.ModuleList()
        for l in range(num_layers):
            self.hidden_encs.append(nn.TransformerEncoderLayer(d_model=pos_dim+h_dim, dim_feedforward=h_dim, nhead=num_heads, dropout=dropout, batch_first=True, layer_norm_eps=layer_norm_eps))
        self.readout_layer = nn.Linear(pos_dim+h_dim, out_dim)

    def forward(self, data, device):
        assert data.batch == None, 'H2GT does not support multiple graphs yet!'
        x = data.x
        
        # process eigenvectors: min-max normalization in range [-1, +1]
        eigenvectors = data.eigenvectors
        eig_min = torch.min(eigenvectors, dim=0, keepdim=True)[0]
        eig_max = torch.max(eigenvectors, dim=0, keepdim=True)[0]
        eigenvectors = (eigenvectors - eig_min)/(eig_max - eig_min) * (1 + 1) - 1 

        # process eigenvalues: scale eigenvalues of Laplacian from [0, 2] --> [-1, +1]
        eigenvalues = data.eigenvalues
        eig_max = torch.max(eigenvalues)
        eig_min = torch.min(eigenvalues)
        eigenvalues = (eigenvalues - eig_min)/(eig_max - eig_min) * (1 + 1) - 1

        # iterative form of Chebyshev polynomial
        # eig_max = torch.max(eigenvalues)
        # eigenvalues_poly = torch.clone(eigenvalues)
        # for k in range(self.K): 
        #     eigenvalues_poly += self.alpha[k] * torch.cos(k * torch.arccos((2 * eigenvalues/eig_max) - 1))

        # vectorized form of Chebyshev polynomial
        eigenvalues = torch.arccos(torch.broadcast_to(eigenvalues[:, None], (eigenvalues.shape[0], self.K)))
        eigenvalues = torch.mul(eigenvalues, torch.arange(self.K).to(device))
        eigenvalues = torch.cos(eigenvalues)
        eigenvalues = torch.matmul(eigenvalues, self.alpha)
        
        # pos encoding is the eigenvectors multiplied by chebyshev filters
        pos = torch.matmul(eigenvectors, eigenvalues)
        x = torch.cat([self.feat_enc(x), pos], dim=-1)

        for encoder in self.hidden_encs:
            x = encoder(x)

        x = self.readout_layer(x)
        x = F.log_softmax(x, dim=-1)
        
        return x, eigenvalues







