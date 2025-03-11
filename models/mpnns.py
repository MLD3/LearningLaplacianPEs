import math
import numpy as np

import torch_geometric
import torch_geometric.utils as utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SAGE_LLPE(nn.Module):
    '''
    SAGE with Learnable Laplacian Position Encodings
    '''
    def __init__(self, in_dim, pos_dim=8, h_dim=32, out_dim=1, K=64, num_layers=2, dropout=0):
        super(SAGE_LLPE, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feat_enc = nn.Linear(in_dim, h_dim)

        # Chebyshev polynomial parameters
        self.K = K # order of Chebyshev polynomial
        self.alpha = torch.nn.parameter.Parameter(torch.empty((self.K, pos_dim), requires_grad=True), requires_grad=True) # coefficients of Chebyshev Polynomials
        nn.init.uniform_(self.alpha, -1/math.sqrt(self.K), 1/math.sqrt(self.K))

        self.hidden_encs = nn.ModuleList()
        for l in range(num_layers):
            self.hidden_encs.append(SAGEConv(pos_dim+h_dim, pos_dim+h_dim))
            self.hidden_encs.append(nn.ReLU())
            self.hidden_encs.append(nn.Dropout(p=dropout))
            
        self.readout_layer = nn.Linear(pos_dim+h_dim, out_dim)
        # self.readout_layer = nn.Linear(h_dim, out_dim)

    def forward(self, data, device):
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

        # vectorized form of Chebyshev polynomial
        eigenvalues = torch.arccos(torch.broadcast_to(eigenvalues[:, None], (eigenvalues.shape[0], self.K)))
        eigenvalues = torch.mul(eigenvalues, torch.arange(self.K).to(device))
        eigenvalues = torch.cos(eigenvalues)
        eigenvalues = torch.matmul(eigenvalues, self.alpha)
        
        # pos encoding is the eigenvectors multiplied by chebyshev filters
        pos = torch.matmul(eigenvectors, eigenvalues)
        x = torch.cat([self.feat_enc(x), pos], dim=-1)
        # x = torch.cat([x, pos], dim=-1)

        edge_index = data.edge_index
        for l, layer in enumerate(self.hidden_encs[:-1]): 
            if np.mod(l, 3) == 0: 
                x = layer(x, edge_index) # GCN layer
            else: 
                x = layer(x) # dropout and relu layers

        x = self.readout_layer(x)
        x = F.log_softmax(x, dim=-1)
        
        return x, eigenvalues

class GCN_LLPE(nn.Module):
    '''
    GCN with Learnable Laplacian Position Encodings
    '''
    def __init__(self, in_dim, pos_dim=8, h_dim=32, out_dim=1, K=64, num_layers=2, dropout=0):
        super(GCN_LLPE, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feat_enc = nn.Linear(in_dim, h_dim)

        # Chebyshev polynomial parameters
        self.K = K # order of Chebyshev polynomial
        self.alpha = torch.nn.parameter.Parameter(torch.empty((self.K, pos_dim), requires_grad=True), requires_grad=True) # coefficients of Chebyshev Polynomials
        nn.init.uniform_(self.alpha, -1/math.sqrt(self.K), 1/math.sqrt(self.K))

        self.hidden_encs = nn.ModuleList()
        for l in range(num_layers):
            self.hidden_encs.append(GCNConv(pos_dim+h_dim, pos_dim+h_dim))
            self.hidden_encs.append(nn.ReLU())
            self.hidden_encs.append(nn.Dropout(p=dropout))
            
        self.readout_layer = nn.Linear(pos_dim+h_dim, out_dim)
        # self.readout_layer = nn.Linear(h_dim, out_dim)

    def forward(self, data, device):
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

        # vectorized form of Chebyshev polynomial
        eigenvalues = torch.arccos(torch.broadcast_to(eigenvalues[:, None], (eigenvalues.shape[0], self.K)))
        eigenvalues = torch.mul(eigenvalues, torch.arange(self.K).to(device))
        eigenvalues = torch.cos(eigenvalues)
        eigenvalues = torch.matmul(eigenvalues, self.alpha)
        
        # pos encoding is the eigenvectors multiplied by chebyshev filters
        pos = torch.matmul(eigenvectors, eigenvalues)
        x = torch.cat([self.feat_enc(x), pos], dim=-1)
        # x = torch.cat([x, pos], dim=-1)

        edge_index = data.edge_index
        for l, layer in enumerate(self.hidden_encs[:-1]): 
            if np.mod(l, 3) == 0: 
                x = layer(x, edge_index) # GCN layer
            else: 
                x = layer(x) # dropout and relu layers

        x = self.readout_layer(x)
        x = F.log_softmax(x, dim=-1)
        
        return x, eigenvalues

class MLP_LLPE(nn.Module):
    '''
    MLP with Learnable Laplacian Position Encodings
    '''
    def __init__(self, in_dim, pos_dim=8, h_dim=32, out_dim=1, K=64, num_layers=2, dropout=0):
        super(MLP_LLPE, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feat_enc = nn.Linear(in_dim, h_dim)

        # Chebyshev polynomial parameters
        self.K = K # order of Chebyshev polynomial
        self.alpha = torch.nn.parameter.Parameter(torch.empty((self.K, pos_dim), requires_grad=True), requires_grad=True) # coefficients of Chebyshev Polynomials
        nn.init.uniform_(self.alpha, -1/math.sqrt(self.K), 1/math.sqrt(self.K))

        self.hidden_encs = nn.ModuleList()
        for l in range(num_layers):
            self.hidden_encs.append(nn.Linear(pos_dim+h_dim, pos_dim+h_dim))
            # self.hidden_encs.append(nn.Linear(h_dim, h_dim))
            self.hidden_encs.append(nn.ReLU())
            self.hidden_encs.append(nn.Dropout(p=dropout))
            
        self.readout_layer = nn.Linear(pos_dim+h_dim, out_dim)
        # self.readout_layer = nn.Linear(h_dim, out_dim)

    def forward(self, data, device):
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

        # vectorized form of Chebyshev polynomial
        eigenvalues = torch.arccos(torch.broadcast_to(eigenvalues[:, None], (eigenvalues.shape[0], self.K)))
        eigenvalues = torch.mul(eigenvalues, torch.arange(self.K).to(device))
        eigenvalues = torch.cos(eigenvalues)
        eigenvalues = torch.matmul(eigenvalues, self.alpha)
        
        # pos encoding is the eigenvectors multiplied by chebyshev filters
        pos = torch.matmul(eigenvectors, eigenvalues)
        x = torch.cat([self.feat_enc(x), pos], dim=-1)
        # x = torch.cat([x, pos], dim=-1)

        for encoder in self.hidden_encs:
            x = encoder(x)

        x = self.readout_layer(x)
        x = F.log_softmax(x, dim=-1)
        
        return x, eigenvalues









