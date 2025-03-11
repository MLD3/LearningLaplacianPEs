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

class GraphTransformer(nn.Module):
    '''
    Graph Transformer from Dwivedi et al. 2020
        - adds positional encodings instead of concatenate as in original AAAI implementation
    '''
    def __init__(self, in_dim, pos_dim, h_dim=32, out_dim=1, num_layers=2, num_heads=4, dropout=0, layer_norm_eps=1e-5, pos_method='lap-fk'):
        super(GraphTransformer, self).__init__()
        self.in_dim = in_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_layers = num_layers 
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.pos_method = pos_method

        if pos_method == 'spec-attn':
            self.pos_encs = nn.ModuleList()
            self.pos_encs.append(nn.Linear(2, h_dim)) # input is N x N x 2 (since we use full spectrum) --> output is N x N x k
            self.pos_encs.append(nn.TransformerEncoderLayer(d_model=h_dim, dim_feedforward=h_dim, nhead=num_heads, dropout=dropout, norm_first=True, batch_first=True, layer_norm_eps=layer_norm_eps))
        elif pos_method == 'signnet':
            self.pos_encs_phi = nn.ModuleList()
            self.pos_encs_phi.append(nn.Linear(2, pos_dim)) # input is N x k x 2
            self.pos_encs_phi.append(nn.Linear(pos_dim, pos_dim)) # output is N x k x k

            self.pos_encs_rho = nn.ModuleList()
            self.pos_encs_rho.append(nn.Linear(pos_dim * pos_dim, h_dim)) # input is N x k * k
            self.pos_encs_rho.append(nn.Linear(h_dim, h_dim)) # output is N x k
        elif pos_method == 'lap-flk': 
            self.pos_enc = nn.Linear(2*pos_dim, h_dim)
        else: 
            self.pos_enc = nn.Linear(pos_dim, h_dim)
            
        self.feat_enc = nn.Linear(in_dim, h_dim)
        
        self.hidden_encs = nn.ModuleList()
        if pos_method != 'none': 
            h_dim = 2 * h_dim
        
        for _ in range(num_layers):
            self.hidden_encs.append(nn.TransformerEncoderLayer(d_model=h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout, norm_first=True, batch_first=True, layer_norm_eps=layer_norm_eps))
            
        self.readout_layer = nn.Linear(h_dim, out_dim)

    def forward(self, data):
        x = data.x
        if self.pos_method == 'none': 
            x = self.feat_enc(x)
        elif self.pos_method == 'lap-fk':
            pos = data.eigenvectors[:, 1:self.pos_dim+1]
            x = torch.cat([self.feat_enc(x), self.pos_enc(pos)], dim=-1)
        elif self.pos_method == 'lap-fkf':
            pos = data.eigenvectors[:, 1:self.pos_dim+1]
            rand_sign = torch.broadcast_to(torch.pow(-1, torch.randint(1, 3, (pos.shape[1],))), pos.shape) # randomly flip signs of eigenvectors
            pos = pos * rand_sign.to(torch.device(f'cuda:{pos.get_device()}'))
            x = torch.cat([self.feat_enc(x), self.pos_enc(pos)], dim=-1)
        elif self.pos_method == 'lap-flk':
            pos = torch.cat([data.eigenvectors[:, 1:(1+self.pos_dim)], data.eigenvectors[:, -self.pos_dim:]], dim=-1)
            x = torch.cat([self.feat_enc(x), self.pos_enc(pos)], dim=-1)
        elif self.pos_method == 'lap-full':
            pos = data.eigenvectors
            x = torch.cat([self.feat_enc(x), self.pos_enc(pos)], dim=-1)
        elif self.pos_method == 'elastic':
            pos = data.elastic_pe
            pos_min = torch.min(pos, dim=0, keepdim=True)[0]
            pos_max = torch.max(pos, dim=0, keepdim=True)[0]
            pos = (pos - pos_min)/(pos_max - pos_min) * (1 + 1) - 1 
            x = torch.cat([self.feat_enc(x), self.pos_enc(pos)], dim=-1)
        elif self.pos_method == 'rwse':
            pos = data.random_walks[:, :self.pos_dim]
            x = torch.cat([self.feat_enc(x), self.pos_enc(pos)], dim=-1)
        elif self.pos_method == 'spec-attn': 
            # process eigenvalues and eigenvectors: (Note: using k=N runs out of memory!)
            eigenvalues = torch.broadcast_to(data.eigenvalues[1:(1+self.pos_dim)], (data.eigenvalues.shape[0], self.pos_dim))[:, :, None] # broadcast to N x k x 1
            eigenvectors = data.eigenvectors[:, 1:(self.pos_dim+1), None] # expand dims to N x k x 1

            # concatenate eigenvectors and eigenvalues N x k x 2
            pos = torch.cat([eigenvectors, eigenvalues], dim=-1) 

            # feed position encoding to pos-encoder network: (N x k x 2) --> Linear --> (N x k x d) --> Attention --> (N x k x d)
            for layer in self.pos_encs: 
                pos = layer(pos)

            # sum-pool position encoding: (N x k x d) --> sum-pool over dimension 1 --> N x d
            pos = torch.sum(pos, dim=1)

            # concatenate feature encoding and position encoding
            x = torch.cat([self.feat_enc(x), pos], dim=-1)
        elif self.pos_method == 'signnet':
            # process eigenvalues and eigenvectors
            eigenvalues = torch.broadcast_to(data.eigenvalues[:self.pos_dim], (data.eigenvalues.shape[0], self.pos_dim))[:, :, None] # broadcast to N x k x 1
            eigenvectors = data.eigenvectors[:, 1:(1+self.pos_dim)][:, :, None] # broadcast to N x k x 1

            pos_pos = torch.cat([eigenvectors, eigenvalues], dim=-1) 
            pos_neg = torch.cat([-1 * eigenvectors, eigenvalues], dim=-1) 

            # feed positive and negatvive eigenvectors to phi network: (N x k x 2) --> MLP --> (N x k x k)
            for layer in self.pos_encs_phi[:-1]: 
                pos_pos = layer(pos_pos)
                pos_pos = F.relu(pos_pos)
            pos_pos = self.pos_encs_phi[-1](pos_pos)

            for layer in self.pos_encs_phi[:-1]: 
                pos_neg = layer(pos_neg)
                pos_neg = F.relu(pos_neg)
            pos_neg = self.pos_encs_phi[-1](pos_neg)

            # sum positive and negatvive representations
            pos = pos_pos + pos_neg

            # flatten the hidden dimension: (N x k x k) --> N x k * k
            pos = torch.flatten(pos, start_dim=1, end_dim=2)

            # feed position encoding to rho network: (N x k * k) --> MLP --> (N x k)
            for layer in self.pos_encs_rho[:-1]: 
                pos = layer(pos)
                pos = F.relu(pos)
            pos = self.pos_encs_rho[-1](pos)

            # concatenate feature encoding and position encoding
            x = torch.cat([self.feat_enc(x), pos], dim=-1)
        else: 
            raise NotImplementedError()

        for encoder in self.hidden_encs:
            x = encoder(x)

        x = self.readout_layer(x)
        x = F.log_softmax(x, dim=-1)
        # x = x.flatten(0, 1) # we need to flatten the batch dimension for main_train.py
        
        return x

class GraphTransformer_LLPE(nn.Module):
    '''
    Graph Transformer from Dwivedi et al. 2020
        - adds positional encodings instead of concatenate as in original AAAI implementation
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







