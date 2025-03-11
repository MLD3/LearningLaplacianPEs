# load python modules
import os
import argparse
import numpy as np
import pickle as pkl
import networkx as nx
from sklearn.metrics import roc_auc_score

import torch_geometric
import torch_geometric.utils as utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# load custom modules
from models.gt import * 
from models.mpnns import *
from generation.utils import *

SEED = 2023
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    
    # dataset parameters
    parser.add_argument('--device_idx', type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--data_name', type=str, default='Cora')
    
    # model hyperparameters
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--h_dim', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--K', type=int)
    parser.add_argument('--l12', type=float)
    parser.add_argument('--ln', type=float)
    parser.add_argument('--model_idx', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--pos_dim', type=int, default=1)
    parser.add_argument('--pos_method', type=str, default='lap-base')
    
    # load arguments
    args = parser.parse_args()
    device_idx = args.device_idx
    data_name = args.data_name
    save_path = args.save_path
    dropout = args.dropout
    pos_dim = args.pos_dim
    h_dim = args.h_dim
    num_layers = args.num_layers
    lr = args.lr
    K = args.K
    l12 = args.l12
    ln = args.ln
    model_idx = args.model_idx
    model_name = args.model_name
    pos_method = args.pos_method
    
    # create model path and data path
    model_path = f'{data_name}_{model_name}_{pos_method}_{model_idx}.pkl'
    data_path = f'/data/mbito/benchmarks_proc/gpe_heterophily/{data_name}.pkl'

    # load the data
    device = torch.device(f'cuda:{device_idx}')
    with open(data_path, 'rb') as file: 
        data = pkl.load(file).to(device)
    
    # iterate over random data splits
    valid_accs = []
    test_accs = []
    for s in range(10): 
        data = add_masks(data, train_split=.6, valid_split=.2, state=s).to(device)
        in_dim = data.x.shape[1]
        out_dim = torch.unique(data.y).shape[0]

        if pos_method in ['lap-full', 'elastic']: 
            pos_dim=data.x.shape[0]

        # build the model and optimizer
        if model_name == 'GT_LLPE': 
            model = GraphTransformer_LLPE(in_dim=in_dim, pos_dim=pos_dim, h_dim=h_dim, out_dim=out_dim, K=K, num_layers=num_layers, dropout=dropout, layer_norm_eps=ln).to(device)
        elif model_name == 'SAGE_LLPE': 
            model = SAGE_LLPE(in_dim=in_dim, pos_dim=pos_dim, h_dim=h_dim, out_dim=out_dim, K=K, num_layers=num_layers, dropout=dropout).to(device)
        elif model_name == 'MLP_LLPE': 
            model = MLP_LLPE(in_dim=in_dim, pos_dim=pos_dim, h_dim=h_dim, out_dim=out_dim, K=K, num_layers=num_layers, dropout=dropout).to(device)
        else: 
            raise NotImplementedError()

        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # initializations
        epochs = 2000
        early_stopping = 50
        stop_counter = 0
        best_loss = float('inf')
        MIN_DELTA = 0

        # model training loop
        model.train()
        for epoch in range(epochs): # full batch mode
            if early_stopping == stop_counter:
                break

            optimizer.zero_grad()
            
            if model_name in ['MLP_LLPE', 'GCN_LLPE', 'SAGE_LLPE', 'FAGCN_LLPE', 'GT_LLPE', 'GCN2_LLPE']: 
                out, eigenvalues = model(data, device)
                l1_loss = torch.linalg.vector_norm(eigenvalues, 1)
                l2_loss = torch.linalg.vector_norm(eigenvalues, 2)
                train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) + l1_loss * l12  + l2_loss * l12
            else: 
                out = model(data)
                train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

            # early stopping count
            valid_loss = F.nll_loss(out[data.valid_mask, :], data.y[data.valid_mask])
            if valid_loss < best_loss - MIN_DELTA: 
                best_loss = valid_loss
                stop_counter = 0
                
                # save the best model for evaluation
                # torch.save(model, os.path.join('/data/mbito/models/', model_path))
                torch.save(model.state_dict(), os.path.join('/data/mbito/models/', model_path))
            else: 
                stop_counter+=1

            train_loss.backward()
            optimizer.step()

        # load the best model for evaluation
        # best_model = torch.load(os.path.join('/data/mbito/models/', model_path))
        model.load_state_dict(torch.load(os.path.join('/data/mbito/models/', model_path), weights_only=True))
        model.eval()
        if model_name in ['MLP_LLPE', 'GCN_LLPE', 'SAGE_LLPE', 'GT_LLPE', 'FAGCN_LLPE', 'GCN2_LLPE']: 
            pred = model(data, device)[0]
        else: 
            pred = model(data)
        
        if data_name not in ['Minesweeper', 'Tolokers', 'Questions']: 
            pred = pred.argmax(dim=1)
            valid_correct = (pred[data.valid_mask] == data.y[data.valid_mask]).sum()
            test_correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            valid_accs.append(int(valid_correct) / int(data.valid_mask.sum()))
            test_accs.append(int(test_correct) / int(data.test_mask.sum()))
        else:
            valid_accs.append(roc_auc_score(data.y[data.valid_mask].detach().cpu().numpy(), pred[data.valid_mask, 1].detach().cpu().numpy()))
            test_accs.append(roc_auc_score(data.y[data.test_mask].detach().cpu().numpy(), pred[data.test_mask, 1].detach().cpu().numpy()))
    
    # save results
    print(f'Finished training model {model_idx}!')
        
    save_dict = vars(args)
    save_dict['valid_accs'] = valid_accs
    save_dict['test_accs'] = test_accs
    save_dict['mean_valid_acc'] = np.mean(valid_accs)
    save_dict['mean_test_acc'] = np.mean(test_accs)
    save_dict['std_valid_acc'] = np.std(valid_accs)
    save_dict['std_test_acc'] = np.std(test_accs)
    with open(os.path.join(save_path, model_path), 'wb') as file: 
        pkl.dump(save_dict, file)




