# import python modules
import os
import argparse
import numpy as np
import pickle as pkl
import networkx as nx

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Actor
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.datasets import LINKXDataset

# import custom modules
from utils import *

def compatibility_matrix(g, labels): 
    n_classes = np.unique(labels).shape[0]
    compatibility = np.zeros((n_classes, n_classes))
    compatibility_d = np.zeros((n_classes, 1))

    for node in g.nodes: 
        node_label = labels[node]
        for neighbor in g.neighbors(node): 
            compatibility[node_label, labels[neighbor]] += 1
            
        compatibility_d[node_label, 0] += g.degree[node]
        
    compatibility = compatibility / compatibility_d
    
    return compatibility

if __name__ == '__main__': 
    # argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cc', type=int, default=0) # select only the nodes in the largest component
    parser.add_argument('--inc_deg', type=int, default=-1) # add edges based on compatibility matrix for semi-synthetic analysis
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--data_directory', type=str)
    parser.add_argument('--save_directory', type=str)
    
    args = parser.parse_args()
    cc = args.cc
    inc_deg = args.inc_deg
    data_name = args.data_name
    data_directory = args.data_directory
    save_directory = args.save_directory

    # load data from pytorch geometric
    if data_name in ['Cora', 'Citeseer']: 
        data = Planetoid(root=data_directory, name=data_name)[0]
    elif data_name in ['Cora_full', 'Cora_ML', 'Citeseer_full', 'DBLP', 'Pubmed']: 
        data_name_c = data_name
        if data_name in ['Cora_full', 'Citeseer_full']: 
            data_name_c = data_name.split('_')[0]
        data = CitationFull(root=data_directory, name=data_name_c)[0]
    elif data_name in ['Cornell', 'Texas', 'Wisconsin']: 
        data = WebKB(root=data_directory, name=data_name)[0]
    elif data_name in ['Computers', 'Photo']: 
        data = Amazon(root=data_directory, name=data_name)[0]
    elif data_name in ['Actor']: 
        data = Actor(root='~/data/benchmarks/Actor')[0]
    elif data_name in ['Minesweeper', 'Tolokers', 'Questions', 'Roman-empire', 'Amazon-ratings']: 
        data = HeterophilousGraphDataset(root=data_directory, name=data_name)[0]
    elif data_name in ['Squirrel', 'Chameleon']:
        # data = np.load(f'{data_directory}.npz')
        # x = torch.tensor(data['node_features'])
        # y = torch.tensor(data['node_labels'])
        # edge_index = torch_geometric.utils.to_undirected(torch.tensor(data['edges']).T) # make sure undirected!
        # data = Data(x=x, y=y, edge_index=edge_index)
        # data_name = f'{data_name}_filt'
        data = WikipediaNetwork(root=data_directory, name=data_name)[0]
    elif data_name in ['CS', 'Physics']: 
        data = Coauthor(root=data_directory, name=data_name)[0]
    elif data_name in ['BlogCatalog', 'Flickr', 'PPI']: 
        data = AttributedGraphDataset(root=data_directory, name=data_name)[0]
    elif data_name in ['Penn94']: 
        data = LINKXDataset(root=data_directory, name=data_name)[0]
        if data_name == 'Penn94': 
            data.y = data.y + 1

    if cc == 1 or inc_deg >= 0: 
        A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze().numpy()
        g = nx.from_numpy_matrix(A)
        largest_cc = max(nx.connected_components(g), key=len)
        g_cc = g.subgraph(largest_cc)
        ids_cc = list(largest_cc)
        y = data.y[ids_cc]
        x = data.x[ids_cc, :]
        g_cc = nx.convert_node_labels_to_integers(g_cc)
        C = compatibility_matrix(g_cc, y.numpy())
        
        if inc_deg >= 0: 
            cand_nodes = {}
            unique_labels = np.unique(y.numpy()).flatten()
            for label in unique_labels: 
                cand_nodes[label] = np.nonzero(y.numpy() == label)[0].flatten()
                
            for node in g_cc.nodes: 
                node_label = y[node]
                # cand_nodes = torch.nonzero(y == node_label).flatten()
                # add_nodes = np.random.choice(cand_nodes.numpy(), size=inc_deg, replace=False)
                # add_edges = [(node, node_a) for node_a in add_nodes]
                # g_cc.add_edges_from(add_edges)
                for i in range(inc_deg): 
                    node_a_label = np.random.choice(unique_labels.shape[0], p=C[node_label, :]/np.sum(C[node_label, :]))
                    node_a = np.random.choice(cand_nodes[node_a_label], size=1, replace=False)[0]
                    g_cc.add_edge(node, node_a)

            data_name = f'{data_name}_dd{inc_deg}'
        else: 
            data_name = f'{data_name}_cc'

        edge_index = torch_geometric.utils.to_undirected(torch.tensor(list(g_cc.edges), dtype=torch.long).T) # makes sure edge indices are undirected
        data = Data(x=x, y=y, edge_index=edge_index)

    # add position encodings
    if data_name in ['Physics', 'Penn94', 'Questions', 'PPI']: 
        data = add_laplacian_eigs_sparse(data, k=1024)
    else: 
        data = add_laplacian_eigs(data, pos_dim=data.x.shape[0])
        # data = add_rwse(data, pos_dim=24)

    # save dataset 
    data_name = f'{data_name}.pkl'
    with open(os.path.join(save_directory, data_name), 'wb') as file: 
        pkl.dump(data, file)

    print(f'Processed dataset {data_name}!')
        



