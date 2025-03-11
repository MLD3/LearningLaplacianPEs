import numpy as np
import networkx as nx

import torch
import torch_geometric

from scipy.sparse.linalg import eigs

def add_masks(data, train_split=.6, valid_split=.2, state=2023): 
    '''
    Adds train, valid, test splits to pytorch data object
    '''
    # use np.random.RandomState(2023) for all random operations
    rs = np.random.RandomState(state)
    
    # create train, valid, test indices
    N = data.x.shape[0]
    indices = np.arange(N)
    indices_train = rs.choice(indices, size=int(N * train_split), replace=False)
    indices_train = np.unique(np.concatenate([indices_train, np.array([0])])) # add zero automatically since nonzero masks it
    
    indices[indices_train] = 0
    indices_valid = rs.choice(np.nonzero(indices)[0], size=int(N * valid_split), replace=False) # use valid_split 
    
    indices[indices_valid] = 0
    indices_test = np.nonzero(indices)[0]
    
    train_mask = np.zeros((N, ))
    train_mask[indices_train] = 1
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    
    valid_mask = np.zeros((N, ))
    valid_mask[indices_valid] = 1
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    
    test_mask = np.zeros((N, ))
    test_mask[indices_test] = 1
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask
    
    # check for data leakage
    indices = np.sort(np.concatenate([np.nonzero(data.train_mask.numpy())[0], np.nonzero(data.valid_mask.numpy())[0], np.nonzero(data.test_mask.numpy())[0]]))
    for i in range(N): 
        assert i == indices[i], 'Data leakage! Check train, valid, test masks!'
    
    return data

def add_rwse(data, pos_dim): 
    """
    Initializing positional encoding with RWSE (diagonal of m-step random walk matrix) (Dwivedi et al., 2022, Rampasek et al., 2022, Mueller et al., 2024)

    Code adapted from https://github.com/vijaydwivedi75/gnn-lspe/blob/main/data/molecules.py
    """
    # construct random walk matrix: RW := D^-1 x A (since we are only concerned with diagonals, rowwise is equivalent to colwise)
    # A = np.array(torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index).todense())
    A = torch_geometric.utils.to_dense_adj(data.edge_index).numpy()
    D_vec = np.sum(A, axis=1).flatten()
    D_vec[D_vec == 0] = 1
    D_vec_inv = 1 / D_vec
    D_inv = np.diag(D_vec_inv)
    RW = A @ D_inv
    RW_k = RW.copy()

    # iterate k-steps of the random walk matrix and add diagonals to random_walks
    random_walks = [RW.diagonal()]
    for k in range(pos_dim - 1): 
        RW_k = RW_k @ RW
        random_walks.append(RW_k.diagonal())

    # concatenate the encodings and add to pytorch dataset
    data.random_walks = torch.from_numpy(np.concatenate(random_walks, axis=-1)).float()
    
    return data
    
def add_laplacian_eigs(data, pos_dim):
    '''
    Adds eigenvector and eigenvalue positional encoding to pytorch data object

    Code from https://github.com/cptq/SignNet-BasisNet/blob/main/LearningFilters/utils.py
    '''
    A = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index).todense()
    # A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze().numpy()
    nnodes = A.shape[0]
    D_vec = np.sum(A, axis=1).A1
    D_vec[D_vec == 0] = 1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = np.diag(D_vec_invsqrt_corr)
    L = np.eye(nnodes)-D_invsqrt_corr @ A @ D_invsqrt_corr
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    eigenvalues_torch, eigenvectors_torch = torch.from_numpy(eigenvalues).float(), torch.from_numpy(eigenvectors).float()
    
    data.eigenvectors = eigenvectors_torch[:, :pos_dim]
    data.eigenvalues = eigenvalues_torch[:pos_dim]

    # # add ElasticPE as defined in https://arxiv.org/pdf/2307.07107.pdf
    # eig_mask = eigenvalues > 0
    # eigenvalues_inv = eigenvalues.copy()[eig_mask]
    # # eigenvalues_inv[eigenvalues_inv == 0] = eigenvalues[eigenvalues > 0][0] # set zero eigenvalues to smallest nonzero eigenvalue
    # eigenvalues_inv = 1 / eigenvalues_inv
    # Q = eigenvectors[:, eig_mask] @ np.diag(eigenvalues_inv) @ eigenvectors[:, eig_mask].T
    # # Q = Q - np.broadcast_to(np.diag(Q), Q.shape)
    # # pe_min_ij = np.min(Q, axis=1).reshape(-1, 1)
    # # pe_avg_ij = np.mean(Q, axis=1).reshape(-1, 1)
    # # pe_std_ij = np.std(Q, axis=1).reshape(-1, 1)
    
    # # pe_min_ji = np.min(Q, axis=0).reshape(-1, 1)
    # # pe_std_ji = np.std(Q, axis=0).reshape(-1, 1)

    # # I = A @ Q
    # # pe_avg_int_ij = np.mean(I, axis=1).reshape(-1, 1)
    # # pe_avg_int_ji = np.mean(I, axis=0).reshape(-1, 1)
    # # elastic_pe = np.concatenate([pe_min_ij, pe_avg_ij, pe_std_ij, pe_min_ji, pe_std_ji, pe_avg_int_ij, pe_avg_int_ji], axis=1)
    # # data.elastic_pe = torch.from_numpy(elastic_pe).float()
    # data.elastic_pe = torch.from_numpy(Q).float()

    return data

def add_laplacian_eigs_sparse(data, k):
    '''
    Adds eigenvector and eigenvalue positional encoding to pytorch data object
    '''
    A = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index).todense()
    # A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze().numpy()
    nnodes = A.shape[0]
    D_vec = np.sum(A, axis=1).A1
    D_vec[D_vec == 0] = 1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = np.diag(D_vec_invsqrt_corr)
    L = np.eye(nnodes)-D_invsqrt_corr @ A @ D_invsqrt_corr
    eigenvalues_sm, eigenvectors_sm = eigs(L, k=k, which='SM')
    eigenvalues_lm, eigenvectors_lm = eigs(L, k=k, which='LM')
    
    eigs_sids = np.argsort(eigenvalues_sm) # scipy eigs may return eigenvalues in non-increasing order
    eigenvalues_sm = eigenvalues_sm[eigs_sids]
    eigenvectors_sm = eigenvectors_sm[:, eigs_sids]
    eigs_lids = np.argsort(eigenvalues_lm) # scipy eigs may returns eigenvalues in non-decreasing order
    eigenvalues_lm = eigenvalues_lm[eigs_lids]
    eigenvectors_lm = eigenvectors_lm[:, eigs_lids]
    
    eigenvalues = np.concatenate([np.real(eigenvalues_sm), np.real(eigenvalues_lm)], axis=0)
    eigenvectors = np.concatenate([np.real(eigenvectors_sm), np.real(eigenvectors_lm)], axis=1)
    eigenvalues_torch, eigenvectors_torch = torch.from_numpy(eigenvalues).float(), torch.from_numpy(eigenvectors).float()
    
    data.eigenvectors = eigenvectors_torch
    data.eigenvalues = eigenvalues_torch

    return data

def add_shortest_paths(data): 
    '''
    Adds n x n matrix of all pairs shortest path lenghts
    '''
    A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze().numpy()
    g = nx.from_numpy_matrix(A)
    largest_cc = max(nx.connected_components(g), key=len)
    diameter = nx.diameter(g.subgraph(largest_cc).copy())
    pos_encoding = (diameter + 1) * np.ones((g.number_of_nodes(), g.number_of_nodes()))
    lengths = nx.all_pairs_shortest_path_length(g)
    for node, length in enumerate(lengths): 
        ids_sort = np.sort(np.array(list(length[1].keys())))
        ids_argsort = np.argsort(np.array(list(length[1].keys())))
        pos_encoding[node, ids_sort] = np.array(list(length[1].values()))[ids_argsort]
        # keys = list(dict(sorted(length[1].items())).keys())
        # values = list(dict(sorted(length[1].items())).values())
        # pos_encoding[:, keys] = values

    data.shortest_paths = torch.from_numpy(pos_encoding).int() # nn.Embedding() requires integer argumnents

    return data









