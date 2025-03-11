import os 
import numpy as np
import pickle as pkl

def get_results_benchmarks(path, data_name, model_name, pos_enc, total_ids, last_id=1, verbose=False, history=False): 
    val_results, test_results = np.zeros((total_ids, 2)), np.zeros((total_ids, 2))
    hyperparameters = []
    
    for i in range(total_ids): 
        id_path = os.path.join(path, f'{data_name}_{model_name}_{pos_enc}_{i}.pkl')
        if os.path.exists(id_path): # we are looking at accuracy which we aim to maximize, so this is okay
            with open(id_path, 'rb') as file: 
                results = pkl.load(file)
                val_results[i, 0], val_results[i, 1] = results['mean_valid_acc'], results['std_valid_acc']
                test_results[i, 0], test_results[i, 1] = results['mean_test_acc'], results['std_test_acc']
                hyperparameters.append(results)

    if len(hyperparameters) > 0 : 
        best_id = np.argsort(val_results[:, 0])[-last_id]
        best_val_mean, best_val_std = val_results[:, 0][best_id], val_results[:, 1][best_id]
        best_acc_mean, best_acc_std = test_results[:, 0][best_id], test_results[:, 1][best_id]
        best_hyperparameters = hyperparameters[best_id]
        
        if verbose: 
            if model_name in ['MLP_LLPE', 'SAGE_LLPE', 'GCN_LLPE', 'GT_LLPE', 'FAGCN_LLPE']: 
                print(f"{best_hyperparameters['data_name']} {best_hyperparameters['model_name']} LR {best_hyperparameters['lr']} Pos Dim {best_hyperparameters['pos_dim']} Hid Dim {best_hyperparameters['h_dim']} K {best_hyperparameters['K']} DR {best_hyperparameters['dropout']} L12 {best_hyperparameters['l12']}, Test Performance: {best_acc_mean:.4f} +/- {best_acc_std:.4f}")
            else:
                print(f"{best_hyperparameters['data_name']} {best_hyperparameters['model_name']}-{best_hyperparameters['pos_method']} LR {best_hyperparameters['lr']} Pos Dim {best_hyperparameters['pos_dim']} Hidden Dim {best_hyperparameters['h_dim']}  Num Layers {best_hyperparameters['num_layers']}, Test Performance: {best_acc_mean:.4f} +/- {best_acc_std:.4f}")
    else: 
        best_acc_mean = -1
        best_acc_std = 0

    if history: 
        return best_acc_mean, best_acc_std, np.array(best_hyperparameters['test_accs'])
    else: 
        return best_acc_mean, best_acc_std

def get_results_singlegraph(path, c_size, homophily, model_name, pos_enc, total_ids, power=False, verbose=False): 
    val_results, test_results = np.zeros((total_ids, 2)), np.zeros((total_ids, 2))
    hyperparameters = []
    
    for i in range(total_ids): 
        if power: 
            id_path = os.path.join(path, f'SBM_power_c{c_size}_h{homophily}_{model_name}_{pos_enc}_{i}.pkl')
        else: 
            id_path = os.path.join(path, f'SBM_c{c_size}_h{homophily}_{model_name}_{pos_enc}_{i}.pkl')

        if os.path.exists(id_path): # we are looking at accuracy which we aim to maximize, so this is okay
            with open(id_path, 'rb') as file: 
                results = pkl.load(file)
                val_results[i, 0], val_results[i, 1] = results['mean_valid_acc'], results['std_valid_acc']
                test_results[i, 0], test_results[i, 1] = results['mean_test_acc'], results['std_test_acc']
                hyperparameters.append(results)

    best_id = np.argsort(val_results[:, 0])[-1]
    best_acc_mean, best_acc_std = test_results[:, 0][best_id], test_results[:, 1][best_id]
    best_hyperparameters = hyperparameters[best_id]

    if verbose: 
        print(f"SBM h={best_hyperparameters['homophily']} {best_hyperparameters['model_name']}-{best_hyperparameters['pos_method']} LR {best_hyperparameters['lr']} Pos Dim {best_hyperparameters['pos_dim']} Hidden Dim {best_hyperparameters['h_dim']}  Num Layers {best_hyperparameters['num_layers']}")
    
    return best_acc_mean, best_acc_std

def get_results_multigraph(path, n_graphs, c_size, homophily, model_name, pos_enc, total_ids, verbose=False): 
    val_results, test_results = np.zeros((total_ids, 2)), np.zeros((total_ids, 2))
    hyperparameters = []
    
    for i in range(total_ids): 
        id_path = os.path.join(path, f'SBM_n{n_graphs}_c{c_size}_h{homophily}_{model_name}_{pos_enc}_{i}.pkl')
        if os.path.exists(id_path): # we are looking at accuracy which we aim to maximize, so this is okay
            with open(id_path, 'rb') as file: 
                results = pkl.load(file)
                val_results[i, 0], val_results[i, 1] = results['mean_valid_acc'], results['std_valid_acc']
                test_results[i, 0], test_results[i, 1] = results['mean_test_acc'], results['std_test_acc']
                hyperparameters.append(results)

    best_id = np.argsort(val_results[:, 0])[-1]
    best_acc_mean, best_acc_std = test_results[:, 0][best_id], test_results[:, 1][best_id]
    best_hyperparameters = hyperparameters[best_id]

    if verbose: 
        print(f"SBM h={best_hyperparameters['homophily']} {best_hyperparameters['model_name']}-{best_hyperparameters['pos_method']} LR {best_hyperparameters['lr']} Pos Dim {best_hyperparameters['pos_dim']} Hidden Dim {best_hyperparameters['h_dim']}  Num Layers {best_hyperparameters['num_layers']}")
    
    return best_acc_mean, best_acc_std




