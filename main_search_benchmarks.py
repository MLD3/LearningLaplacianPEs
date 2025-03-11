import numpy as np
import subprocess
import argparse
import os
import time
import shutil

def get_lrs(base, mult):
    lrs = []
    for b in base: 
        for m in mult: 
            lrs.append(b * m)
    return lrs

def format_command(command, hyperparameters, device_idx, model_idx, model_name, data_name, save_path): 
    args = [command]
    for key in hyperparameters.keys(): 
        args.append(f' --{key} {hyperparameters[key]}')
        
    args.append(f' --device_idx {device_idx}')
    args.append(f' --model_idx {model_idx}')
    args.append(f' --model_name {model_name}')
    args.append(f' --data_name {data_name}')
    args.append(f' --save_path {save_path}')
    
    string = ''
    for arg in args: 
        string += arg
        
    return string

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--lrs_b', type=float, nargs='+')
    parser.add_argument('--lrs_m', type=int, nargs='+')
    parser.add_argument('--dropouts', type=float, nargs='+')
    parser.add_argument('--h_dims', type=int, nargs='+')
    parser.add_argument('--num_layers', type=int, nargs='+')
    parser.add_argument('--Ks', type=int, nargs='+', default=[1])
    parser.add_argument('--l12s', type=float, nargs='+', default=[0])
    parser.add_argument('--lns', type=float, nargs='+', default=[0])
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--pos_dims', type=int, nargs='+', default=[1])
    parser.add_argument('--pos_method', type=str, default='lap-base')

    # dataset and search parameters
    parser.add_argument('--command', type=str, default='python3 main_train_benchmarks.py')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--data_name', type=str, default='Cora')
    parser.add_argument('--exp_name', type=str, default='benchmarks')

    # load arguments
    args = parser.parse_args()
    lrs_b = list(args.lrs_b)
    lrs_m = list(args.lrs_m)
    dropouts = list(args.dropouts)
    pos_dims = list(args.pos_dims)
    h_dims = list(args.h_dims)
    num_layers = list(args.num_layers)
    Ks = list(args.Ks)
    l12s = list(args.l12s)
    lns = list(args.lns)
    model_name = args.model_name
    pos_method = args.pos_method
    command = args.command
    gpus = args.gpus
    data_name = args.data_name
    exp_name = args.exp_name

    print(f'Starting hyperparameter search for model {model_name}-{pos_method} on dataset {data_name}...')
    
    # create save directory
    save_path = os.path.join('/home/mbito/project_gpe_heterophily/results/', exp_name)
    if not os.path.exists(save_path): 
        os.mkdir(save_path)

    # create model directory
    model_path = '/data/mbito/models/'
    if not os.path.exists(model_path): 
        os.mkdir(model_path)

    # create search grid
    grid = []
    hyperparameters = {}
    hyperparameters['lrs'] = get_lrs(lrs_b, lrs_m)
    hyperparameters['dropouts'] = dropouts
    hyperparameters['h_dims'] = h_dims
    hyperparameters['num_layers'] = num_layers
    hyperparameters['pos_dims'] = pos_dims
    hyperparameters['lns'] = lns
    hyperparameters['pos_method'] = pos_method
    
    if model_name in ['MLP_LLPE', 'GCN_LLPE', 'SAGE_LLPE', 'GT_LLPE', 'FAGCN_LLPE', 'GCN2_LLPE']: 
        hyperparameters['Ks'] = Ks
        hyperparameters['l12s'] = l12s
        for lr in hyperparameters['lrs']: 
            for K in hyperparameters['Ks']: 
                for pd in hyperparameters['pos_dims']: 
                    for hd in hyperparameters['h_dims']:
                        for nl in hyperparameters['num_layers']:
                            for dr in hyperparameters['dropouts']: 
                                for l12 in hyperparameters['l12s']: 
                                    for ln in hyperparameters['lns']: 
                                        grid.append({'lr': lr, 'pos_dim':pd, 'h_dim': hd, 'K': K, 'l12': l12, 'num_layers': nl, 'dropout': dr, 'ln':ln,'pos_method': pos_method})
    else: 
        for lr in hyperparameters['lrs']: 
            for pos_dim in hyperparameters['pos_dims']: 
                for h_dim in hyperparameters['h_dims']:
                    for nl in hyperparameters['num_layers']:
                        for dr in hyperparameters['dropouts']: 
                            for ln in hyperparameters['lns']: 
                                grid.append({'lr': lr, 'h_dim': h_dim, 'pos_dim': pos_dim, 'num_layers': nl, 'dropout': dr, 'pos_method': pos_method, 'ln':ln})
    
    # run search in parallel
    available_gpus = {gpu: subprocess.Popen('', shell=True) for gpu in gpus} # open a process for each available gpu
    model_idx = 0
    while len(grid) > 0: 
        for gpu in available_gpus.keys(): 
            if available_gpus[gpu].poll() == 0 and len(grid) > 0: 
                hypers = grid.pop()
                command = format_command(command, hypers, gpu, model_idx, model_name, data_name, save_path)
                available_gpus[gpu] = subprocess.Popen(command, shell=True)
                model_idx += 1
        
        time.sleep(1) # wait 1 seconds to check for available gpus

    # wait for all processes to finish, so that next search doesn't overlap with current one
    for gpu in available_gpus.keys():
        available_gpus[gpu].wait()

    # delete model path
    shutil.rmtree(model_path)




