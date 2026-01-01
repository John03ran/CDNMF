import os
import pickle
import torch
import numpy as np
import scipy.sparse as sp
from Dataset.dataset import Dataset
from PreTrainer.pretrainer import PreTrainer

def generate_cache():
    dataset_name = 'pubmed'  # Change as needed: 'cora', 'citeseer', 'pubmed'
    device = torch.device('cpu')
    
    dataset_config = {
        'feature_file': f'./Database/{dataset_name}/features.txt',
        'graph_file': f'./Database/{dataset_name}/edges.txt',
        'walks_file': f'./Database/{dataset_name}/walks.txt',
        'label_file': f'./Database/{dataset_name}/group.txt',
        'device': device
    }
    graph = Dataset(dataset_config)
    
    cache_dir = f'./Log/{dataset_name}/cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    # Define search space for architecture
    net_h1_options = [256, 512, 1024]
    net_h2_options = [32, 64, 128]
    att_h1_options = [100, 200, 400]
    att_h2_options = [50, 100, 200]
    
    # Pre-train NET parts
    print("Generating NET cache...")
    
    # Convert to sparse matrix to save memory and avoid WSL driver issues
    print("Converting Adjacency matrix to sparse format...")
    A_sparse = sp.csr_matrix(graph.A.detach().cpu().numpy())
    
    for h1 in net_h1_options:
        for h2 in net_h2_options:
            filename = f'net_{h1}_{h2}.pkl'
            filepath = os.path.join(cache_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Skipping {filename}, already exists.")
                continue
                
            print(f"Pre-training {filename}...")
            net_shape = [h1, h2, graph.num_classes]
            # Dummy att_shape, not used for net pretraining
            att_shape = [100, 50, graph.num_classes] 
            
            config = {
                'net_shape': net_shape,
                'att_shape': att_shape,
                'net_input_dim': graph.num_nodes,
                'att_input_dim': graph.num_feas,
                'seed': 42,
                'pre_iterations': 500, 
                'pretrain_params_path': filepath
            }
            
            pt = PreTrainer(config)
            # Only train net, pass sparse matrix
            pt.pre_training(A_sparse, 'net')

    # Pre-train ATT parts
    print("Generating ATT cache...")
    
    # Convert to sparse matrix
    print("Converting Feature matrix (transposed) to sparse format...")
    X_T_sparse = sp.csr_matrix(graph.X.t().detach().cpu().numpy())
    
    for h1 in att_h1_options:
        for h2 in att_h2_options:
            filename = f'att_{h1}_{h2}.pkl'
            filepath = os.path.join(cache_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Skipping {filename}, already exists.")
                continue
                
            print(f"Pre-training {filename}...")
            # Dummy net_shape
            net_shape = [256, 32, graph.num_classes]
            att_shape = [h1, h2, graph.num_classes]
            
            config = {
                'net_shape': net_shape,
                'att_shape': att_shape,
                'net_input_dim': graph.num_nodes,
                'att_input_dim': graph.num_feas,
                'seed': 42,
                'pre_iterations': 1000,
                'pretrain_params_path': filepath
            }
            
            pt = PreTrainer(config)
            # Only train att, pass sparse matrix
            pt.pre_training(X_T_sparse, 'att')

    print("Cache generation complete.")

if __name__ == '__main__':
    generate_cache()
