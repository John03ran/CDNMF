import os
import random
import numpy as np
import torch
import optuna
from time import perf_counter as t
from Utils.evaluate import clusterscores
from Dataset.dataset import Dataset
from Model.my_model import Model
from PreTrainer.pretrainer import PreTrainer

import pickle

dataset_name = 'cora'  # Change as needed: 'cora', 'citeseer', 'pubmed'

# Ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train(model: Model, graph, optimizer):
    optimizer.zero_grad()
    V = model()

    loss, loss1, loss2, loss3, loss4, loss5 = model.loss(graph)
    loss.backward()
    optimizer.step()

    y_pred = np.argmax(V.detach().cpu().numpy(), axis=0)
    y_true = graph.L.detach().cpu().numpy()
    scores = clusterscores(y_pred, y_true)

    return loss.item(), scores

def objective(trial):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Dataset configuration (Using Pubmed as default based on context)
    # You can change this to 'cora' or 'citeseer'
    dataset_config = {
        'feature_file': f'./Database/{dataset_name}/features.txt',
        'graph_file': f'./Database/{dataset_name}/edges.txt',
        'walks_file': f'./Database/{dataset_name}/walks.txt',
        'label_file': f'./Database/{dataset_name}/group.txt',
        'device': device
    }
    graph = Dataset(dataset_config)

    
    # Hyperparameters to optimize
    tau = trial.suggest_float('tau', 0.1, 2.0)
    conc = trial.suggest_float('conc', 0.1, 10.0)
    negc = trial.suggest_int('negc', 50, 5000)
    rec = trial.suggest_float('rec', 0.1, 5.0)
    r = trial.suggest_float('r', 0.1, 5.0)
    dropout = trial.suggest_float('dropout', 0.1, 0.9)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Network Architecture Search
    net_h1 = trial.suggest_categorical('net_h1', [256, 512, 1024]) #[256, 512, 1024]
    net_h2 = trial.suggest_categorical('net_h2', [32, 64, 128]) #[32, 64, 128]
    att_h1 = trial.suggest_categorical('att_h1', [100, 200, 400]) #[100, 200, 400]
    att_h2 = trial.suggest_categorical('att_h2', [50, 100, 200]) #[50, 100, 200]
    '''
    tau=1.2710687360404127
    conc=4.287434090993711
    negc=4456
    rec=4.869836504444923
    r=0.5580075117131884
    dropout = trial.suggest_float('dropout', 0.1, 0.9)
    learning_rate=0.0008202897737101775
    weight_decay=0.00017907435848601785
    net_h1=256
    net_h2=128
    att_h1=100
    att_h2=200
    '''

    net_shape = [net_h1, net_h2, graph.num_classes]
    att_shape = [att_h1, att_h2, graph.num_classes]
    
    print(f"Trial {trial.number} started. Params: {trial.params}")

    # Construct pretrain params from cache
    cache_dir = f'./Log/{dataset_name}/cache'
    net_cache_file = os.path.join(cache_dir, f'net_{net_h1}_{net_h2}.pkl')
    att_cache_file = os.path.join(cache_dir, f'att_{att_h1}_{att_h2}.pkl')
    
    # Temporary file for this trial
    pretrain_params_path = f'./Log/{dataset_name}/pretrain_params_trial_{trial.number}.pkl'
    
    # Merge cache files
    U_init_merged = {}
    V_init_merged = {}
    
    with open(net_cache_file, 'rb') as f:
        U_net, V_net = pickle.load(f)
        U_init_merged.update(U_net)
        V_init_merged.update(V_net)
        
    with open(att_cache_file, 'rb') as f:
        U_att, V_att = pickle.load(f)
        U_init_merged.update(U_att)
        V_init_merged.update(V_att)
        
    with open(pretrain_params_path, 'wb') as f:
        pickle.dump([U_init_merged, V_init_merged], f, protocol=pickle.HIGHEST_PROTOCOL)

    model_config = {
        'device': device,
        'net_shape': net_shape,
        'att_shape': att_shape,
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': pretrain_params_path,
        'tau': tau,
        'conc': conc,
        'negc': negc,
        'rec': rec,
        'r': r,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'epoch': 400, # Reduced epochs for faster optimization
        'run': 5,     # Reduced runs for faster optimization
        'model_path': f'./Log/{dataset_name}/{dataset_name}_model_optuna_{trial.number}.pkl'
    }

    # Training loop
    # We run fewer times for optimization speed
    accuracies = []
    nmis = []

    try:
        for i in range(model_config['run']):
            set_seed(42 + i) # Different seed for each run
            model = Model(model_config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_acc_run = 0
            
            for epoch in range(1, model_config['epoch'] + 1):
                loss, scores = train(model, graph, optimizer)
                
                if scores['ACC'] > best_acc_run:
                    best_acc_run = scores['ACC']
            
            accuracies.append(best_acc_run)
            print(f"Trial {trial.number}, Run {i+1}/{model_config['run']} completed. Best ACC: {best_acc_run:.4f}")
    finally:
        # Cleanup pretrain params to save space
        if os.path.exists(pretrain_params_path):
            os.remove(pretrain_params_path)

    mean_acc = np.mean(accuracies)
    print(f"Trial {trial.number} finished. Mean ACC: {mean_acc:.4f}")
    return mean_acc

if __name__ == '__main__':
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    print("Starting optimization...")
    study.optimize(objective, n_trials=1000) # Adjust n_trials as needed

    print("Number of finished trials: ", len(study.trials))
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best params
    import json
    with open(f'best_params_optuna_{dataset_name}.json', 'w') as f:
        json.dump(trial.params, f, indent=4)
    print(f"Best params saved to best_params_optuna_{dataset_name}.json")