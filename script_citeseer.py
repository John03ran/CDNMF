import os
import random
import pickle
import numpy as np
import linecache
import matplotlib.pyplot as plt
import torch
from time import perf_counter as t
from Utils.evaluate import clusterscores
from Dataset.dataset import Dataset
from Model.my_model import Model
from PreTrainer.pretrainer import PreTrainer
from Utils import gpu_info

def train(model: Model, graph, optimizer):
    optimizer.zero_grad()
    V = model()

    loss, loss1, loss2, loss3, loss4, loss5 = model.loss(graph)
    loss.backward()
    optimizer.step()

    y_pred = np.argmax(V.detach().cpu().numpy(), axis=0)
    y_true = graph.L.detach().cpu().numpy()
    # print(y_pred)
    scores = clusterscores(y_pred, y_true)

    return loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), scores


if __name__=='__main__':

    random.seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset_config = {'feature_file': './Database/citeseer/features.txt',
                      'graph_file': './Database/citeseer/edges.txt',
                      'walks_file': './Database/citeseer/walks.txt',
                      'label_file': './Database/citeseer/group.txt',
                      'device': device}
    graph = Dataset(dataset_config)

    # Default hyperparameters (or from Optuna if you run it later)
    net_h1 = 256
    net_h2 = 32
    att_h1 = 400
    att_h2 = 100
    
    net_shape = [net_h1, net_h2, graph.num_classes]
    att_shape = [att_h1, att_h2, graph.num_classes]
    
    pretrain_params_path = './Log/citeseer/pretrain_params_best.pkl'
    
    # Construct pretrain params from cache
    print("Constructing pretrain params from cache...")
    cache_dir = './Log/citeseer/cache'
    net_cache_file = os.path.join(cache_dir, f'net_{net_h1}_{net_h2}.pkl')
    att_cache_file = os.path.join(cache_dir, f'att_{att_h1}_{att_h2}.pkl')
    
    if os.path.exists(net_cache_file) and os.path.exists(att_cache_file):
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
        print(f"Saved best pretrain params to {pretrain_params_path}")
    else:
        print("Warning: Cache files not found. You may need to run generate_pretrain_cache.py (modified for citeseer) or enable pre-training.")
        # Fallback to original path if cache construction fails
        pretrain_params_path = './Log/citeseer/pretrain_params.pkl'

    model_config = {
        'device': device,
        'net_shape': net_shape,
        'att_shape': att_shape,
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': True,
        'pretrain_params_path': pretrain_params_path,
        "tau": 1.0521905831005203,
        "conc": 9.888632139642308,
        "negc": 4232,
        "rec": 3.91856241684849,
        "r": 2.6626667859600475,
        "dropout": 0.35223692438317733,
        "learning_rate": 0.002898904290648575,
        "weight_decay": 2.011869449419384e-06,
        'epoch': 1500,
        'run': 10,
        'model_path': './Log/citeseer/citeseer_model.pkl'
    }

    # 'Pre-training stage'
    # pretrainer = PreTrainer(pretrain_config)
    # pretrainer.pre_training(graph.A.detach().cpu().numpy(), 'net')
    # pretrainer.pre_training(graph.X.t().detach().cpu().numpy(), 'att')


    learning_rate = model_config['learning_rate']
    weight_decay = model_config['weight_decay']


    start = t()
    prev = start

    M = []
    N = []
    
    # 'Fine-tuning stage'
    all_loss_history = []
    all_acc_history = []
    all_nmi_history = []

    for i in range(model_config['run']):

        model = Model(model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        loss_history = []
        acc_history = []
        nmi_history = []

        print(f"Run {i+1} started...")
        best_acc_run = 0
        best_nmi_run = 0
        
        for epoch in range(1, model_config['epoch']):
            loss, loss1, loss2, loss3, loss4, loss5, scores = train(model, graph, optimizer)
            
            loss_history.append(loss)
            acc_history.append(scores['ACC'])
            nmi_history.append(scores['NMI'])
            
            if scores['ACC'] > best_acc_run:
                best_acc_run = scores['ACC']
                best_nmi_run = scores['NMI']

            if epoch % 50 == 0:
                print(f"Run {i+1}, Epoch {epoch}: Loss {loss:.4f}, ACC {scores['ACC']:.4f}, NMI {scores['NMI']:.4f}")

            now = t()
            prev = now

        M.append(best_acc_run)
        N.append(best_nmi_run)
        
        all_loss_history.append(loss_history)
        all_acc_history.append(acc_history)
        all_nmi_history.append(nmi_history)
        
    # Plotting All Runs
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    for i, history in enumerate(all_loss_history):
        plt.plot(history, alpha=0.3, label=f'Run {i+1}')
    plt.title('Training Loss (All Runs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    for i, history in enumerate(all_acc_history):
        plt.plot(history, alpha=0.3, label=f'Run {i+1}')
    plt.title('Training ACC (All Runs)')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')

    plt.subplot(1, 3, 3)
    for i, history in enumerate(all_nmi_history):
        plt.plot(history, alpha=0.3, label=f'Run {i+1}')
    plt.title('Training NMI (All Runs)')
    plt.xlabel('Epoch')
    plt.ylabel('NMI')
    
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    plt.savefig('./figures/citeseer_training_all_runs.png')
    print("All runs curves saved to ./figures/citeseer_training_all_runs.png")

    # Plotting Mean Curves
    mean_loss = np.mean(all_loss_history, axis=0)
    mean_acc = np.mean(all_acc_history, axis=0)
    mean_nmi = np.mean(all_nmi_history, axis=0)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mean_loss, color='blue', linewidth=2)
    plt.title('Mean Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(mean_acc, label='Mean ACC', color='green', linewidth=2)
    plt.plot(mean_nmi, label='Mean NMI', color='orange', linewidth=2)
    plt.title('Mean Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.savefig('./figures/citeseer_training_mean.png')
    print("Mean curves saved to ./figures/citeseer_training_mean.png")

    print('ACC: ', np.mean(M), '; NMI: ', np.mean(N))
    print("=== Final ===")





