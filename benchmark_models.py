import os
import time
import torch
import numpy as np
import importlib.util
import sys
import gc
import pickle
import signal
from Dataset.dataset import Dataset

# Define configurations for each dataset based on script_*.py
DATASET_CONFIGS = {
    'citeseer': {
        'dataset': {
            'feature_file': './Database/citeseer/features.txt',
            'graph_file': './Database/citeseer/edges.txt',
            'walks_file': './Database/citeseer/walks.txt',
            'label_file': './Database/citeseer/group.txt',
        },
        'model': {
            'net_h1': 1024, 'net_h2': 128, 'att_h1': 400, 'att_h2': 200,
            "tau": 1.567, "conc": 3.807, "negc": 3962, "rec": 3.450, "r": 0.893,
            "dropout": 0.644, "learning_rate": 0.001, "weight_decay": 0.0005
        }
    },
    'cora': {
        'dataset': {
            'feature_file': './Database/cora/features.txt',
            'graph_file': './Database/cora/edges.txt',
            'walks_file': './Database/cora/walks.txt',
            'label_file': './Database/cora/group.txt',
        },
        'model': {
            'net_h1': 1024, 'net_h2': 128, 'att_h1': 400, 'att_h2': 200,
            "tau": 0.615, "conc": 9.324, "negc": 2349, "rec": 0.923, "r": 1.292,
            "dropout": 0.303, "learning_rate": 0.002, "weight_decay": 8.5e-06
        }
    },
    'pubmed': {
        'dataset': {
            'feature_file': './Database/pubmed/features.txt',
            'graph_file': './Database/pubmed/edges.txt',
            'walks_file': './Database/pubmed/walks.txt',
            'label_file': './Database/pubmed/group.txt',
        },
        'model': {
            'net_h1': 1024, 'net_h2': 128, 'att_h1': 400, 'att_h2': 200,
            "tau": 0.201, "conc": 1.460, "negc": 4365, "rec": 4.955, "r": 3.825,
            "dropout": 0.667, "learning_rate": 0.011, "weight_decay": 1.76e-06
        }
    }
}

MODELS = {
    'No Optimization': 'Model/my_model_origin.py',
    'Core Opt + Loss Opt': 'Model/my_model_opt_loss.py',
    'Full Optimization': 'Model/my_model.py'
}

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def load_model_class(path):
    name = os.path.basename(path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model

def create_dummy_pretrain_params(filepath, net_shape, att_shape, num_nodes, num_feas):
    U_init = {}
    V_init = {}
    
    # Net
    dims = [num_nodes] + net_shape
    for i in range(len(net_shape)):
        name = f'net{i}'
        U_init[name] = np.random.rand(dims[i], dims[i+1]).astype(np.float32)
    
    name = f'net{len(net_shape)-1}'
    V_init[name] = np.random.rand(dims[-1], num_nodes).astype(np.float32)
    
    # Att
    dims = [num_feas] + att_shape
    for i in range(len(att_shape)):
        name = f'att{i}'
        U_init[name] = np.random.rand(dims[i], dims[i+1]).astype(np.float32)
        
    name = f'att{len(att_shape)-1}'
    V_init[name] = np.random.rand(dims[-1], num_nodes).astype(np.float32)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump([U_init, V_init], f)

def run_benchmark(dataset_name, model_name, model_path, device, epochs=10, timeout=60):
    print(f"--- Benchmarking {model_name} on {dataset_name} ---")
    
    # Load Dataset
    config = DATASET_CONFIGS[dataset_name]
    ds_config = config['dataset']
    ds_config['device'] = device
    
    try:
        graph = Dataset(ds_config)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Load Model Class
    try:
        ModelClass = load_model_class(model_path)
    except Exception as e:
        print(f"Error loading model class from {model_path}: {e}")
        return None

    # Configure Model
    m_params = config['model']
    net_shape = [m_params['net_h1'], m_params['net_h2'], graph.num_classes]
    att_shape = [m_params['att_h1'], m_params['att_h2'], graph.num_classes]
    
    # Create dummy pretrain params
    dummy_params_path = f'./temp_params_{dataset_name}.pkl'
    create_dummy_pretrain_params(dummy_params_path, net_shape, att_shape, graph.num_nodes, graph.num_feas)
    
    model_config = {
        'device': device,
        'net_shape': net_shape,
        'att_shape': att_shape,
        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'is_init': False, # Disable pretrain loading for speed test
        'pretrain_params_path': dummy_params_path,
        'model_path': 'dummy_model_path',
        **m_params
    }

    # Set signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        try:
            model = ModelClass(model_config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=m_params['learning_rate'])
        except Exception as e:
            print(f"Error initializing model: {e}")
            if os.path.exists(dummy_params_path):
                os.remove(dummy_params_path)
            return None
            
        if os.path.exists(dummy_params_path):
            os.remove(dummy_params_path)

        # Warmup
        print("Warming up...")
        for _ in range(20):
            optimizer.zero_grad()
            _ = model()
            loss, _, _, _, _, _ = model.loss(graph)
            loss.backward()
            optimizer.step()

        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        print(f"Running {epochs} epochs...")
        times = []
        for epoch in range(epochs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            optimizer.zero_grad()
            _ = model()
            loss, _, _, _, _, _ = model.loss(graph)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append(end - start)
            print(f"Epoch {epoch+1}: {end - start:.4f}s", end='\r')
        print("")

        avg_time = np.mean(times)
        max_memory = 0
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
        print(f"Average time per epoch: {avg_time:.4f}s")
        print(f"Max memory allocated: {max_memory:.2f}MB")
        return {'time': avg_time, 'memory': max_memory}

    except TimeoutException:
        print(f"\nTimeout after {timeout}s!")
        return "Timeout"
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("\nOOM detected!")
            torch.cuda.empty_cache()
            return "OOM"
        else:
            print(f"\nRuntimeError: {e}")
            return "Error"
    except Exception as e:
        print(f"\nError: {e}")
        return "Error"
    finally:
        signal.alarm(0) # Disable alarm

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    datasets = ['citeseer', 'cora', 'pubmed']
    
    for ds in datasets:
        results[ds] = {}
        for m_name, m_path in MODELS.items():
            # Run benchmark with timeout
            t = run_benchmark(ds, m_name, m_path, device, epochs=20, timeout=30)
            results[ds][m_name] = t
            
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
            
    print("\n\n=== Final Results (Time / Memory) ===")
    print(f"{'Dataset':<10} | {'No Optimization':<30} | {'Core Opt + Loss Opt':<30} | {'Full Optimization':<30}")
    print("-" * 110)
    for ds in datasets:
        r = results[ds]
        
        def fmt(val):
            if isinstance(val, dict):
                return f"{val['time']:.4f}s / {val['memory']:.1f}MB"
            return str(val)
            
        print(f"{ds:<10} | {fmt(r.get('No Optimization', 'N/A')):<30} | {fmt(r.get('Core Opt + Loss Opt', 'N/A')):<30} | {fmt(r.get('Full Optimization', 'N/A')):<30}")
