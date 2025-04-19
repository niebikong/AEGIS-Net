import time
import os
import numpy as np
import faiss
import torch
from util import metrics  

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

IN_DATASET = 'malicious_TLS'   # choices=['malicious_TLS', 'IDS_2017', 'IDS_2018_friday']
OOD_DATASETS = ['Darknet2020'] 
CACHE_DIR = "AEGIS-Net/cache"  

def load_features(dataset_name, split='train'):
    cache_path = os.path.join(CACHE_DIR, f"{dataset_name}_{split}_features.npy")
    data = np.load(cache_path, allow_pickle=True).item()
    return data['feat_log'], data['score_log'], data['label_log']

def preprocess_features(feat_matrix, layer_range=slice(1792, 1920)):
    # L2
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    
    # Last 128 dim
    processed = np.ascontiguousarray(
        normalizer(feat_matrix[:, layer_range]), 
        dtype=np.float32
    )
    return processed

def build_faiss_index(features, nlist=100, nprobe=20):
    dim = features.shape[1]
    
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    assert len(features) >= nlist
    index.train(features)
    
    index.add(features)
    index.nprobe = nprobe  
    
    return index

def calculate_ood_scores(index, features, k=50):

    # parallel computing threads
    faiss.omp_set_num_threads(80)  
    
    # Search for k-nearest neighbors 
    distances, _ = index.search(features, k)
    return -distances[:, -1]

if __name__ == "__main__":
    # Loading known dataset features
    ftrain, _, _ = load_features(IN_DATASET, 'train')
    ftest, _, _ = load_features(IN_DATASET, 'val')
    
    # Loading unknown dataset features
    ood_features = {}
    for ood_name in OOD_DATASETS:
        cache_path = os.path.join(CACHE_DIR, f"{ood_name}_features.npy")
        data = np.load(cache_path, allow_pickle=True).item()
        ood_features[ood_name] = data['feat_log']

    # selection of last 128-dimensional features
    ftrain = preprocess_features(ftrain)
    ftest = preprocess_features(ftest)
    food = {k: preprocess_features(v) for k, v in ood_features.items()}

    # Building the FAISS Index
    print("\n Building the FAISS Index...")
    ann_index = build_faiss_index(ftrain)
    
    # Evaluating the detection effect of different K values
    K_LIST = [50]  # Expandable to multiple K-value comparisons
    for K in K_LIST:
        start_time = time.time()
        
        in_scores = calculate_ood_scores(ann_index, ftest, K)
        
        all_results = []
        for ood_name, ood_feat in food.items():
            ood_scores = calculate_ood_scores(ann_index, ood_feat, K)
            
            results = metrics.cal_metric(in_scores, ood_scores)
            all_results.append(results)
            
            print(f"\n{OOD_DATASETS[0]} detection result:")

        metrics.print_all_results(all_results, OOD_DATASETS, f'ANN-IVFFlat k={K}')
        print(f'Time: {time.time()-start_time:.2f} seconds')