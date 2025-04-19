import time
import os
import torch.nn.functional as F
from Network import DeepResNet
import argparse
import torch.utils.data as Data
from util.datasets import load_data
import numpy as np
from collections import Counter
import torch
import warnings
warnings.filterwarnings("ignore")

def symmetric_noise(y, noise_percent, k=10):
    """
    Apply symmetric noise (uniform noise) to labels
    Randomly replace labels for specified percentage of samples with other classes
    
    Args:
        y: Original label array, shape (n_samples,)
        noise_percent: Noise ratio in [0.0, 1.0]
        k: Total number of classes (default: 10)
    
    Returns:
        y_noise: Noisy label array with symmetric noise
    """
    y_noise = y.copy()
    indices = np.random.permutation(len(y))  
    num_noise = int(noise_percent * len(y))
    
    for idx in indices[:num_noise]:
        y_noise[idx] = np.random.randint(k, dtype=np.int32)
    return y_noise

def asymmetric_noise(y, noise_percent, k=10):
    """
    Apply asymmetric noise (class-dependent noise) to labels
    Replace labels according to predefined class transition rules (e.g., 5->7, 2->6)
    
    Args:
        y: Original label array, shape (n_samples,)
        noise_percent: Noise ratio in [0.0, 1.0] 
        k: Total number of classes (default: 10)
    
    Returns:
        y_noise: Noisy label array with asymmetric noise
    """
    y_noise = y.copy()
    # Class transition rules (adjustable for different datasets)
    transition_rules = {
        5: 7,
        2: 6,
        4: 2,
        7: 5,
        6: 2,
        0: 3 if k != 2 else 1,  # Special handling for IDS2018
        3: 4
    }
    
    for true_class, noisy_class in transition_rules.items():
        class_indices = np.where(y == true_class)[0]
        num_noise = int(noise_percent * len(class_indices))

        selected = np.random.choice(class_indices, num_noise, replace=False)
        y_noise[selected] = noisy_class
    
    return y_noise

def extract_and_cache_features(loader, classifier, cache_path, featdims, n_clusters, batch_size, force_run=False):

    if force_run or not os.path.exists(cache_path):
        feat_log = np.zeros((len(loader.dataset), sum(featdims)))
        score_log = np.zeros((len(loader.dataset), n_clusters))
        label_log = np.zeros(len(loader.dataset))
        
        classifier.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                start = batch_idx * batch_size
                end = min(start + batch_size, len(loader.dataset))

                scores, features = classifier.feature_list(inputs)
                
                pooled_features = [
                    F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
                    for feat in features
                ]
                concatenated = torch.cat(pooled_features, dim=1)

                feat_log[start:end] = concatenated.cpu().numpy()
                label_log[start:end] = targets.cpu().numpy()
                score_log[start:end] = scores.cpu().numpy()

                if batch_idx % 20 == 0:
                    print(f'progress: [{batch_idx:03d}/{len(loader):03d}]', 
                          f'Current batch: {inputs.shape[0]} sample', 
                          f'Cumulative processing: {end} sample')

        cache_data = {
            'feat_log': feat_log,
            'score_log': score_log,
            'label_log': label_log
        }
        np.save(cache_path, cache_data)
        print(f'Features have been cached to: {cache_path}')
    else:
        cache_data = np.load(cache_path, allow_pickle=True).item()
        feat_log = cache_data['feat_log']
        score_log = cache_data['score_log']
        label_log = cache_data['label_log']
    
    return feat_log, score_log, label_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS-Net Training")
    parser.add_argument('--dataset', default='malicious_TLS', 
                      choices=['malicious_TLS', 'IDS_2017', 'IDS_2018_friday'],
                      help='Known Dataset')
    parser.add_argument('--dataset_channel', default=1, type=int, choices=[1,3],
                      help='channel')
    parser.add_argument('--classifier_train_epochs', default=50, type=int,
                      help='Full epochs')
    parser.add_argument('--classifier_start_epoch', default=20, type=int,
                      help='Pretraining epochs')
    parser.add_argument('--classifier_train_batch_size', default=128, type=int,
                      help='batch_size')
    parser.add_argument('--classifier_model_path', 
                      default=r'AEGIS-Net/cache/trained_model/classifier_%s_r%s_%s.model',
                      help='model save path')
    parser.add_argument('--num_classes', default=24, type=int,
                      help='Number of classes')
    parser.add_argument('--noise_percent', default=0.1, type=float,
                      help='[0.0, 1.0]')
    parser.add_argument('--y_pseudo_generate', 
                      type=lambda x: x.lower() in ['true', '1', 'yes'], 
                      default=False,
                      help='Whether to use asymmetric noise generation methods')
    args = parser.parse_args()


    print(f"noise_percent: {args.noise_percent}")
    print(f"classifier_train_epochs: {args.classifier_train_epochs}")
    print(f"Noise type: {'Asymmetric' if args.y_pseudo_generate else 'Symmetric'}")

    # Load Dataset
    x_train, y_train, x_test, y_test = load_data(args.dataset)
    print(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")

    # Setting parameters according to the dataset
    dataset_config = {
        'malicious_TLS': {'classes': 23, 'batch_size': 128},
        'IDS_2017': {'classes': 15, 'batch_size': 1024},
        'IDS_2018_friday': {'classes': 2, 'batch_size': 128}
    }
    if args.dataset in dataset_config:
        args.num_classes = dataset_config[args.dataset]['classes']
        args.classifier_train_batch_size = dataset_config[args.dataset]['batch_size']
    else:  
        args.num_classes = 10

    # Generating Noise Labels
    if args.y_pseudo_generate:
        y_pseudo = asymmetric_noise(y_train, args.noise_percent, args.num_classes)
    else:
        y_pseudo = symmetric_noise(y_train, args.noise_percent, args.num_classes)
    
    # Noise distribution
    print("Noise Label Distribution:", Counter(y_pseudo))
    clean_ratio = np.mean(y_pseudo == y_train)
    print(f'Original label accuracy: {clean_ratio:.2%}')
    
    classifier = DeepResNet(input_size=x_train.shape[0], num_classes=args.num_classes)
    classifier.load_model(args.classifier_model_path %(args.dataset, str(args.noise_percent), str(args.y_pseudo_generate)))
    # print(classifier)
    classifier.cuda()

    begin = time.time()
    # Feature Dimension Analysis
    dummy_input = torch.zeros((100, 1, 79)).cuda()
    _, feature_list = classifier.feature_list(dummy_input)
    featdims = [feat.shape[1] for feat in feature_list]  
    
    # Data preprocessing pipeline
    def create_dataloader(data, labels, batch_size):
        tensor_x = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  
        tensor_y = torch.tensor(labels, dtype=torch.long)
        dataset = Data.TensorDataset(tensor_x, tensor_y)
        return Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_loader = create_dataloader(x_train, y_train, args.classifier_train_batch_size)
    test_loader = create_dataloader(x_test, y_test, args.classifier_train_batch_size)

    cache_base = r"AEGIS-Net/cache"
    FORCE_REGEN = False  
    
    # Training set/testing set feature processing
    for split, loader in [('train', train_loader), ('val', test_loader)]:
        cache_path = os.path.join(cache_base, f"{args.dataset}_{split}_features.npy")
        print(f'\n{" Begin  ":^20} {split.upper()} {" dataset ":^20}')
        
        features, scores, labels = extract_and_cache_features(
            loader=loader,
            classifier=classifier,
            cache_path=cache_path,
            featdims=featdims,
            n_clusters=args.num_classes,
            batch_size=args.classifier_train_batch_size,
            force_run=FORCE_REGEN
        )
        
        print(f'features.shape: {features.shape}')
        print(f'scores.shape: {scores.shape}')
        print(f'labels.shape: {labels.shape}')

    # OOD dataset processing
    OOD_NAME = 'Darknet2020'
    print(f'\n{" Start processing the OOD dataset ":*^60}')
    x_ood, y_ood, _, _ = load_data(OOD_NAME)
    ood_loader = create_dataloader(x_ood, y_ood, args.classifier_train_batch_size)
    
    ood_cache_path = os.path.join(cache_base, f"{OOD_NAME}_features.npy")
    ood_features, ood_scores, _ = extract_and_cache_features(
        loader=ood_loader,
        classifier=classifier,
        cache_path=ood_cache_path,
        featdims=featdims,
        n_clusters=args.num_classes,
        batch_size=args.classifier_train_batch_size,
        force_run=FORCE_REGEN
    )
    
    # 结果验证
    print(f'\n{" Feature processing complete ":=^60}')
    print(f'ood_features.shape: {ood_features.shape}')
    print(f'ood_scores.shape: {ood_scores.shape}')
    print(f'total time-consuming: {time.time()-begin:.2f}秒')