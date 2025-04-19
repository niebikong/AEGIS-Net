from Network import DeepResNet
import argparse
from util.datasets import load_data
import numpy as np
from collections import Counter
import torch
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")  # 忽略sklearn的警告信息

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

    classifier = DeepResNet(input_size=x_train.shape[0], 
                          num_classes=args.num_classes)

    # Model training
    classifier.fit(
        args=args,
        x=x_train,
        y=y_pseudo,
        y_true=y_train,
        num_epochs=args.classifier_train_epochs,
        start_epoch=args.classifier_start_epoch,
        batch_size=args.classifier_train_batch_size
    )
    
    # Model save
    model_path = args.classifier_model_path % (
        args.dataset, 
        str(args.noise_percent),
        str(args.y_pseudo_generate)
    )
    classifier.save_model(model_path)
    print(f"The model has been saved to: {model_path}")

    # Evaluation of the cleaning effect of the training set
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_pred_train, _, _ = classifier.predict(x_train_tensor)
    train_acc = np.mean(y_pred_train == y_train)
    print(f'Accuracy after training set cleaning: {train_acc:.2%}')

    # Evaluations
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_pred, _, _ = classifier.predict(x_test_tensor)
    
    # Calculation of evaluation indicators
    test_acc = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f'\n{" Result ":=^60}')
    print(f'test_acc: {test_acc:.2%}')
    print(f'F1: {f1:.4f}')
    print(f'precision: {precision:.4f}')
    print(f'recall: {recall:.4f}')
    print('\n classification_report:')
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_str = np.array2string(
        cm, 
        separator='\t', 
        formatter={'int': lambda x: f"{x:4d}"},
        prefix='           '
    )

    # save result
    result_path = 'AEGIS-Net/cache/output.txt'
    with open(result_path, 'a', encoding="utf-8") as f:
        f.write(f"\n{' Experimental records ':=^60}\n")
        f.write(f"noise_percent: {args.noise_percent}\n")
        f.write(f"Accuracy after training set cleaning: {train_acc:.2%}\n")
        f.write(f"test_acc: {test_acc:.2%}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"recall: {recall:.4f}\n")
        f.write("\nconfusion matrix:\n")
        f.write(cm_str)
        f.write("\n\n classification_report:\n")
        f.write(classification_report(y_test, y_pred))
    print(f"\n The results have been saved to: {result_path}")