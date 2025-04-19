import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import label_correction
import copy

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

class DeepResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepResNet, self).__init__()
        self.classes = num_classes

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3) 
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 1024, stride=2),
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)
        self.feature_dim = 1024

    def forward(self, x):
        # [B, 1, seq_len] -> [B, 64, seq_len]
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.res_blocks(x)
        
        x = self.global_avg_pool(x)
        # [B, 512]
        features = x.view(x.size(0), -1)
        
        logits = self.fc(features)
        
        return logits, features  # [128, 24], [128, 512]
    
    def feature_list(self, x):
        out_list = []
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.res_blocks[0](x)  # ResidualBlock(64, 128)
        out_list.append(x)  # 第一残差块输出

        x = self.res_blocks[1](x)  # ResidualBlock(128, 256, stride=2)
        out_list.append(x)  # 第二残差块输出

        x = self.res_blocks[2](x)  # ResidualBlock(256, 512, stride=2)
        out_list.append(x)  # 第三残差块输出

        x = self.res_blocks[3](x)  # ResidualBlock(512, 1024, stride=2)
        out_list.append(x)  # 第四残差块输出

        x = self.global_avg_pool(x)

        # 展平为 [B, 1024]
        x = x.view(x.size(0), -1)

        # 分类
        y = self.fc(x)
        # 1024 + 512 + 256 + 128 = 1920 
        return y, out_list  # 返回分类结果和每一层的特征图列表

    @staticmethod
    def info_nce_loss(features, labels, temperature):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        device = features.device

        mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        mask = mask.masked_fill(self_mask, 0)

        similarity_matrix = torch.matmul(features, features.T) / temperature
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        return -mean_log_prob_pos.mean()


    def fit(self, args, x, y, y_true=None, lr=0.001, num_epochs=50, batch_size=256,
            start_epoch=10, alpha=0.5, beta=0.8, temperature=0.07, patience=10):  # alpha=0.5, beta=0.8, temperature=0.07
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.long)
        indices = torch.arange(len(x))
        dataset = Data.TensorDataset(x, y, indices)
        loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.cuda()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss().cuda()
        
        best_acc = -np.inf
        early_stop_counter = 0
        best_model_weights = None
        
        for epoch in range(num_epochs):
            tag = ''
            if epoch >= start_epoch:
                tag = 'with label_correction'
                _, _, z = self.predict(x)
                y_pseudo = label_correction.produce_pseudo_labels(z, y.numpy(), self.classes, epoch)
                self.y_pseudo = torch.tensor(y_pseudo, dtype=torch.long)
            
            train_loss = 0.0
            train_cls_loss = 0.0
            train_contrast_loss = 0.0
            
            for step, (batch_x, batch_y, batch_indices) in enumerate(loader):
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                optimizer.zero_grad()
                outputs, features = self(batch_x)
                
                cls_loss = criterion(outputs, batch_y)
                contrast_loss = self.info_nce_loss(features, batch_y, temperature)
                
                if epoch >= start_epoch:
                    batch_y_pseudo = self.y_pseudo[batch_indices].cuda()
                    pseudo_loss = criterion(outputs, batch_y_pseudo)
                    loss = (1 - alpha)*cls_loss + alpha*pseudo_loss + beta*contrast_loss
                else:
                    loss = cls_loss + beta*contrast_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                train_cls_loss += cls_loss.item() * batch_x.size(0)
                train_contrast_loss += contrast_loss.item() * batch_x.size(0)
            
            avg_loss = train_loss / len(dataset)
            avg_cls = train_cls_loss / len(dataset)
            avg_con = train_contrast_loss / len(dataset)
            
            if (epoch+1) % 1 == 0:
                y_pred, _, _ = self.predict(x)
                if y_true is not None:
                    acc = np.sum(y_pred == y_true) / len(y_true)
                    print(f"# {tag} Epoch {epoch+1:3d}: Total Loss: {avg_loss:.6f} "
                        f"(Cls: {avg_cls:.6f}, Contrast: {avg_con:.6f}) Acc: {acc:.6f}")
                    
                    if (epoch+1) > args.classifier_start_epoch:
                        if acc > best_acc:
                            best_acc = acc
                            best_model_weights = copy.deepcopy(self.state_dict())
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= patience:
                                print(f"Early stopping after {epoch+1} epochs.")
                                self.load_state_dict(best_model_weights)
                                return
                else:
                    print(f"# {tag} Epoch {epoch+1:3d}: Total Loss: {avg_loss:.6f} "
                        f"(Cls: {avg_cls:.6f}, Contrast: {avg_con:.6f})")
        
        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)


    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        
    def predict(self, x):
        self.eval()
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  
        dataset = Data.TensorDataset(x)
        loader = Data.DataLoader(dataset, batch_size=5000, shuffle=False)
        
        outputs, features = [], []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].cuda()
                batch_x = batch_x.squeeze(1)  
                logits, feats = self(batch_x)
                outputs.append(logits.cpu())
                features.append(feats.cpu())
                
        outputs = torch.cat(outputs).numpy()
        features = torch.cat(features).numpy()
        y_pred = np.argmax(outputs, axis=1)
        y_prob = np.max(F.softmax(torch.tensor(outputs), dim=1).numpy(), axis=1)
        return y_pred, y_prob, features

if __name__ == "__main__":
    classifier = DeepResNet(input_size=117, num_classes=10)
    input_image = torch.randn(100, 1, 117)  
    
    output, feature_map = classifier(input_image)
    
    print("Output shape:", output.shape)
    print("Feature map shape:", feature_map.shape)
