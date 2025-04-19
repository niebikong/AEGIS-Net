import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_malicious_TLS():
    output_path = r"AEGIS-Net/dataset/label_encodered_malicious_TLS-1.csv"
    data = pd.read_csv(output_path)
    label_encoder = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)  # 将所有数据转换为字符串类型
            data[col] = label_encoder.fit_transform(data[col])
    data = data.astype(float)
    X = data.iloc[:, :-1].values  # 特征矩阵为除最后一列外的所有列
    y = data.iloc[:, -1].values.astype(int)  # 最后一列作为标签
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test


def load_IDS_2017():
    """
    Label
    0     2272688
    1        1966
    2      128027
    3       10293
    4      230124
    5        5499
    6        5796
    7        7938
    8          11
    9          36
    10     158930
    11       5897
    12       1507
    13         21
    14        652
    Name: Label, dtype: int64
    """
    output_path = r'AEGIS-Net/dataset/CIC_IDS_2017_Formated_DATA.csv'
    data = pd.read_csv(output_path)
    label_encoder = LabelEncoder()

    # 转换所有非数值列为数字
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)  # 将所有数据转换为字符串类型
            data[col] = label_encoder.fit_transform(data[col])

    # 将数据转换为浮动类型
    data = data.astype(float)

    # 检查并处理无穷大值和NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)  # 替换无穷大为NaN
    data.fillna(0, inplace=True)  # 可以选择填充NaN为0，或者使用其他方法

    # 获取特征和标签
    X = data.iloc[:, :-1].values  # 特征矩阵为除最后一列外的所有列
    y = data.iloc[:, -1].values.astype(int)  # 最后一列作为标签

    # 对每类样本进行采样
    sampled_indices = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]  # 获取当前类别的所有样本索引
        if 15000 > len(class_indices) > 11000:  # TODO: 如果样本量大于15000，随机取10% 
            sampled_class_indices = np.random.choice(class_indices, size=int(len(class_indices) * 0.8), replace=False) # TODO: 0.1
        elif 300000 >= len(class_indices) >= 100000:
            sampled_class_indices = np.random.choice(class_indices, size=int(len(class_indices) * 0.08), replace=False) # TODO: 0.1
        elif len(class_indices) >= 1000000:
            sampled_class_indices = np.random.choice(class_indices, size=int(len(class_indices) * 0.008), replace=False) # TODO: 0.1
        else:  # 如果样本量小于等于15000，不进行采样
            sampled_class_indices = class_indices
        sampled_indices.extend(sampled_class_indices)

    X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sampled, test_size=0.3, random_state=42)

    return X_train, y_train, X_test, y_test


def load_IDS_2018_friday():
    output_path = r'AEGIS-Net/dataset/processed_friday_dataset.csv'
    data = pd.read_csv(output_path)
    label_encoder = LabelEncoder()

    # 转换所有非数值列为数字
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)  # 将所有数据转换为字符串类型
            data[col] = label_encoder.fit_transform(data[col])

    # 将数据转换为浮动类型
    data = data.astype(float)

    # 检查并处理无穷大值和NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)  # 替换无穷大为NaN
    data.fillna(0, inplace=True)  # 可以选择填充NaN为0，或者使用其他方法

    # 获取特征和标签
    X = data.iloc[:, :-1].values  # 特征矩阵为除最后一列外的所有列
    y = data.iloc[:, -1].values.astype(int)  # 最后一列作为标签

    # 对每类样本进行采样
    sampled_indices = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]  # 获取当前类别的所有样本索引
        if len(class_indices) > 15000:  # TODO: 如果样本量大于15000，随机取10% 
            sampled_class_indices = np.random.choice(class_indices, size=int(len(class_indices) * 0.1), replace=False) # TODO: 0.1
        else:  # 如果样本量小于等于15000，不进行采样
            sampled_class_indices = class_indices
        sampled_indices.extend(sampled_class_indices)

    
    # 根据采样后的索引提取数据
    X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sampled)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sampled, test_size=0.3, random_state=42)

    return X_train, y_train, X_test, y_test

def load_Darknet2020():
    output_path = r'/home/ju/Desktop/NetMamba/PNP/SMP/dataset/cleaned_data.csv'
    data = pd.read_csv(output_path)
    label_encoder = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)  # 将所有数据转换为字符串类型
            data[col] = label_encoder.fit_transform(data[col])
    data = data.astype(float)
    X = data.iloc[:, :-2].values  # 特征矩阵为除最后一列外的所有列
    y = data.iloc[:, -1].values.astype(int)  # 最后一列作为标签
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test


def load_data(dataset):
    if dataset == 'malicious_TLS':
        return load_malicious_TLS()
    
    elif dataset == 'IDS_2017':
        return load_IDS_2017()
    
    elif dataset == 'IDS_2018_friday':
        return load_IDS_2018_friday()
    
    elif dataset == 'Darknet2020':
        return load_Darknet2020()
    
    else:
        raise ValueError('Not defined for loading %s' % dataset)



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data('malicious_TLS')
    print(x_train, x_train.shape, y_train, len(y_train))
