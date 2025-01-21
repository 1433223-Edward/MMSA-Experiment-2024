import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import itertools
import json
from main import MultiModalDataset, MultiModalModel, train_model
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_experiment(params, data_dir, train_df_path, val_df_path):
    """运行单个实验"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建数据加载器
    train_dataset = MultiModalDataset(data_dir, train_df_path, tokenizer, transform)
    val_dataset = MultiModalDataset(data_dir, val_df_path, tokenizer, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    
    # 初始化模型
    model = MultiModalModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'])
    
    # 训练模型
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs=params['num_epochs'])
    
    # 返回最佳验证准确率
    best_val_acc = max(history['val_acc'])
    return best_val_acc, history

def ablation_study():
    # 定义要测试的超参数组合
    param_grid = {
        'batch_size': [8, 16],
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'num_epochs': [10]
    }
    
    # 生成所有可能的超参数组合
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in itertools.product(*param_grid.values())]
    
    # 存储实验结果
    results = []
    
    # 设置数据目录
    data_dir = 'data'
    if not os.path.exists(data_dir):
        data_dir = '../data'
    
    # 运行所有实验
    for params in param_combinations:
        print(f"\n运行实验，参数: {params}")
        best_acc, history = run_experiment(params, data_dir, 'train.txt', 'val.txt')
        
        result = {
            'params': params,
            'best_val_acc': best_acc,
            'history': history
        }
        results.append(result)
        
        # 保存中间结果
        with open('ablation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 绘制结果图表
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 按不同维度绘制结果
    for i, param_name in enumerate(['batch_size', 'learning_rate', 'num_epochs']):
        plt.subplot(1, 3, i+1)
        
        # 收集特定参数的结果
        param_values = sorted(set(r['params'][param_name] for r in results))
        accuracies = []
        
        for value in param_values:
            relevant_results = [r['best_val_acc'] for r in results 
                              if r['params'][param_name] == value]
            avg_acc = sum(relevant_results) / len(relevant_results)
            accuracies.append(avg_acc)
        
        plt.plot(param_values, accuracies, 'o-')
        plt.title(f'{param_name}对验证准确率的影响')
        plt.xlabel(param_name)
        plt.ylabel('最佳验证准确率 (%)')
        
    plt.tight_layout()
    plt.savefig('ablation_study_1.png')
    plt.close()
    
    # 找出最佳参数组合
    best_result = max(results, key=lambda x: x['best_val_acc'])
    print("\n最佳参数组合:")
    print(f"参数: {best_result['params']}")
    print(f"验证准确率: {best_result['best_val_acc']:.2f}%")

if __name__ == '__main__':
    ablation_study()
