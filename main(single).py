import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# 设置随机种子
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_text_file(file_path):
    """通用的文本文件读取函数"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'ascii']
    
    # 首先以二进制模式读取文件内容
    with open(file_path, 'rb') as f:
        content = f.read()
    
    # 尝试不同的编码
    for encoding in encodings:
        try:
            text = content.decode(encoding).strip()
            return text
        except:
            continue
    
    # 如果所有编码都失败，使用二进制方式强制解码
    try:
        text = content.decode('utf-8', errors='ignore').strip()
    except:
        text = "无文本"
    
    return text if text else "无文本"

# 定义数据集类
class MultiModalDataset(Dataset):
    def __init__(self, data_dir, guid_label_file, tokenizer, transform, is_test=False):
        self.data_dir = data_dir
        self.df = pd.read_csv(guid_label_file)
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test
        
        # 标签映射
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        
        # 检查数据目录
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录 {data_dir} 不存在")
        
        # 获取所有可用的文件
        self.available_files = set()
        for root, _, files in os.walk(data_dir):
            for file in files:
                name_without_ext = os.path.splitext(file)[0]
                self.available_files.add(name_without_ext)
        
        # 过滤数据集，只保留文件存在的样本
        valid_guids = []
        for guid in self.df['guid']:
            guid_str = str(guid).replace('.0', '')
            if guid_str in self.available_files:
                valid_guids.append(guid)
        
        self.df = self.df[self.df['guid'].isin(valid_guids)].reset_index(drop=True)
        print(f"总样本数: {len(self.df)}, 有效样本数: {len(valid_guids)}")
        
        if len(valid_guids) == 0:
            raise RuntimeError("没有找到有效的样本！请检查数据目录和文件名是否匹配。")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        guid = str(self.df.iloc[idx]['guid']).replace('.0', '')
        
        # 构建文件路径
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        
        # 确保文件存在
        if not os.path.exists(text_path) or not os.path.exists(image_path):
            raise FileNotFoundError(
                f"文件不存在:\n"
                f"文本文件: {text_path} ({'存在' if os.path.exists(text_path) else '不存在'})\n"
                f"图像文件: {image_path} ({'存在' if os.path.exists(image_path) else '不存在'})"
            )
        
        # 加载文本
        text = read_text_file(text_path)
        
        # 处理文本
        encoded = self.tokenizer(text, padding='max_length', truncation=True, 
                               max_length=128, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        if not self.is_test:
            label = self.label_map[self.df.iloc[idx]['tag']]
            return input_ids, attention_mask, image, label, guid
        return input_ids, attention_mask, image, -1, guid

# 定义多模态融合模型
class MultiModalModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiModalModel, self).__init__()
        
        # BERT文本编码器
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.text_projection = nn.Linear(768, 512)
        
        # ResNet图像编码器
        self.resnet = models.resnet50(pretrained=True)
        # 移除原始的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.image_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512)
        )
        
        # 多模态融合
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, images):
        # 文本特征
        text_features = self.bert(input_ids, attention_mask=attention_mask)[1]
        text_features = self.text_projection(text_features)
        
        # 图像特征
        image_features = self.resnet(images)
        image_features = self.image_projection(image_features)
        
        # 特征融合
        combined_features = torch.cat([text_features, image_features], dim=1)
        output = self.fusion(combined_features)
        return output

class TextOnlyModel(nn.Module):
    def __init__(self, num_classes=3):
        super(TextOnlyModel, self).__init__()
        # BERT文本编码器
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, images=None):
        text_features = self.bert(input_ids, attention_mask=attention_mask)[1]
        return self.classifier(text_features)

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ImageOnlyModel, self).__init__()
        # ResNet图像编码器
        self.resnet = models.resnet50(pretrained=True)
        # 移除原始的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(2048, 512),  # ResNet50的特征维度是2048
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids=None, attention_mask=None, images=None):
        features = self.resnet(images)
        return self.classifier(features)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_acc = 0
    # 记录训练过程
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids, attention_mask, images, labels, _ = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, images, labels, _ = batch
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(input_ids, attention_mask, images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算平均loss和准确率
        epoch_train_loss = train_loss/len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss/len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        # 记录到history中
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': epoch_val_acc,
                'history': history
            }, 'best_model.pth')
    
    # 保存完整的训练历史
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    return history

def run_ablation_study():
    # 设置内存优化选项
    torch.cuda.empty_cache()  # 清空GPU缓存
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # 减小图像尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 检查数据目录
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"警告: 当前目录下找不到 {data_dir} 文件夹")
        data_dir = os.path.join('..', 'data')
        if not os.path.exists(data_dir):
            raise FileNotFoundError("找不到数据目录！请确保数据文件夹位置正确。")
    
    print(f"\n使用数据目录: {os.path.abspath(data_dir)}")
    
    # 加载数据集
    train_df = pd.read_csv('train.txt')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    val_df.to_csv('val.txt', index=False)
    
    # 创建数据加载器，进一步减小batch_size
    batch_size = 4  # 减小批次大小
    train_dataset = MultiModalDataset(data_dir, 'train.txt', tokenizer, transform)
    val_dataset = MultiModalDataset(data_dir, 'val.txt', tokenizer, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 实验配置
    experiments = {
        '多模态融合': lambda: MultiModalModel().to(device),
        '仅文本': lambda: TextOnlyModel().to(device),
        '仅图像': lambda: ImageOnlyModel().to(device)
    }
    
    results = {}
    
    for model_name, model_fn in experiments.items():
        print(f"\n开始训练 {model_name} 模型...")
        
        # 每次实验前清空GPU缓存
        torch.cuda.empty_cache()
        
        # 动态创建模型
        model = model_fn()
        
        # 使用混合精度训练
        scaler = torch.cuda.amp.GradScaler()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        
        # 修改train_model函数调用，添加梯度累积
        def train_with_amp(model, train_loader, val_loader, criterion, optimizer, num_epochs, accumulation_steps=2):
            best_val_acc = 0
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            for epoch in range(num_epochs):
                # 训练阶段
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                optimizer.zero_grad()
                
                for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
                    input_ids, attention_mask, images, labels, _ = batch
                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                    images, labels = images.to(device), labels.to(device)
                    
                    # 使用混合精度训练
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask, images)
                        loss = criterion(outputs, labels) / accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    train_loss += loss.item() * accumulation_steps
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                    
                    # 清理不需要的缓存
                    del input_ids, attention_mask, images, labels, outputs
                    torch.cuda.empty_cache()
                
                # 验证阶段
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, images, labels, _ = batch
                        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                        images, labels = images.to(device), labels.to(device)
                        
                        with torch.cuda.amp.autocast():
                            outputs = model(input_ids, attention_mask, images)
                            loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        
                        # 清理不需要的缓存
                        del input_ids, attention_mask, images, labels, outputs
                        torch.cuda.empty_cache()
                
                # 计算平均loss和准确率
                epoch_train_loss = train_loss/len(train_loader)
                epoch_train_acc = 100. * train_correct / train_total
                epoch_val_loss = val_loss/len(val_loader)
                epoch_val_acc = 100. * val_correct / val_total
                
                # 记录到history中
                history['train_loss'].append(epoch_train_loss)
                history['train_acc'].append(epoch_train_acc)
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)
                
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
                print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
                
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': epoch_train_loss,
                        'val_loss': epoch_val_loss,
                        'train_acc': epoch_train_acc,
                        'val_acc': epoch_val_acc,
                        'history': history
                    }, f'best_model_{model_name}.pth')
            
            return history
        
        history = train_with_amp(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
        results[model_name] = history
        
        # 加载最佳模型并评估
        checkpoint = torch.load(f'best_model_{model_name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint['val_acc']
        print(f"{model_name} 最佳验证集准确率: {best_val_acc:.2f}%")
        
        # 清理GPU内存
        del model, optimizer, criterion, scaler
        torch.cuda.empty_cache()
    
    # 绘制对比图
    plt.figure(figsize=(15, 5))
    
    # 损失曲线对比
    plt.subplot(1, 2, 1)
    for model_name, history in results.items():
        plt.plot(history['val_loss'], label=f'{model_name}')
    plt.title('验证损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线对比
    plt.subplot(1, 2, 2)
    for model_name, history in results.items():
        plt.plot(history['val_acc'], label=f'{model_name}')
    plt.title('验证准确率对比')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ablation_study.png')
    plt.close()
    
    # 保存实验结果
    ablation_results = {
        model_name: {
            'best_val_acc': torch.load(f'best_model_{model_name}.pth')['val_acc']
        }
        for model_name in experiments.keys()
    }
    
    with open('ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=4)
    
    return ablation_results

if __name__ == '__main__':
    # 运行消融实验
    ablation_results = run_ablation_study()
    
    # 打印结果总结
    print("\n消融实验结果总结:")
    print("-" * 40)
    for model_name, results in ablation_results.items():
        print(f"{model_name}:")
        print(f"最佳验证集准确率: {results['best_val_acc']:.2f}%")
    print("-" * 40)
