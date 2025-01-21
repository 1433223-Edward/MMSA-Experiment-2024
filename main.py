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
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        guid = str(self.df.iloc[idx]['guid'])
        
        # 加载文本
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        text = read_text_file(text_path)
        
        # 处理文本
        encoded = self.tokenizer(text, padding='max_length', truncation=True, 
                               max_length=128, return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 加载图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
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
        self.resnet.fc = nn.Linear(2048, 512)
        
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
        
        # 特征融合
        combined_features = torch.cat([text_features, image_features], dim=1)
        output = self.fusion(combined_features)
        return output

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

def main():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 设置数据目录
    data_dir = 'data'  # 数据文件夹路径
    if not os.path.exists(data_dir):
        print(f"警告: {data_dir} 目录不存在，尝试在上级目录查找...")
        data_dir = '../data'  # 尝试上级目录
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"找不到数据目录 'data'，请确保数据文件夹位置正确")
    
    # 加载数据集
    train_df = pd.read_csv('train.txt')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # 保存验证集
    val_df.to_csv('val.txt', index=False)
    
    print(f"数据目录: {data_dir}")
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")
    
    # 创建数据加载器
    train_dataset = MultiModalDataset(data_dir, 'train.txt', tokenizer, transform)
    val_dataset = MultiModalDataset(data_dir, 'val.txt', tokenizer, transform)
    test_dataset = MultiModalDataset(data_dir, 'test_without_label.txt', tokenizer, transform, is_test=True)
    
    # 检查几个样本的文件是否存在
    #def check_file_exists(dataset, num_samples=5):
        #print(f"\n检查{num_samples}个样本文件:")
        #for i in range(min(num_samples, len(dataset))):
            #guid = str(dataset.df.iloc[i]['guid'])
            #text_path = os.path.join(dataset.data_dir, f"{guid}.txt")
            #image_path = os.path.join(dataset.data_dir, f"{guid}.jpg")
            #print(f"样本 {guid}:")
            #print(f"  文本文件: {'存在' if os.path.exists(text_path) else '不存在'} ({text_path})")
            #print(f"  图像文件: {'存在' if os.path.exists(image_path) else '不存在'} ({image_path})")
    
    #check_file_exists(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 初始化模型
    model = MultiModalModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # 训练模型
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    
    # 绘制训练过程图
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('模型准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    # 加载最佳模型
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n最佳模型性能 (Epoch {checkpoint['epoch']+1}):")
    print(f"验证集准确率: {checkpoint['val_acc']:.2f}%")
    
    # 预测测试集
    model.eval()
    
    predictions = []
    guids = []
    label_map_reverse = {0: 'positive', 1: 'neutral', 2: 'negative'}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, images, _, batch_guids = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            images = images.to(device)
            
            outputs = model(input_ids, attention_mask, images)
            _, predicted = outputs.max(1)
            
            predictions.extend([label_map_reverse[p.item()] for p in predicted])
            guids.extend(batch_guids)
    
    # 保存预测结果
    results_df = pd.DataFrame({'guid': guids, 'tag': predictions})
    results_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()
