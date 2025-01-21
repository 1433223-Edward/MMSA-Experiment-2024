# 多模态情感分析模型

本项目实现了一个基于BERT和ResNet的多模态情感分析模型，用于处理文本-图像对的情感分类任务（positive/neutral/negative）。

## 项目结构

├── data/ # 数据文件夹
│ ├── .jpg # 图像文件
│ └── .txt # 文本文件
├── main.py # 主要模型实现和训练代码
├── main(single).py # 单模态对比实验代码
├── ablation_study.py # 消融实验代码
├── requirements.txt # 环境依赖
├── train.txt # 训练集标签文件
├── val.txt # 验证集标签文件（由训练集划分）
├── test_without_label.txt # 测试集文件
├── training_curves.png # 训练过程可视化
├── ablation_study.png # 消融实验结果可视化
└── predictions.csv # 测试集预测结果

## 环境配置

1. 创建并激活虚拟环境（推荐）：

python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

2. 安装依赖：

pip install -r requirements.txt


## 运行说明

1. 训练多模态模型：

python main.py


2. 进行多种超参数对比实验：

python ablation_study.py


3. 单模态对比实验：

python main\(single\).py


## 模型架构

- 文本编码：使用中文BERT（bert-base-chinese）
- 图像编码：使用预训练ResNet50
- 多模态融合：将文本和图像特征投影到相同维度后拼接，通过MLP进行分类

## 实验结果

- 多模态模型验证集准确率：98%
- 仅文本模型验证集准确率：94%
- 仅图像模型验证集准确率：97%

## 技术栈

- PyTorch：深度学习框架
- Transformers (Hugging Face)：BERT模型实现
- torchvision：ResNet模型和图像处理
- pandas：数据处理
- matplotlib：可视化
- scikit-learn：数据集划分
- tqdm：进度显示

## 参考

1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
2. Deep Residual Learning for Image Recognition
3. Hugging Face Transformers
4. PyTorch Documentation

## 注意事项

1. 请确保data文件夹中包含所有必要的图像和文本文件
2. 文本文件可能使用不同的编码，代码中已处理多种编码格式
3. 建议使用GPU进行训练，以获得更好的性能
4. 模型会自动保存验证集性能最好的checkpoint
5. 因为数据过于庞大，数据集不在此上传，后面可以自行下载。
