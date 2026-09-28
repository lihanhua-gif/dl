[README.md](https://github.com/user-attachments/files/32755004/README.md)
# 深度学习入门练习（deep_learning）

一个用于深度学习入门的 Python 练习项目，用最直观的方式演示机器学习与神经网络的核心概念：**从手工实现感知机，到用 PyTorch 训练一个识别手写数字的神经网络**。

## 项目结构

```
deep_learning/
├── learning1.py        # 感知机（Perceptron）——用身高/体重判断"偏胖/偏瘦"
├── learning2.py        # PyTorch 三层线性网络——学习"输入 1 输出 0"的映射
├── simple_nn.py        # SimpleMLP 全连接网络——MNIST 手写数字识别（10 分类）
├── simple_nn_副本.txt  # simple_nn.py 的文本副本
├── data/               # MNIST 数据集（脚本首次运行时会自动下载到此处）
└── README.md
```

## 脚本说明

### 1. learning1.py —— 感知机（Perceptron）

从零实现的二分类感知机，目标是根据**身高和体重**预测一个人属于"偏胖"还是"偏瘦"。

- 对 10 条身高/体重样本做 **Min-Max 归一化**（压缩到 0~1 区间）
- 手动实现感知机训练（权重更新 + 偏置更新），支持设置学习率和迭代次数
- 训练结束后用训练好的模型预测新样本，打印真实标签与预测结果对比

核心思路：`加权和 z = w1·x1 + w2·x2 + b`，当 `z >= 0` 判为类别 1，否则为类别 0。

### 2. learning2.py —— PyTorch 三层线性网络

用 PyTorch 的 `nn.Sequential` 搭一个「线性 → ReLU → 线性 → ReLU → 线性」的小网络，学习将输入 `1.0` 映射到目标 `0.0`。

- 手动初始化各层权重（2.0 / 3.0 / -1.0）
- 使用 `MSELoss`（均方误差）作为损失函数
- 使用 `SGD`（随机梯度下降）优化器，训练 100 轮，每 10 轮打印一次预测值与损失

适合理解：前向传播、反向传播、损失收敛过程。

### 3. simple_nn.py —— MNIST 手写数字识别（SimpleMLP）

用自定义类（继承 `nn.Module`）实现的全连接网络，识别 MNIST 手写数字（0~9）。

- **网络结构**：`784 → 128 → 10`（784 个像素 → 128 个隐藏单元 → 10 个分类），隐藏层用 ReLU 激活
- **数据**：MNIST 训练集 60000 张、测试集 10000 张，自动下载到 `./data`
- **预处理**：转张量（像素 0~255 → 0~1）+ 标准化（均值为 0.1307、标准差 0.3081）
- **训练配置**：`batch_size=128`、`shuffle=True`、SGD 优化器（lr=0.1）、交叉熵损失、训练 5 轮
- **结果可视化**：随机抽取 9 张测试图片，展示模型预测结果与置信度（预测正确显示绿色、错误显示红色）

> 注：`simple_nn.py` 中保留了一段被注释掉的「手工矩阵乘法」训练代码（直接操作 `w1/b1/w2/b2` 张量），与上面用 `nn.Module` 封装的方式对比，可以直观理解两种写法。

## 环境依赖

- Python 3.x
- PyTorch（含 `torchvision`）
- NumPy
- Matplotlib

安装依赖：

```bash
pip install torch torchvision numpy matplotlib
```

> 首次运行 `simple_nn.py` 会自动下载 MNIST 数据集到 `./data` 目录。

## 运行方式

在项目根目录分别运行：

```bash
python learning1.py      # 感知机：偏胖/偏瘦分类
python learning2.py      # 三层线性网络：1 → 0 映射
python simple_nn.py      # SimpleMLP：MNIST 手写数字识别（会弹出可视化窗口）
```

## 学习路线

1. **learning1.py** —— 理解最基础的线性分类：感知机、归一化、权重更新
2. **learning2.py** —— 进入 PyTorch：用现成模块搭网络、训练、观察损失下降
3. **simple_nn.py** —— 完整的小项目：数据加载、模型封装、批量训练、结果可视化

三份脚本由浅入深，覆盖了从「手工实现算法」到「使用深度学习框架」的完整入门路径。
