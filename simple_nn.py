import torch                       #导入pytorch核心库
import torch.nn as nn              #神经网络包，Module，Linear，损失函数在这里
import torch.nn.functional as F    #函数式接口：relu，softmax，cross_entropy
import torch.optim as optim        #优化器：SGD，Adam，用来更新权重
import torchvision                 #视觉数据集，MNIST在这里
import torchvision.transforms as transforms      #图片预处理工具
from torch.utils.data import DataLoader          #批量加载数据的工具
import matplotlib.pyplot as plt                  #绘图，画图片，图表
import numpy as np                               #数值计算库

torch.manual_seed(42)                 #设置随机种子42；保证每次运行，随机数都一样

plt.rcParams["font.family"] = ['Arial Unicode MS'] #设置Matplotlib中文字体
plt.rcParams['axes.unicode_minus'] = False         #解决负号显示方框bug

#数据预处理
transform = transforms.Compose([                   #Compose：把多个预处理步骤打包，依次执行
    transforms.ToTensor(),                         #图片转张量：像素0-255 -》 0-1浮点数
    transforms.Normalize((0.1307,),(0.3081,)),     #标准化：（x-均值）/标准差；MNISET统计好的均值，标准差
])

train_set = torchvision.datasets.MNIST(
    root='./data',                         #训练用的数据集存到当前文件夹下的data目录
    train=True,                            #True=训练集  60000张
    download=True,                         #文件不存在就自动下载；存在直接读取
    transform=transform                    #使用上面定义好预处理
)                                          
test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,                           #False=测试集 10000张
    download=True,
    transform=transform
)

#Dataloader：把数据集分成小批次，支持打乱，循环读取
train_loader = DataLoader(train_set,batch_size=128,shuffle=True)
#batch——size=128：一次拿128张图片；shuffle=True：每轮epoch打乱训练集顺序
test_loader = DataLoader(test_set,batch_size=256,shuffle=False)
#测试集不需要打乱

print(f'训练集:{len(train_set)}张')
print(f'测试集:{len(test_set)}张')
print(f'每张图:{train_set[0][0].shape}(通道，高，宽)')
'''
torch.manual_seed(42)

w1 = (torch.randn(784,128) * (2.0 / 784) ** 0.5).requires_grad_()
b1 = torch.zeros(128,requires_grad=True)

w2 = (torch.randn(128,10) * (2.0 / 128) ** 0.5).requires_grad_()
b2 = torch.zeros(10,requires_grad=True)

params = [w1,b1,w2,b2]
lr = 0.1

print(f'网络共{sum(p.numel() for p in params):,}个参数')

train_losses = []
test_accs = []

for epoch in range(5):
    running_loss = 0.0
    n_samples = 0

    for xb, yb in train_loader:
        x = xb.view(-1,784)
        h = F.relu(x @ w1 + b1)          #隐藏层：内层线性变换，使用Relu函数激活
        logits = h @ w2 + b2             #输出层：只做线性变换，不加softmax

        loss = F.cross_entropy(logits, yb)

        loss.backward()

        with torch.no_grad():
           for p in params:
               p -= lr * p.grad
               p.grad.zero_()

        running_loss += loss.item() * yb.size(0)
        n_samples += yb.size(0)

    train_loss = running_loss / n_samples

    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            x = xb.view(-1, 784)
            logits = F.relu(x @ w1 + b1) @ w2 + b2
            correct += (logits.argmax(1)==yb).sum().item()

    test_acc = correct / len(test_set)

    train_losses.append(train_loss)
    test_accs.append(test_acc)
    print(f'epoch{epoch+1}/5 train_loss={train_loss:.4}. test_acc={test_acc:.4}')
'''
#定义神经网络类，继承nn.Module
class SimpleMLP(nn.Module):               
    def __init__(self, *args, **kwargs):   
        super().__init__(*args, **kwargs)    #调用父类nn.Module的构造
        self.fc1 = nn.Linear(784, 128)       #全连接层1:输入784，输出128；内部自w1，b1
        self.fc2 = nn.Linear(128, 10)        #全连接层2:输入128，输出10；对应数字0-9十个分类

    #前向传播，定义数据怎么走
    def forward(self, x):
        x = x.view(-1, 784)               #把图片（批量，1，28，28）展平 -> (批量，784)；-1自动推导batch大小
        x = F.relu(self.fc1(x))           #传播第一层链接后，用relu激活，去掉负数
        return self.fc2(x)                #第二层全连接输出logits（不用做softmax）
    
torch.manual_seed(42)                     #固定随机种子，网络初始化固定
model = SimpleMLP()                       #实例化模型对象；创建网络实例
#SGD随机梯度下降优化器；model.parameters（）自动拿到网络全部，w，b权重；lr学习率0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)
#交叉熵损失函数，分类任务用，内部自带softMax计算  
loss_fn = nn.CrossEntropyLoss()


train_losses_v2 = [] #保存每轮epoch训练loss
test_accs_v2 = []    #保存每轮epoch测试集准确率

for epoch in range(5):  #一共循环5轮；一轮epoch等于把全部训练集过一遍
    model.train()       #设置模型为训练模式；
    running_loss = 0.0  #累计损失
    n_samples = 0       #累计样本数量

    #循环遍历训练集每一个小批次 xb图片，yb标签
    for xb, yb in train_loader:
        optimizer.zero_grad()       #上一轮梯度清零
        logits = model(xb)          #前向传播；把每一批图片送入网络，得到输出logits
        loss = loss_fn(logits, yb)  #计算损失，对比预测和真实标签
        loss.backward()             #优化器按照梯度更新全部网络权重
        optimizer.step()            #优化器按照梯度更新全部网络权重w，b
        running_loss += loss.item() * yb.size(0)#loss.item取出数值；累计总损失
        n_samples += yb.size(0)                 #统计样本数

        model.eval()
        correct = 0
        
import random
random.seed(7)
indices = random.sample(range(len(test_set)), 9)

model.eval()
fig, axes = plt.subplots(3, 3, figsize=(7, 7))
with torch.no_grad():
    for ax, idx in zip(axes.flat, indices):
        img, label = test_set[idx]
        logits = model(img.unsqueeze(0))
        probs = F.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])

        ax.imshow(img.squeeze() * 0.3081 + 0.1307, cmap='gray')
        ok = (pred == label)
        ax.set_title(f'真值{label}  预测{pred} ({conf:.0%})',
                     fontsize=10,color=('green' if ok else 'red'))
        ax.axis('off')
plt.suptitle('测试集预测',fontsize=13,y=1.00)
plt.tight_layout()
plt.show()