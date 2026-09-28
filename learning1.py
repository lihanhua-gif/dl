import numpy as np
import matplotlib.pyplot as plt

#原始数据
raw_data = np.array([
    [175, 71],
    [160, 63],
    [172, 69],
    [162, 65],
    [170, 66],
    [164, 67],
    [168, 64],
    [166, 69],
    [165, 62],
    [168, 72],
])

labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

#计算每个特征的最大值和最小值
x_min = raw_data.min(axis = 0)
x_max = raw_data.max(axis = 0)

#Min-Max 归一化，把所有值压缩到0～1
X = (raw_data - x_min) / (x_max - x_min)

#打印归一化结果
for i in range(len(X)):
    print(f"样本{i+1}: 身高={X[i][0]:.3f},体重={X[i][1]:.3f},标签={labels[i]}")
print("=" * 60)
print("【原始数据】")
print("=" * 60)
print(f"{'编号':<6}{'身高(cm)':<10}{'体重(kg)':<10} {'标签'}")
print("-" * 60)
for i in range(len(raw_data)):
    label_str = "偏胖" if labels[i] == 1 else "偏瘦"
    print(f" {i+1:<6} {raw_data[i][0]:<10} {raw_data[i][1]:<10} {label_str}")


#定义感知机
class Perceptron():
    def __init__(self, learning_rate = 1.0, n_iterations = 100):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weight = None
        self.bias = None
        self.errors_per_epoch = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        #权重和偏置全部初始化为0
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_per_epoch = []

        for epoch in range(self.n_iterations):
            errors = 0
            for idx in range(n_samples):
                x_i = X[idx]
                y_true = y[idx]

                #1.计算加权和：z = w1 * x1 + w2 * x2 +b
                z = np.dot(self.weights, x_i) + self.bias

                #2.激活判断：z >= 0 输出1，否则输出0
                y_pred = 1 if z >= 0 else 0

                #3.计算误差
                error = y_true - y_pred

                #4.雨过预测错误，更新权重和偏置
                if error != 0:
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    errors += 1
                    print(f"第{epoch+1}轮,第{idx+1}个样本，更新权重:{self.weights},更新偏置:{self.bias}")

            self.errors_per_epoch.append(errors)

            #本轮0错误，收敛，训练结束
            if errors == 0:
                print(f"第{epoch+1}轮,错误数归零,训练收敛！")
                break

        return self

    def predict(self, X):
        #计算加权和: z = w1*x1 + w2*x2 + b
        z = np.dot(X, self.weights) + self.bias

        #激活判断：z >= 0输出1，否则输出0
        return np.where(z >= 0, 1, 0)

    
model = Perceptron(learning_rate = 0.5, n_iterations = 100)
model.fit(X, labels)

predictions = model.predict(X)

for i in range((len(X))):
    true_label = "偏胖" if labels[i] == 1 else "偏瘦"
    pred_label = "偏胖" if predictions[i] == 1 else"偏瘦"
    status = "✅" if predictions[i] == labels[i] else "❌"
    print(f"样本{i+1}: 真实={true_label}, 预测={pred_label}   {status}")

print(f"\n最终参数：w1={model.weights[0]:.4f},w2={model.weights[1]:.4f},b={model.bias:.4f}")

new_data = np.array([
    [168,40],
    [162,89],
    [178,70],
])
new_data_norm = (new_data - x_min) /(x_max - x_min)

new_pred = model.predict(new_data_norm)

for i in range((len(new_data))):
    result = "偏胖" if new_pred[i] == 1 else "偏瘦"
    print(f"身高={new_data[i][0]}cm,体重={new_data[i][1]}kg->{result}")
