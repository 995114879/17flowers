import os

import torch
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.optim as optim
import numpy as np


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    @torch.no_grad()
    def forward(self, y_score, y_true):
        """
        计算预测的准确率
        :param y_score: [N,C]
        :param y_true:  [N]
        :return:
        """
        # 获取预测的标签值
        pred_indices = torch.argmax(y_score, dim=1)  # [N,C] --> [N]
        # 两者进行比较
        pred_indices = pred_indices.to(y_true.device, dtype=y_true.dtype)
        acc = torch.mean((pred_indices == y_true).to(dtype=torch.float))
        return acc


class IrisNetwork(nn.Module):
    def __init__(self, in_features=4, num_classes=3):
        super(IrisNetwork, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_features, out_features=128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classify(x)


def training_v0(
        total_epoch=100, batch_size=8, model_output_path="./output/ml_dl/iris/mv0.pkl"
):
    # 1. 数据加载
    X, Y = load_iris(return_X_y=True)
    X = X.astype('float32')
    Y = Y.astype('int64')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=28)

    # 2. 特征转换

    # 3. 模型构建(模型构建 + 损失求解的对象 + 优化器对象 + 模型参数恢复等等)
    net = IrisNetwork(4, 3)
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy()
    opt = optim.SGD(params=net.parameters(), lr=0.01)
    if os.path.exists(model_output_path):
        print(f"模型参数恢复：{model_output_path}")
        m = torch.load(model_output_path, map_location='cpu')
        if 'net_param' in m:
            state_dict = m['net_param']
        else:
            state_dict = m['net'].state_dict()
        # state_dict: 给定参数字典mapping对象，key为参数的name，value为参数值
        # strict: 是否不允许存在没法恢复的参数或者多余的参数, 默认为True， True就表示不允许，也就是此时给定的参数mapping和模型的参数mapping必须完全一致/能够名称匹配的
        # print(state_dict.keys())
        # del state_dict['classify.6.bias']
        print(next(net.parameters()).view(-1)[0])
        state_dict['w'] = torch.tensor([1])
        missing_keys, unexpected_keys = net.load_state_dict(state_dict=state_dict, strict=False)
        print(f"未进行参数迁移初始化的key列表：{missing_keys}")
        print(f"多余的参数key列表：{unexpected_keys}")
        print(next(net.parameters()).view(-1)[0])

    # 4. 训练
    total_batch = len(x_train) // batch_size
    test_total_batch = len(x_test) // batch_size
    for epoch in range(total_epoch):
        # 4.a 训练阶段
        net.train()
        rnd_indcies = np.random.permutation(len(x_train))  # 产生[0,n)的n个数，并打乱顺序
        for batch in range(total_batch):
            # 前向过程
            _indices = rnd_indcies[batch * batch_size:(batch + 1) * batch_size]
            _x = torch.from_numpy(x_train[_indices])
            _y = torch.from_numpy(y_train[_indices])

            pred_score = net(_x)
            loss = loss_fn(pred_score, _y)
            acc = acc_fn(pred_score, _y)

            # 反向过程
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"epoch:{epoch}/{total_epoch} train loss:{loss.item():.3f} train accuracy:{acc.item():.3f}")

        # 4.b 评估阶段
        net.eval()
        rnd_indcies = np.random.permutation(len(x_test))  # 产生[0,n)的n个数，并打乱顺序
        for batch in range(test_total_batch):
            # 前向过程
            _indices = rnd_indcies[batch * batch_size:(batch + 1) * batch_size]
            _x = torch.from_numpy(x_test[_indices])
            _y = torch.from_numpy(y_test[_indices])

            pred_score = net(_x)
            loss = loss_fn(pred_score, _y)
            acc = acc_fn(pred_score, _y)

            print(f"epoch:{epoch}/{total_epoch} test loss:{loss.item():.3f} test accuracy:{acc.item():.3f}")

    # 6. 持久化
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    # torch.save API底层调用python模块库pickle进行二进制格式的持久化
    # NOTE: 在使用pickle进行二进制保存的时候，如果保存的是对象，那么会将这个对象的package信息也保存，在恢复的时候，如果无法加载对应的package，那么直接报错
    torch.save(
        {
            # 'net': net,  # 保存模型对象
            'net_param': net.state_dict(),
            'total_epoch': total_epoch
        },
        model_output_path
    )


def training_v1(
        total_epoch=100, batch_size=8, model_output_path="./output/ml_dl/wine/mv1.pkl"
):
    # 1. 数据加载
    X, Y = load_wine(return_X_y=True)
    X = X.astype('float32')
    Y = Y.astype('int64')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=28)

    # 2. 特征转换
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    # 3. 模型构建(模型构建 + 损失求解的对象 + 优化器对象 + 模型参数恢复等等)
    net = IrisNetwork(13, 3)
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy()
    opt = optim.SGD(params=net.parameters(), lr=0.01)
    if os.path.exists(model_output_path):
        print(f"模型参数恢复：{model_output_path}")
        m = torch.load(model_output_path, map_location='cpu')
        if 'net_param' in m:
            state_dict = m['net_param']
        else:
            state_dict = m['net'].state_dict()
        # state_dict: 给定参数字典mapping对象，key为参数的name，value为参数值
        # strict: 是否不允许存在没法恢复的参数或者多余的参数, 默认为True， True就表示不允许，也就是此时给定的参数mapping和模型的参数mapping必须完全一致/能够名称匹配的
        # print(state_dict.keys())
        # del state_dict['classify.6.bias']
        print(next(net.parameters()).view(-1)[0])
        state_dict['w'] = torch.tensor([1])
        missing_keys, unexpected_keys = net.load_state_dict(state_dict=state_dict, strict=False)
        print(f"未进行参数迁移初始化的key列表：{missing_keys}")
        print(f"多余的参数key列表：{unexpected_keys}")
        print(next(net.parameters()).view(-1)[0])

    # 4. 训练
    total_batch = len(x_train) // batch_size
    test_total_batch = len(x_test) // batch_size
    for epoch in range(total_epoch):
        # 4.a 训练阶段
        net.train()
        rnd_indcies = np.random.permutation(len(x_train))  # 产生[0,n)的n个数，并打乱顺序
        for batch in range(total_batch):
            # 前向过程
            _indices = rnd_indcies[batch * batch_size:(batch + 1) * batch_size]
            _x = torch.from_numpy(x_train[_indices])
            _y = torch.from_numpy(y_train[_indices])

            pred_score = net(_x)
            loss = loss_fn(pred_score, _y)
            acc = acc_fn(pred_score, _y)

            # 反向过程
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"epoch:{epoch}/{total_epoch} train loss:{loss.item():.3f} train accuracy:{acc.item():.3f}")

        # 4.b 评估阶段
        net.eval()
        rnd_indcies = np.random.permutation(len(x_test))  # 产生[0,n)的n个数，并打乱顺序
        for batch in range(test_total_batch):
            # 前向过程
            _indices = rnd_indcies[batch * batch_size:(batch + 1) * batch_size]
            _x = torch.from_numpy(x_test[_indices])
            _y = torch.from_numpy(y_test[_indices])

            pred_score = net(_x)
            loss = loss_fn(pred_score, _y)
            acc = acc_fn(pred_score, _y)

            print(f"epoch:{epoch}/{total_epoch} test loss:{loss.item():.3f} test accuracy:{acc.item():.3f}")

    # 6. 持久化
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    # torch.save API底层调用python模块库pickle进行二进制格式的持久化
    # NOTE: 在使用pickle进行二进制保存的时候，如果保存的是对象，那么会将这个对象的package信息也保存，在恢复的时候，如果无法加载对应的package，那么直接报错
    torch.save(
        {
            # 'net': net,  # 保存模型对象
            'net_param': net.state_dict(),
            'total_epoch': total_epoch
        },
        model_output_path
    )
