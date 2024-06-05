import os
import sys
import os

import torch
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.optim as optim
import numpy as np

from ..commons.metrics import Accuracy

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib

from . import extract_feature_from_img_path


class FlowerNetworkV0(nn.Module):
    def __init__(self):
        super(FlowerNetworkV0, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(7, 7))
        )
        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 17)
        )

    def forward(self, x):
        """
        :param x: [N,C,H,W]
        :return: [N,17]
        """
        z = self.features(x)
        z = torch.flatten(z, 1, -1)
        return self.classify(z)


def run(
        img_path_dir="../../datas/17flowers",
        total_epoch=100, batch_size=8, model_output_path="./output/flowers/dl_v0/model.pkl"
):
    # 1. 数据加载 --> 只加载图像的img_path以及对应的类别
    img_paths, img_labels, y = [], [], []
    label_2_idx = {}
    for img_label in os.listdir(img_path_dir):
        try:
            img_label_idx = label_2_idx[img_label]
        except KeyError:
            img_label_idx = len(label_2_idx)
            label_2_idx[img_label] = img_label_idx

        cur_label_img_path_dir = os.path.join(img_path_dir, img_label)
        img_names = os.listdir(cur_label_img_path_dir)
        for img_name in img_names:
            img_path = os.path.join(cur_label_img_path_dir, img_name)
            img_paths.append(img_path)
            img_labels.append(img_label)
            y.append(img_label_idx)
    print(label_2_idx)

    # 2. 特征工程
    with torch.no_grad():
        x = [extract_feature_from_img_path(img_path)[None].numpy() for img_path in img_paths]
        x = np.concatenate(x, axis=0)
        y = np.asarray(y, dtype=np.int64)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

    # 3. 模型构建(模型构建 + 损失求解的对象 + 优化器对象 + 模型参数恢复等等)
    net = FlowerNetworkV0()
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
