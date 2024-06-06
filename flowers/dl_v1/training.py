import sys
import os

import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, datasets

from ..commons.metrics import Accuracy

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from sklearn.model_selection import train_test_split

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
    dataset = datasets.ImageFolder(
        root=img_path_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128))
        ])
    )


    dataloader = DataLoader(
        dataset=dataset,  # 给定数据集对象
        batch_size=batch_size,  # 构建成的批次数量
        shuffle=True,  # 是否打乱数据顺序，True表示打乱，False表示不打乱
        num_workers=0,  # 加载数据时候的线程数目，0表示在当前主线程上执行加载逻辑
        collate_fn=None,  # 聚合函数，决定如何将多条数据合并成一个批次数据对象，一般情况下直接默认即可（cv用默认即可）
        # prefetch_factor=2  # 和num_workers一起生效，当num_workers>0的时候，表示预加载的样本数量；当num_workers=0的时候，该值必须为默认值
    )


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
    for epoch in range(total_epoch):
        # 4.a 训练阶段
        net.train()
        for _x, _y in dataloader:
            # 前向过程
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
        for _x,_y in dataloader:
            # 前向过程
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
            'total_epoch': total_epoch,
            'label_2_idx': dataset.class_to_idx
        },
        model_output_path
    )
    print(dataset.class_to_idx)
