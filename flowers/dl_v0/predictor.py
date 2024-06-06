"""
预测器
"""
import os.path
import torch

from . import extract_feature_from_img_path
from .training import FlowerNetworkV0


class Predictor:
    def __init__(self, algo_path):
        super(Predictor, self).__init__()
        algo_result = torch.load(algo_path, map_location='cpu')
        net = FlowerNetworkV0()
        if 'net_param' in algo_result:
            state_dict = algo_result['net_param']
        else:
            state_dict = algo_result['net'].state_dict()
        missing_keys, unexpected_keys = net.load_state_dict(state_dict=state_dict, strict=False)
        if len(missing_keys) > 0:
            raise ValueError(f"部分参数未初始化，不能进行推理：{missing_keys}")
        self.net = net.eval()
        self.label2idx = algo_result['label_2_idx']
        self.idx2label = {}
        for label, idx in self.label2idx.items():
            self.idx2label[idx] = label

    @torch.no_grad()
    def predict(self, img_path: str, k=1):
        """
        基于给定路径进行预测，产生返回结果
        :param img_path:
        :return: 返回一个字典对象
        :k: 获取前topk
        """
        if not os.path.exists(img_path):
            return {'code': 1, 'msg': f'给定图像路径不存在:{img_path}'}
        # 1. 图像数据加载并转换
        x = extract_feature_from_img_path(img_path)  # tensor [C,H,W]

        # 2. 模型预测
        y_score = self.net(x[None])[0]  # [C,H,W] --> [1,C,H,W] --> [1,17] --> [17]
        y_proba = torch.softmax(y_score, dim=0).numpy()
        y_proba_idx = list(zip(y_proba, range(len(y_proba))))
        y_proba_idx = sorted(y_proba_idx, reverse=True, key=lambda t: t[0])
        print(y_proba_idx)

        # 3. 获取预测类别值和概率值
        k = min(max(k, 1), len(y_proba_idx))
        result = {
            'code': 0,
            'topk': k,
            'datas': []
        }
        for proba, label_idx in y_proba_idx:
            r = {
                'label': self.idx2label[label_idx], 'label_idx': label_idx, 'proba': f"{proba:.2f}"
            }
            result['datas'].append(r)
            k -= 1
            if k <= 0:
                break

        return result
