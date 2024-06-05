"""
预测器
"""
import os.path
import joblib
from sklearn.linear_model import LogisticRegression


from . import extract_feature_from_img_path

class Predictor:
    def __init__(self, algo_path):
        super(Predictor, self).__init__()
        algo_result = joblib.load(algo_path)
        self.algo: LogisticRegression = algo_result['algo']
        self.label2idx = algo_result['label_2_idx']
        self.idx2label = {}
        for label, idx in self.label2idx.items():
            self.idx2label[idx] = label

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
        x = extract_feature_from_img_path(img_path)

        # 2. 模型预测
        y_idx = self.algo.predict([x])[0]
        y_proba = self.algo.predict_proba([x])[0]
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
