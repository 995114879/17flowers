import os
import sys


sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib


from . import extract_feature_from_img_path


def run(
        img_path_dir="../../datas/17flowers",
        algo_output_path="./output/flowers/ml/model.pkl"
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
    x = [extract_feature_from_img_path(img_path) for img_path in img_paths]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

    # 3. 模型对象的创建
    algo = LogisticRegression()

    # 4. 模型训练
    algo.fit(x_train, y_train)

    # 5. 模型评估
    y_pred = algo.predict(x)
    acc = metrics.accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.3f}")
    report = metrics.classification_report(y, y_pred)
    print(report)
    print("=" * 100)
    y_pred = algo.predict(x_train)
    acc = metrics.accuracy_score(y_train, y_pred)
    print(f"Train dataset Accuracy: {acc:.3f}")
    report = metrics.classification_report(y_train, y_pred)
    print(report)
    print("=" * 100)
    y_pred = algo.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Test dataset Accuracy: {acc:.3f}")
    report = metrics.classification_report(y_test, y_pred)
    print(report)

    # 6. 模型持久化
    os.makedirs(os.path.dirname(algo_output_path), exist_ok=True)
    joblib.dump({
        'algo': algo,
        'label_2_idx': label_2_idx
    }, algo_output_path)
