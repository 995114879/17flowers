import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from flowers import extract_feature_from_img_path

def run(
        img_path_dir="../../datas/17flowers",
        algo_output_path="./output/flowers/ml/model.pkl"
):
    # 1. 数据加载
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




