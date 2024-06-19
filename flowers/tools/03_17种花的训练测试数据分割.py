# -*- coding: utf-8 -*-

import os
import shutil
from random import shuffle

in_path = r"../../datas/17flowers"
out_path = r"../../17flowers_traintest"
train_out_path = fr"{out_path}/train"
test_out_path = fr"{out_path}/test"
train_size = 0.8

class_names = os.listdir(in_path)
for cls_name in class_names:
    cls_in_path = fr"{in_path}/{cls_name}"
    train_cls_out_path = fr"{train_out_path}/{cls_name}"
    test_cls_out_path = fr"{test_out_path}/{cls_name}"
    os.makedirs(train_cls_out_path, exist_ok=True)
    os.makedirs(test_cls_out_path, exist_ok=True)

    file_names = os.listdir(cls_in_path)
    shuffle(file_names)  # 随机打乱顺序
    train_numbers = int(len(file_names) * train_size)
    # 数据输出
    for file_name in file_names[:train_numbers]:
        src_file_path = rf"{cls_in_path}/{file_name}"
        target_file_path = rf"{train_cls_out_path}/{file_name}"
        shutil.copy(src_file_path, target_file_path)
    for file_name in file_names[train_numbers:]:
        src_file_path = rf"{cls_in_path}/{file_name}"
        target_file_path = rf"{test_cls_out_path}/{file_name}"
        shutil.copy(src_file_path, target_file_path)
