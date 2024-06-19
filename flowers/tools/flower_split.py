import os
import cv2 as cv


def t0():
    input_dir = os.path.abspath("../../17flowers")
    listdirs = os.listdir(input_dir)
    listdirs.sort()  # 确保文件按照编号顺序排序

    c = 1
    for idx, filename in enumerate(listdirs):
        abspath = os.path.join(input_dir, filename)

        # 构建输出目录和输出文件路径
        out_dir = os.path.abspath(f"../../datas/17flowers/c{c}")
        outpath = os.path.join(out_dir, filename)
        os.makedirs(out_dir, exist_ok=True)

        # 读取图像文件
        img = cv.imread(abspath)
        if img is None:
            print(f"无法读取图片: {abspath}")
            continue

        # 保存图像文件
        if not cv.imwrite(outpath, img):
            print(f"无法保存图片: {outpath}")

        # 按每80张图片划分到不同的文件夹
        if (idx + 1) % 80 == 0:
            print(f"处理到第 {idx + 1} 张图片，切换到新的文件夹 c{c + 1}")
            c += 1

if __name__ == '__main__':
    t0()
