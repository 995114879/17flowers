from flowers.ml import predictor
import json
import os

def run(algo_path='../../runner/output/flowers/ml/model.pkl'):
    k = 1
    try:
        p = predictor.Predictor(algo_path)
    except Exception as e:
        print(f"无法加载预测模型: {e}")
        return

    while True:
        value = input("\n\n===============================\n"
                      "1. 输入Q表示退出命令行\n"
                      "2. 输入K表示重新设定topk的值\n"
                      "3. 输入其他值表示为待预测的图像路径\n"
                      "请输入:")
        if value in ['Q', 'q']:
            break
        elif value in ['K', 'k']:
            value = input("请输入有效的topK的值:")
            try:
                k = int(value)
                if k <= 0:
                    raise ValueError("topK的值必须是正整数")
                print(f"当前重新设定k为:{k}")
            except ValueError as ve:
                print(f"输入的值无效: {ve}")
        else:
            if not os.path.isfile(value):
                print(f"文件路径无效: {value}")
                continue

            try:
                r = p.predict(img_path=value, k=k)
                print(f"预测结果为:\n{json.dumps(r, ensure_ascii=False, indent=2)}")
            except Exception as e:
                print(f"预测过程中发生错误: {e}")


