import requests
import base64



def encode(img_path):
    with open(img_path, "rb") as reader:  # rb代表二进制加载
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)
    return img_base64_content

def t0():
    r = requests.get(
        url="http://127.0.0.1:9001/predict",
        params={
            'path': '/mnt/code/shenlan/code/cv_code/17flowers/datas/17flowers/c1/image_0001.jpg',
            'topk': 3
        }
    )
    if r.status_code == 200:
        print("请求服务器并成功返回")
        print(r.json())  # 将返回结果转换为json格式（python中就是字典对象）
        print(r.text)
    else:
        print(f"请求服务器网络异常：{r.status_code}")

def t1():
    r = requests.get(
        url="http://127.0.0.1:9001/predict",
        params={
            'url': 'http://viapi-test.oss-cn-shanghai.aliyuncs.com/viapi-3.0domepic/imagerecog/RecognizeImageColor/RecognizeImageColor1.jpg',
            'topk': 5
        }
    )
    if r.status_code == 200:
        print("请求服务器并成功返回")
        print(r.json())  # 将返回结果转换为json格式（python中就是字典对象）
        print(r.text)
    else:
        print(f"请求服务器网络异常：{r.status_code}")

def t2():
    r = requests.post(
        url="http://127.0.0.1:9001/predict",
        data={
            'image': encode('/mnt/code/shenlan/code/cv_code/17flowers/datas/17flowers/c1/image_0005.jpg'),
            'topk': 2
        }
    )
    if r.status_code == 200:
        print("请求服务器并成功返回")
        print(r.json())  # 将返回结果转换为json格式（python中就是字典对象）
        print(r.text)
    else:
        print(f"请求服务器网络异常：{r.status_code}")

if __name__ == '__main__':
    t1()
