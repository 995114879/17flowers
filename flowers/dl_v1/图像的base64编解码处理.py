import base64
import io

from PIL import Image


def encode(img_path):
    with open(img_path, "rb") as reader:  # rb代表二进制加载
        img_content = reader.read()  # 加载图像的所有二进制数据
        img_base64_content = base64.b64encode(img_content)
        print(img_base64_content)
    return img_base64_content

def decode(img_base64_content):
    img_content = base64.b64decode(img_base64_content)
    img = Image.open(io.BytesIO(img_content))
    img.show()

s = encode("/mnt/code/shenlan/code/cv_code/17flowers/datas/17flowers/c1/image_0001.jpg")
decode(s)
