from PIL import Image
from torchvision import transforms

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])


def extract_feature_from_img_path(img_path):
    img: Image.Image = Image.open(img_path)
    img = img.convert('RGB')
    return trans(img)
