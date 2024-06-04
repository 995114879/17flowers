from skimage import io as ski_io
from skimage import color as ski_color
from skimage import transform as ski_transform
from skimage import feature as ski_feature




def extract_feature_from_img_path(img_path):
    img = ski_io.imread(img_path)
    img = ski_color.rgb2gray(img)
    img = ski_transform.resize(img, (50, 50))
    return ski_feature.hog(img)