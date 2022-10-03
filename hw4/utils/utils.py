import numpy as np
from PIL import Image

def save_img(data, idx):
    images = data[b"data"]
    img = np.transpose(images.reshape(-1, 3, 32, 32), (0, 2, 3, 1))[idx]
    img = Image.fromarray(img)
    img.save("img.png")