import numpy as np
from PIL import Image


def open_image(path):
    image = Image.open(path)
    return image


def resize_image(image, new_x, new_y):
    return image.resize((new_x, new_y))


def to_YCrCb(image_rgb):
    R = np.array(image_rgb)[:, :, 0]
    G = np.array(image_rgb)[:, :, 1]
    B = np.array(image_rgb)[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128
    return Y, Cr, Cb


def to_RGB(Y, Cr, Cb):
    R = Y + (Cr - 128) / 0.713
    B = Y + (Cb - 128) / 0.564
    G = (Y - 0.299*R - 0.114*B)/0.587

    rgb_image = np.stack((R, G, B), axis=-1)
    # S'assurer que les valeurs sont bien dans la plage [0, 255]
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    return rgb_image


def preprocess_image(path, new_x=None, new_y=None):
    image = open_image(path)
    if new_x is not None:
        image = resize_image(image, new_x, new_y)
    Y, Cr, Cb = to_YCrCb(image)
    return Y.reshape(np.array(image).shape[:2]).astype(np.float64) / 255, Cr, Cb
