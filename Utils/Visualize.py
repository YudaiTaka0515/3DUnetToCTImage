import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2


def OverlayMasksOnImages(f_images, f_masks):
    if not (f_images.dtype == 'float' and f_masks.dtype == 'float'):
        print("input float numpy array")
        exit()
    if np.all(f_masks == 0) | np.all(f_masks == 1.0):
        print("[0, 1]以外が含まれています")
        exit()

    overlaid_images = np.zeros(f_masks.shape)
    for i in range(overlaid_images.shape[0]):
        overlaid_images[i] = cv2.addWeighted(f_images[i], 0.3, f_masks[i], 0.7, 0)

    return overlaid_images


def SaveImagesAsGif(f_images, save_path):
    if not (f_images.dtype == 'float'):
        print("input float numpy array")
        exit()

    ui_images = (f_images * 255).astype('uint8')

    images_for_save = np.stack([ui_images, ui_images, ui_images], axis=-1)

    # PIL型に変換して，リストに格納
    images_list = []
    for i in range(images_for_save.shape[0]):
        images_list.append(Image.fromarray(images_for_save).convert('P'))

    # GIFを保存
    images_list[0].save(save_path,
                        save_all=True,
                        append_images=blend_images[1:],
                        optimize=False,
                        duration=100,
                        loop=5)

    print("save gif : ", save_path)

