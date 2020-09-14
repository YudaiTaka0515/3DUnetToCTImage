import glob
import numpy as np
from PIL import Image
from Utils.Consts import *
from pprint import pprint
from natsort import natsorted
import os
from Utils.PreprocessImage import *
import pydicom


def LoadDataset(image_dirs, mask_dirs, num_for_testing):
    """
    :param image_dirs: 学習に用いる画像(.png)が格納されているディレクトリ
    :param mask_dirs: 学習に用いるmask画像が格納されているディレクトリ
    :param num_for_testing : テスト用に使用するデータセットの数(患者単位)
    :return:
    学習画像とマスク画像をそれぞれ訓練用，検証用に分割して返す
    float型に変換済み
    """
    print("*"*20 + "in load_data_set" + "*"*20)
    images_dirs = natsorted(glob.glob(image_dirs + "/Patient*"))
    masks_paths = natsorted(glob.glob(mask_dirs + "/Patient*"))
    # ちゃんと読み込めているかチェック
    # pprint(images_dirs)
    # pprint(masks_paths)

    train_images = []
    train_masks = []
    validation_images = []
    validation_masks = []
    for i in range(len(images_dirs)):
        if i + 1 == 8:
            print("patient 8 is skipped")
            continue
        print(i)
        if i < num_for_testing:
            validation_images.append(LoadImageFromPng(images_dirs[i], i+1, should_crop=True))
            validation_masks.append(LoadImageFromPng(masks_paths[i], i+1, should_crop=True))
        else:
            train_images.append(LoadImageFromPng(images_dirs[i], i+1, should_crop=True))
            train_masks.append(LoadImageFromPng(masks_paths[i], i+1, should_crop=True))

    train_images = np.concatenate(train_images, axis=0)
    train_masks = np.concatenate(train_masks, axis=0)
    validation_images = np.concatenate(validation_images, axis=0)
    validation_masks = np.concatenate(validation_masks, axis=0)

    # train_masks = train_masks.astype(float)
    # validation_masks = validation_masks.astype(float)

    return train_images, train_masks, validation_images, validation_masks


def LoadImageFromPng(png_dir, patient_ID, should_crop=True):
    """
    :param png_dir: 学習用に使用する画像が格納されているディレクトリの親ディレクトリ
    :param patient_ID :
    :param should_crop :
    :return: float
    """
    png_paths = natsorted(glob.glob(png_dir+"/*.png"))
    slices = len(png_paths)

    if should_crop:
        shape = (slices, CROPPED_SHAPE["x"], CROPPED_SHAPE["y"])
    else:
        shape = (slices, ORIGINAL_SHAPE["x"], ORIGINAL_SHAPE["y"])

    images = np.zeros(shape, dtype=float)

    for i, file in enumerate(png_paths):
        image = Image.open(file)
        image = image.convert("RGB")
        data = np.asarray(image)/255.0
        images[i] = CropImage(data[:, :, 0], CROPPED_SHAPE, patient_ID)
        images[i] = np.where(images[i] > 0.0, 1.0, 0)

    print(images.shape)
    return images








