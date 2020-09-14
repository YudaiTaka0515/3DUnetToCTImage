from Utils.Consts import *
import numpy as np


# 入出力はuint8ってことだけ言っとく
# 入力は一枚の画像
def CropImage(image, cropped_shape, patientID):
    """
    画像を指定した座標を中心にクロップ
    :param image:　クロップする画像(一枚のみ)
    :param cropped_shape: どのサイズにクロップするか(x, y の辞書型)
    :param patientID: 患者ID(中心点を読み込むのに使用)
    :return:
    """
    print("*"*20 + "In CropImage" + "*"*20)
    patient = "OPE-" + str(patientID)
    center = CENTER[patient]
    cropped_image = image[center[1] - cropped_shape["x"] // 2: center[1] + cropped_shape["x"] // 2,
                          center[0] - cropped_shape["y"] // 2: center[0] + cropped_shape["y"] // 2]
    return cropped_image


def CropImages(images, cropped_shape, patientID):
    """
    画像群を指定した座標を中心にクロップ
    :param images:　クロップする画像群(slices, row, col)
    :param cropped_shape: どのサイズにクロップするか(x, y の辞書型)
    :param patientID: 患者ID(中心点を読み込むのに使用)
    :return:
    """
    print("*"*20 + "In CropImages" + "*"*20)
    patient = "OPE-" + str(patientID)
    center = CENTER[patient]
    slices = images.shape[0]
    shape = (slices, cropped_shape["x"], cropped_shape["y"])

    cropped_images = np.zeros(shape, dtype=np.float)

    for i in range(slices):
        cropped_images[i, :, :] = images[i,
                                        center[1] - cropped_shape["x"] // 2: center[1] + cropped_shape["x"] // 2,
                                        center[0] - cropped_shape["y"] // 2: center[0] + cropped_shape["y"] // 2]

    return cropped_images
