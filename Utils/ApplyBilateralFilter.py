import cv2
import numpy as np
import matplotlib.pyplot as plt


def ApplyBilateralFilter(images):
    """

    :param images: uint8型
    :return:
    """
    if images.dtype != np.uint8:
        print("data type is not correct")

    filtered_images = np.zeros(images.shape, dtype='uint8')
    for i in range(images.shape[0]):
        filtered_images[i] = cv2.bilateralFilter(images[i], 9, 20, 20)
        # plt.imshow(filtered_images[i])
        # plt.show()

    return filtered_images


if __name__ == '__main__':
    pass

