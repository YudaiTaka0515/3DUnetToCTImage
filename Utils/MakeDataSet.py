from Utils.LoadImage import *
import cv2
import matplotlib.pyplot as plt
from Utils.ApplyBilateralFilter import *


def ConvertDicom2Png(dicom_folder, save_parent_dir):
    print("*"*20 + "In ConvertDicom2Png", "*"*20)
    # dicomのフォルダを取得
    dicom_dir_list = natsorted(glob.glob(os.path.join(dicom_folder, "OPE*")))
    print(dicom_dir_list)
    # 患者IDを取得(Pngファイルの名前に使用)
    for dicom_dir in dicom_dir_list:
        patient_ID = str(int((dicom_dir.split('OPE-')[-1].split(".")[0])))
        save_dir = os.path.join(save_parent_dir, "Patient"+str(patient_ID))
        SaveDicomAsPng(dicom_dir, save_dir, preprocessed=True)

    print("Done")


def SaveDicomAsPng(dicom_dir, save_dir, preprocessed=False):
    """
    dicomファイルを読み込んでPngファイルに保存する
    :param preprocessed:
    :param dicom_dir: dicomデータが保存されているディレクトリ
    :param save_dir: dicomデータから変換されたPng画像を保存するディレクトリ
    :return:
    """
    print("*"*20 + "SaveDicomAsPng", "*"*20)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print("mkdir : ", save_dir)

    dicom_paths = natsorted(glob.glob(os.path.join(dicom_dir, "Image*")))
    images_loaded_from_dicom = LoadImageFromDicom(dicom_paths)
    if preprocessed:
        print("preprocess")
        images_loaded_from_dicom = ApplyBilateralFilter(images_loaded_from_dicom)

    print("shape : ", images_loaded_from_dicom.shape)

    slices = images_loaded_from_dicom.shape[0]
    for i in range(slices):
        file_name = str(i+1) + ".png"
        pil_image = Image.fromarray(images_loaded_from_dicom[i])
        # print("saved : ", os.path.join(save_dir, file_name))
        pil_image.save(os.path.join(save_dir, file_name))


def LoadImageFromDicom(dicom_paths):
    print("*"*20 + "In LoadImageFromDicom", "*"*20)

    slice_count = len(dicom_paths)
    # print(dicom_paths)
    # print(slice_count)
    # dicomデータを格納するNumpy配列を確保
    dicom_images = np.empty((slice_count, ORIGINAL_SHAPE["x"], ORIGINAL_SHAPE["y"]), dtype='uint8')
    for i, dicom_path in enumerate(dicom_paths):
        loaded_dicom = pydicom.dcmread(dicom_path)
        # WindowCenter, WindowWidthの取得
        window_center = loaded_dicom.WindowCenter
        window_rescale_intercept = loaded_dicom.RescaleIntercept
        window_center = window_center - window_rescale_intercept
        window_width = loaded_dicom.WindowWidth
        # Numpy配列に変換
        loaded_dicom.convert_pixel_data()
        dicom_image = loaded_dicom.pixel_array
        # 表示画素値の最大と最小を計算
        max_val = window_center + window_width / 2
        min_val = window_center - window_width / 2
        # Window処理
        dicom_image = 255 * (dicom_image - min_val) / (max_val - min_val)  # 最大と最小画素値を0から255に変換
        dicom_image[dicom_image > 255] = 255  # 255より大きい画素値は255に変換
        dicom_image[dicom_image < 0] = 0  # JPEG画像として保存
        dicom_image = dicom_image.astype('uint8')
        dicom_images[i] = dicom_image.copy()
        # print("in load_dicom : shape = ", dicom_images.shape)
    return dicom_images


def SaveMaskAsPng(mask_dir, save_parent_dir):
    print("*" * 20 + "In SaveMaskAsPng", "*" * 20)
    if not os.path.exists(save_parent_dir):
        os.mkdir(save_parent_dir)
        print("mkdir : ", save_parent_dir)

    mask_paths = natsorted(glob.glob(os.path.join(mask_dir, "*raw")))
    for mask_path in mask_paths:
        masks_loaded_from_raw = LoadMaskFromRaw(mask_path)
        print("shape : ", masks_loaded_from_raw.shape)
        patient_id = str(int((mask_path.split('OPE')[-1].split(".")[0])))
        print(patient_id)
        save_dir = os.path.join(save_parent_dir, "Patient" + str(patient_id))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print("mkdir : ", save_dir)
        slices = masks_loaded_from_raw.shape[0]
        for i in range(slices):
            file_name = str(i + 1) + ".png"
            pil_image = Image.fromarray(masks_loaded_from_raw[i])
            # print("saved : ", os.path.join(save_dir, file_name))
            pil_image.save(os.path.join(save_dir, file_name))


# return : (スライス、縦、横）
def LoadMaskFromRaw(raw_file_path):
    print("*"*20 + "In LoadMaskFromRaw", "*"*20)
    file = open(raw_file_path, 'rb')
    img_arr = np.fromfile(file, np.uint8)
    slices = img_arr.shape[0] // (ORIGINAL_SHAPE["x"] * ORIGINAL_SHAPE["y"])
    img_arr = img_arr.reshape(ORIGINAL_SHAPE["x"], ORIGINAL_SHAPE["y"], slices)
    masks = np.array(img_arr)*255
    masks = np.reshape(masks, (slices, ORIGINAL_SHAPE["x"], ORIGINAL_SHAPE["y"]))
    print("unique : ", np.unique(masks))
    return masks


def FillMaskFromRaw(raw_file_path, save_path):
    masks = LoadMaskFromRaw(raw_file_path)
    z, h, w = masks.shape
    filled_masks = np.zeros(shape=(z, h, w), dtype='uint8')

    temp_arr = np.zeros(shape=(h+2, w+2), dtype='uint8')

    for i in range(masks.shape[0]):
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masks[i])
        for j in range(1, n_labels):
            point = (round(centroids[j][0]), round(centroids[j][1]))
            print(point)
            cv2.floodFill(image=labels, mask=temp_arr, seedPoint=point, newVal=1)

        if n_labels > 1:
            print(i)
            plt.imshow(labels)
            plt.show()

        filled_masks[i] = np.where(labels > 0, 1, 0)

    filled_masks = filled_masks.reshape(masks.shape[0]*masks.shape[1]*masks.shape[2])
    filled_masks.tofile(save_path)


def main_dicom():
    dicom_folder_ = r"I:\Data\OPE(1-24)"
    save_parent_dir_ = r"I:\Data\OPE_IMAGES(PNG,bilateral)"
    if not os.path.exists(save_parent_dir_):
        os.mkdir(save_parent_dir_)

    ConvertDicom2Png(dicom_folder_, save_parent_dir_)


def main_raw():
    data_folder_ = r"I:\Data\ForSegmentation"
    save_parent_dir_ = r"I:\Data\OPE_MASKS(PNG)"
    if not os.path.exists(save_parent_dir_):
        os.mkdir(save_parent_dir_)

    SaveMaskAsPng(data_folder_, save_parent_dir_)


def mainFillMasks():
    save_dir = r"I:\Data\Filled"
    raw_dir = r"I:\Data\OPE(1-24)"
    patient_id = "OPE-30"
    base_name = os.path.join(patient_id, patient_id + ".raw")
    raw_file_path = os.path.join(raw_dir, base_name)
    save_path = os.path.join(save_dir, patient_id+".raw")
    FillMaskFromRaw(raw_file_path, save_path)


if __name__ == '__main__':
    main_dicom()



