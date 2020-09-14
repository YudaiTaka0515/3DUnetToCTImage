# リファクタ確認済み(2020/09/03)
# 直すべきところ : margin, load_mask(predictとtrainingで挙動が異なる), make_gif
# TODO
# CT画像およびマスク画像はPngに変更
# 


from Utils.LoadImage import *
from natsort import natsorted
from keras.models import load_model
from Utils.SetGPU import *
from Utils.Visualize import *
from Utils.Analyze import *
import math
import os
import cv2

ID = 25
P = "OPE-" + str(ID)

# データセット(test用)の格納場所
TEST_DIR = r"/home/takahashi/LymphSegmentation/DataSet/"

DIR_OUTPUT_IS_STORED = r"/home/takahashi/PycharmProjects/Segmentation3D/History/he_normarized_filtered/"
# 学習済みのモデルの格納場所
PRE_TRAINED_MODEL = DIR_OUTPUT_IS_STORED + r"model.h5"
PRE_TRAINED_WEIGHTS = DIR_OUTPUT_IS_STORED + r"weights.hdf5"

# 予測の結果を出力する場所
DIR_SAVED_PRED = DIR_OUTPUT_IS_STORED + r"patient" + str(ID)
DIR_SAVED_GIF = DIR_OUTPUT_IS_STORED + "gif"
if not os.path.exists(DIR_SAVED_PRED):
    os.mkdir(DIR_SAVED_PRED)
    print("made directory : ", DIR_SAVED_PRED)

if not os.path.exists(DIR_SAVED_GIF):
    os.mkdir(DIR_SAVED_GIF)
    print("made directory : ", DIR_SAVED_GIF)


# TODO
# あまりよくない
def Predict(patient_id):
    # 使用するGPUを指定（サーバー限定？）
    # SetGPU(2)
    slices = 0

    # テスト画像, マスク画像の読み込み
    images_dir = os.path.join(TEST_DIR, "patient"+str(patient_id))
    masks_dir = os.path.join(TEST_DIR, "patient"+str(patient_id))
    test_images = LoadImageFromPng(images_dir, patient_id, should_crop=True)
    test_masks = LoadImageFromPng(masks_dir, patient_id, should_crop=True)

    # 正常に読み込まれているかチェック
    print("-"*20 + "loading check", "-"*20)
    print("images : shape={}, data_type={}".
          format(test_images.shape, test_images.dtype))
    print("masks  :shape={}, data_type={}, unique={}"
          .format(test_masks.shape, test_masks.dtype, np.unique(test_masks)))

    # 画像のShapeをモデルに合わせる
    # input shape : (n_channels, x_size, y_size, z_size)
    n_block = math.ceil(slices / CROPPED_SHAPE["z"])
    print("block_num: ", n_block)
    input_shape = (n_block, 1, CROPPED_SHAPE["x"], CROPPED_SHAPE["y"], CROPPED_SHAPE["z"])
    test_data = np.zeros(input_shape)
    label_data = np.zeros(input_shape, dtype=np.float)
    for block_i in range(n_block):
        for z in range(CROPPED_SHAPE["z"]):
            if block_i * CROPPED_SHAPE["z"] + z < test_images.shape[0]:
                test_data[block_i, 0, :, :, z] = test_images[block_i * CROPPED_SHAPE["z"] + z]
                label_data[block_i, 0, :, :, z] = test_masks[block_i * CROPPED_SHAPE["z"] + z]
    # 正常なshapeになっているか確認する
    print("input shape|test={}, mask={}".format(test_data.shape, label_data.shape))

    # 学習済みのモデルと重みを読み込む
    pre_trained_model = load_model(PRE_TRAINED_MODEL,
                                   custom_objects={"dice_coefficient_loss": dice_coefficient_loss})
    pre_trained_model.load_weights(PRE_TRAINED_WEIGHTS)

    # モデルからセグメンテーション結果を出力
    pred_labels = pre_trained_model.Predict(test_data)
    loss = pre_trained_model.evaluate(test_data, label_data)
    print("loss={}".format(loss))

    # 閾値を用いて予測結果から(0, 1)のマスクを生成
    threshold = 0.5
    pred_labels = np.where(pred_labels < threshold, 0, 1.0)

    # 推論結果を(スライス，512, 512)に変換
    pred_masks = np.zeros(test_masks.shape)
    # 余白(cropした画像をもとのサイズにreshapeするのに使う）
    margin = (ORIGINAL_SHAPE["x"] - CROPPED_SHAPE["x"]) // 2
    # クロップの中心点(XとYが逆な気がする)
    X = CENTER[P][1]
    Y = CENTER[P][0]
    for block_i in range(n_block):
        for z in range(CROPPED_SHAPE["z"]):
            if block_i * CROPPED_SHAPE["z"] + z < test_images.shape[0]:
                pred_masks[block_i * CROPPED_SHAPE["z"] + z, X - margin: X + margin, Y - margin:Y + margin] \
                    = pred_labels[block_i, 0, :, :, z]
    print("prediction| shape={}, unique={}".format(pred_masks.shape, np.unique(pred_masks)))

    # 推論画像をpngで保存
    dir_saved_prediction = os.path.join(TEST_DIR, "patient" + str(patient_id))
    for i in range(pred_masks.shape[0]):
        file_name = str(i + 1) + ".png"
        ui_masks = (pred_masks[i]*255).astype('uint8')        # 0~255に変換
        temp = np.zeros(ui_masks.shape, dtype='uint8')
        mask_for_save = np.stack([ui_masks, temp, temp], axis=-1)
        pil_image = Image.fromarray(mask_for_save)
        pil_image.save(os.path.join(dir_saved_prediction, file_name))

    # 推論画像をCT画像にoverlayしたGIFを作成
    # overlay
    overlaid_pred = OverlayMasksOnImages(test_images, pred_masks)
    overlaid_test = OverlayMasksOnImages(test_images, test_masks)
    # GIF作成
    path_saved_pred_gif = os.path.join(DIR_SAVED_GIF, "prediction"+str(patient_id))
    SaveImagesAsGif(overlaid_pred, path_saved_pred_gif)
    path_saved_test_gif = os.path.join(DIR_SAVED_GIF, "test"+str(patient_id))
    SaveImagesAsGif(overlaid_test, path_saved_test_gif)


if __name__ == '__main__':
    Predict()

