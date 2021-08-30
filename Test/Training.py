from Model.isensee2017_3DUnet import *
from Utils.LoadImage import *
from Utils.Consts import *
from Utils.SetGPU import *
from Utils.PlotHistory import *
import glob
from Utils.Callback import *
from sklearn.model_selection import train_test_split
import cv2




def main():
    SetGPU(4)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        print("make directory : ", SAVE_DIR)

    # -----------------データの読み込み--------------------
    # TODO
    # load_dataset()の確認
    train_images, train_masks, validation_images, validation_masks = LoadDataset(IMAGE_DIR, MASK_DIR, TEST_NUM)
    # 入力が正しいかチェックする
    print("-" * 20 + "check shape" + "-" * 20)
    print("train | (images, masks) = ({}, {})".format(train_images.shape, train_masks.shape))
    print("validation | (images, masks) = ({}, {})".format(validation_images.shape, validation_masks.shape))

    # データを学習用と検証用に分割
    depth = CROPPED_SHAPE["z"]
    train_shape = (train_images.shape[0] // depth, depth, train_images.shape[1], train_images.shape[2])
    validation_shape = (validation_images.shape[0] // depth, depth, validation_images.shape[1],
                        validation_images.shape[2])
    X_train = np.zeros(train_shape, dtype=float)
    Y_train = np.zeros(train_shape, dtype=float)
    X_validation = np.zeros(validation_shape, dtype=float)
    Y_validation = np.zeros(validation_shape, dtype=float)
    print(train_shape)
    print(train_images.shape)
    print(X_train.shape)
    for i in range(X_train.shape[0]):
        X_train[i] = train_images[i * depth:(i + 1) * depth, :, :]
        Y_train[i] = train_masks[i * depth:(i + 1) * depth, :, :]

    for i in range(X_validation.shape[0]):
        X_validation[i] = validation_images[i * depth:(i + 1) * depth, :, :]
        Y_validation[i] = validation_masks[i * depth:(i + 1) * depth, :, :]

    X_train = X_train.transpose((0, 2, 3, 1))
    Y_train = Y_train.transpose((0, 2, 3, 1))
    X_validation = X_validation.transpose((0, 2, 3, 1))
    Y_validation = Y_validation.transpose((0, 2, 3, 1))

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
    Y_train = Y_train.reshape(Y_train.shape[0], 1, Y_train.shape[1], Y_train.shape[2], Y_train.shape[3])
    X_validation = X_validation.reshape((X_validation.shape[0], 1, X_validation.shape[1], X_validation.shape[2], depth))
    Y_validation = Y_validation.reshape((Y_validation.shape[0], 1, Y_validation.shape[1], Y_validation.shape[2], depth))

    model = isensee2017_model(input_shape=(1, 128, 128, depth))
    model_path = os.path.join(SAVE_DIR, "model.h5")

    print(model.summary())
    print(X_train.shape)

    history = model.fit(x=X_train, y=Y_train,
                        batch_size=N_BATCH, epochs=N_EPOCH,
                        validation_data=(X_validation, Y_validation),
                        verbose=2,
                        callbacks=get_callbacks(model_file=model_path))
    # history = model.fit(Temp_X, Temp_Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=0.2)
    model.save_weights(os.path.join(SAVE_DIR, "unet_weights.hdf5"))

    PlotHistory(history, SAVE_DIR)


main()
