import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K


def SetGPU(device):
    """
    GPUの設定を行う
    :return:
    """
    # 使用するGPUを設定
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # TensorFlowがGPUを認識しているか確認
    print("Device : ", device_lib.list_local_devices())


if __name__ == '__main__':
    SetGPU()

