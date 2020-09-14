import matplotlib.pyplot as plt
import os
from Utils.Consts import *


def PlotHistory(history, save_dir):
    """
    学習履歴を表示・保存する
    :param history:  学習履歴
    :param save_dir: 保存先のディレクトリ
    :return:
    なし
    """
    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'], "o-", label="loss", )
    plt.plot(history.history['val_loss'], "o-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(save_dir, "loss.png"))

    # plt.show()
    # 損失の履歴をプロット
    # plt.show()

    print("次のディレクトリに学習履歴が保存されました : ", save_dir)

