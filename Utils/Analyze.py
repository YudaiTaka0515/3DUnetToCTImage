import matplotlib.pyplot as plt
from Utils.Consts import *
from Utils.Metric import *
from keras import backend as K
import numpy as np
import tensorflow as tf


def analyze_dice_by_slice(predicted, ground_true, save_file):
    print(predicted.shape)
    print(ground_true.shape)

    lymph_size_by_slice = []
    predicted_size_by_slice = []
    dice_score_by_slice = []
    for i in range(ground_true.shape[0]):
        lymph_size = np.count_nonzero(ground_true[i]==1)
        lymph_size_by_slice.append(lymph_size)
        predicted_size = np.count_nonzero(predicted[i]==1)
        predicted_size_by_slice.append(predicted_size)
        dice_score = dice_coefficient(predicted[i], ground_true[i]).numpy()
        dice_score_by_slice.append(dice_score)

    print(lymph_size_by_slice)

    X = [i for i in range(len(dice_score_by_slice))]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(X, lymph_size_by_slice, 'C0', label="Ground True")

    ax2 = ax1.twinx()
    ln2 = ax2.plot(X, predicted_size_by_slice, 'C1', label="Predicted")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right')

    ax1.set_xlabel('slice')
    ax1.set_ylabel("pixel")
    ax1.grid(True)
    ax2.set_ylabel("dice score")

    plt.savefig(save_file)
    plt.show()











