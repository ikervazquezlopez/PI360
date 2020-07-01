import sys
import os
import cv2
from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_results(mse):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.array(list(range(0, mse.shape[1])))
    y = np.array(list(range(0, mse.shape[0])))
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, np.flip(y), mse)
    ax.set_xlabel("Interpolation length")
    ax.set_xticks(np.array(list(range(0,mse.shape[1]))))

    ax.set_ylabel("Image row")
    ax.set_zlabel("MSE")
    plt.savefig("MSE_impact_500.png")


if __name__ == '__main__':

    in_dir = sys.argv[1]

    onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    f = open(join(in_dir, onlyfilespaths[0]), 'rb')
    mse = np.load(f)
    f.close()

    for filename in tqdm(onlyfilespaths[1:]):
        f = open(join(in_dir, filename), 'rb')
        new_mse= np.load(f)
        mse = mse + new_mse
        f.close()


    mse = mse / len(onlyfilespaths)

    plot_results(mse)
