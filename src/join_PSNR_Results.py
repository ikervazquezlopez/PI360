import sys
import os
import cv2
from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epsilon = 0.000000001


def plot_results(psnr):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.array(list(range(0, psnr.shape[1])))
    y = np.array(list(range(0, psnr.shape[0])))
    x, y = np.meshgrid(x, y)

    ax.plot_surface(y, np.flip(x), psnr)
    ax.set_ylabel("Interpolation length")
    ax.set_yticklabels(list(range(psnr.shape[1], 0, -1)))

    ax.set_xlabel("Image row")
    ax.set_zlabel("PSNR")
    plt.savefig("PSNR_impact_500.png")


if __name__ == "__main__":

    in_dir = sys.argv[1]

    onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    f = open(join(in_dir, onlyfilespaths[0]), 'rb')
    mse = np.load(f)
    f.close()
    psnr_results = 10*np.log10(255*255/(mse+epsilon))

    for filename in tqdm(onlyfilespaths[1:]):
        f = open(join(in_dir, filename), 'rb')
        mse= np.load(f)
        psnr_results = psnr_results + 10*np.log10(255*255/(mse+epsilon))
        f.close()

    psnr_results = psnr_results / len(onlyfilespaths)

    plot_results(psnr_results)
