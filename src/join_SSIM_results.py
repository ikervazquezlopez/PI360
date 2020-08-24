import sys
import os
import cv2
from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_results(SSIM):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.array(list(range(0, SSIM.shape[1])))
    y = np.array(list(range(0, SSIM.shape[0])))
    x, y = np.meshgrid(x, y)

    ax.plot_surface(y, np.flip(x), SSIM)
    ax.set_ylabel("Interpolation length")
    ax.set_yticklabels(list(range(SSIM.shape[1], 0, -1)))

    ax.set_xlabel("Image row")
    ax.set_zlabel("SSIM")
    plt.savefig("SSIM_impact_500.png")

def plot_SSIM_in_lines(SSIM):

    last_row = SSIM.shape[0]-1
    half_row = int(last_row/2)
    line_0 = SSIM[0,:]
    line_half = SSIM[half_row,:]
    line_last = SSIM[last_row,:]
    x = np.array(list(range(0, SSIM.shape[1])))
    fig, ax = plt.subplots()
    ax.plot(x, line_0, 'r--', label="Row 0")
    ax.plot(x, line_half, 'b--', label="Row {}".format(half_row))
    ax.plot(x, line_last, 'g--', label="Row {}".format(last_row))
    ax.set_xlabel("Interpolation length")
    ax.set_ylabel("SSIM")

    legend = ax.legend(loc='upper center')
    legend.get_frame().set_facecolor('C0')
    plt.savefig("SSIM_impact_500_per_line.png")


if __name__ == "__main__":

    in_dir = sys.argv[1]

    onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    f = open(join(in_dir, onlyfilespaths[0]), 'rb')
    SSIM_results = np.load(f)
    f.close()

    for filename in tqdm(onlyfilespaths[1:]):
        f = open(join(in_dir, filename), 'rb')
        SSIM= np.load(f)
        SSIM_results = SSIM_results + SSIM
        f.close()

    SSIM_results = SSIM_results / len(onlyfilespaths)

    plot_results(SSIM_results)
    plot_SSIM_in_lines(SSIM_results)
