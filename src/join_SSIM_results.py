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
    plt.imshow(SSIM, cmap='inferno',interpolation="bilinear", aspect='auto')
    plt.colorbar()
    plt.xlabel("Interpolation length")
    plt.ylabel("Image row")
    plt.xticks(list(range(SSIM.shape[1], 0, -1)), labels=list(range(SSIM.shape[1]*2, 0, -2)))
    plt.title("SSIM")
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.array(list(range(0, SSIM.shape[1])))
    y = np.array(list(range(0, SSIM.shape[0])))
    x, y = np.meshgrid(x, y)

    ax.plot_surface(y, np.flip(x), SSIM)
    ax.set_ylabel("Interpolation length")
    ax.set_yticklabels(list(range(SSIM.shape[1], 0, -1)))
    Axes3D.set_zlim(ax, bottom=0,top=1.1)

    ax.set_xlabel("Image row")
    ax.set_zlabel("SSIM")
    """
    plt.savefig("plot_results/SSIM_impact.png")

def plot_SSIM_in_lines(SSIM):

    last_row = SSIM.shape[0]-1
    half_row = int(last_row/2)
    line_0 = SSIM[0,:]
    line_half = SSIM[half_row,:]
    line_last = SSIM[last_row,:]
    x = np.array(list(range(0, SSIM.shape[1])))
    fig, ax = plt.subplots()
    ax.plot(x, line_0, 'bo', label="Row 0")
    ax.plot(x, line_half, 'r--', label="Row {}".format(half_row))
    ax.plot(x, line_last, 'c--', label="Row {}".format(last_row))
    ax.set_ylim(bottom=0,top=1.1)
    ax.set_xlabel("Interpolation length")
    ax.set_ylabel("SSIM")

    legend = ax.legend(loc='lower right')
    #legend.get_frame().set_facecolor('C1')
    plt.savefig("plot_results/SSIM_impact_per_line.png")


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
    SSIM_results = SSIM_results[:,1:]
    plot_results(SSIM_results)
    plot_SSIM_in_lines(SSIM_results)
