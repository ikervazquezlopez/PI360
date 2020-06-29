import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


MAX_I2 = 255*255

res_16k = 8196*16384

res_powers = list(range(10,15))
res_list = np.array([2**i * 2**(i-1) for i in res_powers])
error_in_px = np.array(list(range(1,255*255)))

resolution_labels = ["1K","2K","4K","8K","16K"]


def fixed_resolution_PSNR_impact(resolution):
    psnr = 10 * np.log10(MAX_I2 / (error_in_px/res_16k))
    plt.plot(error_in_px, psnr)
    plt.title("Pixel error impact in PSNR")
    plt.xlabel("Pixel error")
    plt.ylabel("PSNR")
    plt.savefig("PSNR_impact_per_value.png")

def fixed_error_PSNR_impact(error):
    psnr = 10 * np.log10(MAX_I2 / (error/res_list))
    plt.plot(res_powers, psnr)
    ax = plt.axes()
    plt.title("Pixel error impact in PSNR with error {}".format(error))
    plt.xlabel("Resolution")
    ax.set_xticks(np.array(list(range(0,len(resolution_labels)))))
    ax.set_xticklabels(resolution_labels)
    plt.ylabel("PSNR")
    plt.savefig("PSNR_impact_per_resolution.png")

def single_pixel_PSNR_impact():
    error_in_px_mat = np.repeat(error_in_px[:, np.newaxis], res_list.shape[0],axis=1)
    res_list_mat = np.repeat(res_list[:, np.newaxis], error_in_px.shape[0],axis=1)
    psnr = 10 * np.log10(MAX_I2 / (error_in_px_mat/res_list_mat.transpose()))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.array(list(range(0, psnr.shape[1])))
    y = np.array(list(range(0, psnr.shape[0])))
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, np.flip(y), psnr)
    ax.set_xlabel("Resolution")
    ax.set_xticklabels(resolution_labels)
    ax.set_xticks(np.array(list(range(0,5))))

    ax.set_ylabel("Pixel error")
    ax.set_zlabel("PSNR")
    plt.savefig("single_pixel_PSNR_impact.png")


single_pixel_PSNR_impact()
