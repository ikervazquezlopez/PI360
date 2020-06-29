import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

L = 254
k1 = 0.01
k2 = 0.03
c1 = (k1*L)*(k1*L)
c2 = (k2*L)*(k2*L)




res_list = [(1024,512), (2048,1024), (4096,2048), (8192,4096), (16384,8192)]
resolution_labels = ["1K","2K","4K","8K","16K"]
error_in_px = np.array(list(range(0,255)))


def compute_SSIM(Io, Ip):
    uo = np.mean(Io)
    up = np.mean(Ip)

    vo = np.var(Io)
    vp = np.var(Ip)

    cop = np.cov(Io.flatten(),Ip.flatten())[0,1]

    l = (2*uo*up + c1) / (uo*uo + up*up + c1)
    c = (2*vo*vp + c2) / (vo*vo + vp*vp + c2)
    s = (cop + c2/2) / (vo*vp + c2/2)

    SSIM = l * c * s
    #top = (2*uo*up + c1) * (2*cop + c2)
    #bot = (uo*uo + up*up + c1) * (vo*vo + vp*vp + c2)
    #SSIM = top / bot

    return SSIM

def fixed_resolution_SSIM_impact(resolution):

    SSIM_list = []

    Io = np.full(resolution, 1, dtype=np.float64)
    Ip = np.full(resolution, 1, dtype=np.float64)
    for error in range(0,255):
        Ip[0:500,0:1000] = error/255

        SSIM = compute_SSIM(Io, Ip)
        SSIM_list.append(SSIM)

    #SSIM_list = np.array(SSIM_list)
    #print(error_in_px.shape, SSIM_list.shape)

    plt.plot(error_in_px, SSIM_list)
    #ax = plt.axes()
    plt.title("Pixel error impact in SSIM with resoulution {}".format(resolution))
    plt.xlabel("Error in pixel")
    plt.ylabel("SSIM")
    plt.ylim(0, 1.1)
    plt.savefig("SSIM_impact_per_value_{}_{}.png".format(resolution[0],resolution[1]))
    return SSIM_list



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



def SSIM_impact():
    SSIM_data = []
    for res in res_list:
        print(res)
        SSIM_data.append(fixed_resolution_SSIM_impact(res))

    SSIM_data = np.savetxt("SSIM_data.csv", SSIM_data, delimiter=",")
    #pickle.dump(SSIM_data, "SSIM_data.pkl")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.array(res_list)
    y = np.array(list(range(0, np.max(error_in_px))))
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, np.flip(y), np.array(SSIM_data))
    ax.set_xlabel("Resolution")
    ax.set_xticklabels(resolution_labels)
    ax.set_xticks(np.array(list(range(0,5))))

    ax.set_ylabel("Pixel error")
    ax.set_zlabel("SSIM")
    plt.savefig("single_pixel_SSIM_impact.png")


def plot_SSIM_data_from_csv(filename):
    SSIM_data = np.loadtxt(filename, delimiter=",")

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.array(list(range(0,SSIM_data.shape[0])))
    y = np.array(list(range(0, SSIM_data.shape[1])))
    x, y = np.meshgrid(x, y)

    print(x.shape, y.shape, SSIM_data.shape)

    ax.plot_surface(x, y, np.transpose(SSIM_data))
    ax.set_xlabel("Resolution")
    ax.set_xticklabels(resolution_labels)
    ax.set_xticks(np.array(list(range(0,5))))

    ax.set_ylabel("Pixel error")
    ax.set_zlim(0, 1)
    ax.set_zlabel("SSIM")
    plt.savefig("single_pixel_SSIM_impact_1-0_100px.png")


#SSIM_impact()
plot_SSIM_data_from_csv("SSIM_data.csv")
