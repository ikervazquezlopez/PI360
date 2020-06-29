import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from numba import jit, vectorize, float64, int32, cuda
import multiprocessing as mp
from multiprocessing import set_start_method
from os.path import isfile, join, isdir
import sys

ROAD_SIZE = 4
kernel_size = 3

@cuda.jit
def ROAD_3x3(img, out, dim):
    w = dim[0]
    h = dim[1]
    KERNEL_SIZE = 3
    up = int(math.floor(KERNEL_SIZE/2))
    low = -up
    mem = cuda.local.array(3*3, float64)
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    while ty < h:
        while tx < w:
            for x in range(low, up):
                for y in range(low,up):
                    indy = ty+y
                    indx = tx+x
                    if indx < 0 or indx > w:
                        indx = tx;
                    if indy < 0 or indy > h:
                        indy = ty
                    mem[x + up + KERNEL_SIZE*(y+up)] = abs(img[ty,tx] - img[indy,indx])

            # Sort the array in descending order
            for i in range(0,KERNEL_SIZE*KERNEL_SIZE):
                max = mem[i]
                for j in range(i+1,KERNEL_SIZE*KERNEL_SIZE):
                    if mem[j] > max:
                        mem[i] = mem[j]
                        mem[j] = max
                        max = mem[i]

            # Compute ROAD metric
            ROAD = 0
            for i in range(0, ROAD_SIZE):
                ROAD = ROAD + mem[i]
            out[ty,tx] = ROAD

            # Erase all mem data
            for i in range(0,KERNEL_SIZE*KERNEL_SIZE):
                mem[i] = 0

            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.blockDim.x



if __name__ == '__main__':

    img_name = sys.argv[1]
    in_dir = sys.argv[2]
    out_dir = sys.argv[3]

    img_path = join(in_dir, img_name)

    # Get the image Y (luminance)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img = cv2.split(img)[0]
    img = img.astype(np.float64) /255

    """
    rec = cv2.imread("rec.png", cv2.IMREAD_COLOR)
    rec = cv2.cvtColor(rec, cv2.COLOR_BGR2YCR_CB)
    rec = cv2.split(rec)[0]

    diff = np.abs(img-rec)
    diff = diff.astype(np.int32)
    """

    dim = np.array([img.shape[1],img.shape[0]])

    stream = cuda.stream()

    # Copy image data to GPU (device)
    d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)
    d_out = np.zeros_like(img)
    d_out = cuda.to_device(np.ascontiguousarray(d_out), stream=stream)
    d_diff = cuda.to_device(np.ascontiguousarray(img), stream=stream)

    # Compute ROAD metric for the image
    ROAD_3x3[1024,1024](d_diff, d_out, d_dim)

    # Copy results to CPU (host)
    h_road = d_out.copy_to_host()

    out_path = join(out_dir, img_name)
    cv2.imwrite(out_path, h_road*255)
