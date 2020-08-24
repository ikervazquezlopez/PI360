from numba import jit, vectorize, float64, int32, cuda
import numpy as np
import cv2
import time
import math
import sys
from os.path import isfile, join, isdir

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

L = 255
k1 = 0.01
k2 = 0.03
c1 = (k1*L)*(k1*L)
c2 = (k2*L)*(k2*L)
c3 = c2 / 2

epsilon = 0.000000001 # value to avoid dividing by 0

@cuda.jit
def interpolate_image(img, n, dim, out):
    w = dim[0]
    h = dim[1]

    for i in range(1, n[0]):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        inter_length = i*2
        while ty < h:
            while tx < w:
                k = tx % inter_length
                value = 0
                if k == 0:
                    value = img[ty, tx]
                else:
                    p = k/inter_length
                    a0 = tx-k
                    a1 = a0+inter_length
                    if a1 >= w:
                        value = ((1-p)*img[ty, a0] + p*img[ty, w-1])
                    if a0 >= 0 and a1 < w:
                        value = ((1-p)*img[ty, a0] + p*img[ty, a1])
                out[ty, tx, i] =  value
                tx = tx + cuda.blockDim.x
            tx = cuda.threadIdx.x
            ty = ty + cuda.blockDim.x




if __name__ == "__main__":

    filename = sys.argv[1]
    original_dir = sys.argv[2]
    output_dir = sys.argv[3]

    n = np.array([16]) # pixels to interpolate in row

    stream = cuda.stream()

    # Read the input image
    original = cv2.imread(join(original_dir, filename))
    original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0]
    original = original.astype(np.uint64) #/ 255

    dim = np.array([original.shape[1], original.shape[0]])


    max_levels = np.array([int(math.log2(original.shape[1]))]) # summation levels in CUDA for logn performance
    template = np.zeros_like(original, dtype=np.float64)


    # Create images in GPU
    d_original = cuda.to_device(np.ascontiguousarray(original), stream=stream)
    interpolated = np.repeat(template[:,:,np.newaxis], n, axis=2)
    d_interpolated = cuda.to_device(np.ascontiguousarray(interpolated), stream=stream)
    d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)
    d_n = cuda.to_device(np.ascontiguousarray(n), stream=stream)

    interpolate_image[512,512](d_original, d_n, d_dim, d_interpolated)
    interpolated = d_interpolated.copy_to_host()

    # Compute image means
    ori_img_mean = np.mean(original, axis=1)
    ori_img_mean_r = np.repeat(ori_img_mean[:, np.newaxis], original.shape[1], axis=1)
    int_img_mean = np.mean(interpolated, axis=1)
    int_img_mean_r = np.repeat(int_img_mean[:, np.newaxis,:], interpolated.shape[1], axis=1 )


    # Compute image variances
    ori_img_var = np.var(original, axis=1)
    int_img_var = np.var(interpolated, axis=1)


    # Compute covariance
    tmp0 = original-ori_img_mean_r
    tmp0 = np.repeat(tmp0[:,:,np.newaxis], n, axis=2)
    tmp1 = interpolated-int_img_mean_r
    covariance = np.mean(tmp0*tmp1, axis=1)

    # Compute SSIM
    ori_img_mean = np.repeat(ori_img_mean[:,np.newaxis], n, axis=1)
    ori_img_var = np.repeat(ori_img_var[:, np.newaxis], n, axis=1)
    ori_img_stddev = np.sqrt(ori_img_var)
    int_img_stddev = np.sqrt(int_img_var)

    l = (2*ori_img_mean*int_img_mean + c1) / ((ori_img_mean*ori_img_mean) + (int_img_mean*int_img_mean) + c1)
    c = (2*ori_img_stddev*int_img_stddev + c2) / (ori_img_var + int_img_var + c2)
    s = (covariance + c3) / (ori_img_stddev*int_img_stddev + c3)

    SSIM = l * c * s

    # Save SSIM
    name = filename.split('.')[0]
    with open(join(output_dir, "{}.npy".format(name)), 'wb') as f:
        np.save(f, SSIM)
