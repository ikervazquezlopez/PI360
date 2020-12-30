from numba import jit, vectorize, float64, int32, cuda
import numpy as np
import cv2
import time
import math
import sys
from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epsilon = 0.000001


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


@cuda.jit
def region_MSE(original, interpolated, n, dim, out):
    w = dim[0]
    h = dim[1]

    for k in range(1, n[0]):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        while ty < h:
            while tx < w:
                sq_error = 0
                for i in range(-1, 2):
                    for j in range(-1,2):
                        if tx+i>0 and tx+i<w and ty+j>0 and ty+j<h:
                            e = original[ty, tx] - interpolated[ty+j, tx+i,k]
                            sq_error = sq_error + e*e
                out[ty, tx, k] =  sq_error / 9
                tx = tx + cuda.blockDim.x
            tx = cuda.threadIdx.x
            ty = ty + cuda.blockDim.x

@cuda.jit
def region_PSNR(mse_region, n, dim):
    w = dim[0]
    h = dim[1]

    for k in range(1, n[0]):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        while ty < h:
            while tx < w:
                v = (255*255) / (mse_region[ty,tx,k]+epsilon)
                mse_region[ty, tx, k] = v
                tx = tx + cuda.blockDim.x
            tx = cuda.threadIdx.x
            ty = ty + cuda.blockDim.x

def compute_region_PSNR(filename, in_dir):
    n = np.array([16]) # Max length of interpolated pixels

    stream = cuda.stream()

    # Read the original image
    original = cv2.imread(join(in_dir, filename))
    original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0]
    original = original.astype(np.float64)

    dim = np.array([original.shape[1], original.shape[0]])

    template = np.zeros_like(original)
    template = np.repeat(template[:, :, np.newaxis], n, axis=2)

    # Create images in GPU
    d_original = cuda.to_device(np.ascontiguousarray(original), stream=stream)
    d_n = cuda.to_device(np.ascontiguousarray(n), stream=stream)
    d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)

    # Intepolate image
    d_interpolated = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    interpolate_image[512,512](d_original, d_n, d_dim, d_interpolated)

    # Compute MSE of a pixel in a 3x3 region
    d_mse_region = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    region_MSE[512,512](d_original, d_interpolated, d_n, d_dim, d_mse_region)
    region_PSNR[512,512](d_mse_region, d_n, d_dim)
    psnr_region = d_mse_region.copy_to_host()

    return psnr_region


if __name__ == "__main__":

    #filename = sys.argv[1]
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    psnr_region = 10*np.log10(compute_region_PSNR(onlyfilespaths[0], in_dir))

    for f in tqdm(onlyfilespaths[1:]):
        cuda_psnr = compute_region_PSNR(f, in_dir)
        psnr_region = psnr_region + 10*np.log10(cuda_psnr)

    psnr_region = psnr_region / 30#len(onlyfilespaths)

    tmp = psnr_region[:,:,4]
    cv2.imwrite("psnr_region_results_4.png", tmp)



    name = "psnr_region_results"
    with open(join(out_dir, "{}.npy".format(name)), 'wb') as f:
        np.save(f, psnr_region)
