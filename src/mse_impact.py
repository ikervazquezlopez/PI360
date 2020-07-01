from numba import jit, vectorize, float64, int32, cuda
import numpy as np
import cv2
import time
import math
import sys
from os.path import isfile, join, isdir


"""
@img original image
@n range to pixels to interpolate
@dim image dimensions
@out output image (requires an extra dimension to store the different interpolation pixel lengths)
"""
@cuda.jit
def interpolated_image_MSE(img, n, dim, out):
    w = dim[0]
    h = dim[1]

    for i in range(2,n[0]):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        while ty < h:
            while tx < w:
                start_idx = tx-i+1
                end_idx = tx+i-1
                if start_idx < 0:
                    start_idx = 0
                if end_idx >= w:
                    end_idx = w-1
                MSE = 0
                for k in range(0, end_idx-start_idx):
                    inter_value = 0
                    if tx-i < 0:
                        inter_value = (1-k/(2*i))*img[ty,0] + (k/(2*i))*img[ty,tx+i]
                    if tx+i >= w:
                        inter_value = (1-k/(2*i))*img[ty,tx-i] + (k/(2*i))*img[ty,w-1]
                    if tx-i > 0 and tx+i < w:
                        inter_value = (1-k/(2*i))*img[ty,tx-i] + (k/(2*i))*img[ty,tx+i] # Compute the interpolated value
                    sqr_err = (inter_value-img[ty,start_idx+k])*(inter_value-img[ty,start_idx+k]) # Squared error
                    MSE = MSE + sqr_err#/(w*h)
                out[ty,tx,i] = MSE
                tx = tx + cuda.blockDim.x
            tx = cuda.threadIdx.x
            ty = ty + cuda.blockDim.x



@cuda.jit
def row_sum(mse, n, dim, max_levels):
    w = dim[0]
    h = dim[1]
    ty = cuda.blockIdx.x
    levels = max_levels[0]
    while ty < h:
        for offset in range(1, levels+1):
            tx = cuda.threadIdx.x * 2**offset
            while tx < w-(2**offset/2):
                for i in range(1, n[0]):
                    mse[ty,tx,i] = mse[ty, tx, i] + mse[ty, tx+int(2**offset/2), i]
                tx = tx + cuda.blockDim.x * 2**offset
            cuda.syncthreads()
        ty = ty + cuda.blockDim.x


if __name__ == "__main__":

    filename = sys.argv[1]
    original_dir = sys.argv[2]
    output_dir = sys.argv[3]

    n = np.array([16]) # Max length of interpolated pixels

    stream = cuda.stream()

    # Read the original image
    original = cv2.imread(join(original_dir, filename))
    original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0]
    original = original.astype(np.uint64)

    dim = np.array([original.shape[1], original.shape[0]])

    max_levels = np.array([int(math.log2(original.shape[1]))]) # summation levels in CUDA for logn performance
    template = np.zeros_like(original)
    template = np.repeat(template[:, :, np.newaxis], n, axis=2)


    # Create images in GPU
    d_original = cuda.to_device(np.ascontiguousarray(original), stream=stream)
    d_n = cuda.to_device(np.ascontiguousarray(n), stream=stream)
    d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)

    # Compute image squared error original-interpolated
    d_squared_error = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    interpolated_image_MSE[512,512](d_original, d_n, d_dim, d_squared_error)

    inter = d_squared_error.copy_to_host()


    # Sum squared errors and store them in column 0
    d_max_levels = cuda.to_device(np.ascontiguousarray(max_levels), stream=stream)
    row_sum[512,512](d_squared_error, d_n, d_dim, d_max_levels)


    squared_error_results = d_squared_error.copy_to_host()
    squared_error_results = squared_error_results[:, 0, :] /(dim[0]*dim[1])


    name = filename.split('.')[0]
    with open(join(output_dir, "{}.npy".format(name)), 'wb') as f:
        np.save(f, squared_error_results)
