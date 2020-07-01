from numba import jit, vectorize, float64, int32, cuda
import numpy as np
import cv2
import time
import math
import sys


@cuda.jit
def image_squared_error(img, rec, dim, out):
    w = dim[0]
    h = dim[1]
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    while ty < h:
        while tx < w:
            out[ty,tx] = (img[ty,tx]-rec[ty,tx]) * (img[ty,tx]-rec[ty,tx])
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.blockDim.x


@cuda.jit
def row_sum(img, dim, max_levels):
    w = dim[0]
    h = dim[1]
    ty = cuda.blockIdx.x
    levels = max_levels[0]
    while ty < h:
        for offset in range(1, levels+1):
            tx = cuda.threadIdx.x * 2**offset
            while tx < w-(2**offset/2):
                img[ty,tx] = img[ty, tx] + img[ty, tx+int(2**offset/2)]
                tx = tx + cuda.blockDim.x * 2**offset
            cuda.syncthreads()
        ty = ty + cuda.blockDim.x





if __name__ == "__main__":

    filename = sys.argv[1]
    original_dir = sys.argv[2]
    interpolated_dir = sys.argv[3]
    output_dir = sys.argv[4]

    stream = cuda.stream()

    # Read the original image
    original = cv2.imread(join(original_dir, filename))
    original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0]
    original = original.astype(np.uint64) / 255

    # Read interpolated image
    original = cv2.imread(join(original_dir, filename))
    original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0]
    original = original.astype(np.uint64) / 255

    dim = np.array([original.shape[1], original.shape[0]])

    max_levels = np.array([int(math.log2(original.shape[1]))]) # summation levels in CUDA for logn performance
    template = np.zeros_like(original)


    # Create images in GPU
    d_original = cuda.to_device(np.ascontiguousarray(original), stream=stream)
    d_interpolated = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)
    d_p = cuda.to_device(np.ascontiguousarray(p), stream=stream)

    interpolate_image[512,512](d_original, d_p, d_dim, d_interpolated)

    # Compute image squared error original-interpolated
    d_squared_error = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    image_squared_error[512,512](d_original, d_interpolated, d_dim, d_squared_error)

    # Sum squared errors
    d_max_levels = cuda.to_device(np.ascontiguousarray(max_levels), stream=stream)
    row_sum[512,512](d_squared_error, d_dim, d_max_levels)


    psnr_results = d_squared_error.copy_to_host()
    #print(psnr_results)
    psnr_results = 10*np.log10(1/(psnr_results[:,0]/dim[0]+epsilon))
    #print(psnr_results[0:10])
