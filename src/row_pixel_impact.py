from numba import jit, vectorize, float64, int32, cuda
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt

L = 255
k1 = 0.01
k2 = 0.03
c1 = (k1*L)*(k1*L)
c2 = (k2*L)*(k2*L)

epsilon = 0.000000001 # value to avoid dividing by 0


def compute_variance(img):
    img = img.astype(np.float64)
    n = img.shape[0] * img.shape[1]
    mean = np.mean(img)
    xi = np.sum(img)
    xi_2 = np.sum(img*img)
    var = xi_2/n - 2*mean*xi/n + mean*mean
    return var, xi, xi_2

def compute_covariance(img0, img1):
    imgx = img0.astype(np.float64)
    imgy = img1.astype(np.float64)
    n = img1.shape[0] * img1.shape[1]
    if n != img1.shape[0]*img1.shape[1]:
        print("ERROR: image shapes are different!")
        print(img0.shape)
        print(img1.shape)
        exit(-1)
    mean_x = np.mean(imgx)
    mean_y = np.mean(imgy)
    xi = np.sum(imgx)
    yi = np.sum(imgy)
    xiyi = np.sum(imgx*imgy)
    cov = (xiyi - mean_y*xi - mean_x*yi) / n + mean_x*mean_y
    return cov, mean_x, mean_y, xi, yi, xiyi


@cuda.jit
def interpolate_image(img, p, dim, out):
    w = dim[0]
    h = dim[1]
    tx = cuda.threadIdx.x * p[0]
    ty = cuda.blockIdx.x
    while ty < h:
        while tx+p[0]-1 < w:
            out[ty,tx] = img[ty,tx]
            for k in range(1, p[0]):
                if tx+k == w-1: # Set the last column as the original
                    out[ty,tx+k] = img[ty,w-1]
                    break
                else: # Interpolation
                    out[ty,tx+k] = (1-k/p[0])*img[ty, tx] + (k/p[0])*img[ty,tx+p[0]]
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x * p[0]
        ty = ty + cuda.blockDim.x


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

@cuda.jit
def sum_avg(img, dim):
    w = dim[0]
    h = dim[1]
    ty = cuda.threadIdx.x
    while ty < h:
        img[ty,0] = img[ty,0] / w
        ty = ty + cuda.blockDim.x

@cuda.jit
def psnr_gpu(squred_error, dim):
    w = dim[0]
    h = dim[1]
    max_I = 1 # 1 because we are working on 64 bits
    ty = cuda.threadIdx.x
    while ty < h:
        img[ty,0] = math.log10((max_I*max_I) / img[ty,0])
        ty = ty + cuda.blockDim.x


@cuda.jit
def means_from_interpolated(img, rec, p, mean, dim):
    w = dim[0]
    h = dim[1]
    n = w * h
    tx = cuda.threadIdx.x * p[0]
    ty = cuda.blockIdx.x
    while ty < h:
        while tx+p[0]-1 < w:
            for k in range(1, p[0]):
                if tx+k == w-1:
                    break
                else:
                    mean[ty,tx] = mean[ty, tx] - img[ty,tx+k]/n + rec[ty,tx+k]/n
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x * p[0]
        ty = ty + cuda.blockDim.x

@cuda.jit
def means_from_interpolated_per_row(img, rec, p, mean, dim):
    w = dim[0]
    h = dim[1]
    n = w * h
    tx = cuda.threadIdx.x * p[0]
    ty = cuda.blockIdx.x
    while ty < h:
        while tx+p[0]-1 < w:
            for k in range(1, p[0]):
                if tx+k == w-1:
                    break
                else:
                    mean[ty,tx] = mean[ty, tx] - img[ty,tx+k]/w + rec[ty,tx+k]/w
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x * p[0]
        ty = ty + cuda.blockDim.x


@cuda.jit
def variance_from_interpolated(img, inter, var, mean, xi, xi_2, dim):
    w = dim[0]
    h = dim[1]
    n = w * h
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    while ty < h:
        while tx < w:
            _rm_el = img[ty,tx]
            _add_el = inter[ty,tx]
            _xi = xi[0] - _rm_el +_add_el
            _xi_2 = xi_2[0] - (_rm_el*_rm_el) + (_add_el*_add_el)
            var[ty,tx] = _xi_2/n - 2*mean[ty,tx]*_xi/n + mean[ty,tx]*mean[ty,tx]
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.blockDim.x


@cuda.jit
def covariance_from_interpolated(img, inter, cov, mean_x, mean_y, xi, yi, xiyi, dim):
    w = dim[0]
    h = dim[1]
    n = w * h
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    while ty < h:
        while tx < w:
            _rm_el = img[ty,tx]
            _add_el = inter[ty,tx]
            _xi = xi[0] - _rm_el +_add_el
            _xiyi = xiyi[0] - _rm_el*_rm_el + _rm_el*_add_el
            cov[ty,tx] = (_xiyi - mean_y[ty,tx]*_xi - mean_x[ty,tx]*yi[0]) / n + mean_x[ty,tx]*mean_y[ty,tx]
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.blockDim.x



@cuda.jit
def recompute_covariance(cov, mean_x, mean_y, xi, yi, xiyi, n, removed_el, added_el, mirror_el_y):
    _mean_x = mean_x[0] - removed_el[0]/n[0] + added_el[0]/n[0] # replace mean_x element
    _mean_x = _mean_x
    _xi = xi[0] + added_el[0] - removed_el[0]
    _xiyi = xiyi[0] + mirror_el_y[0]*(added_el[0] - removed_el[0])
    cov[0] = (_xiyi - mean_y[0]*_xi - _mean_x*yi[0]) / n[0] + _mean_x*mean_y[0]


@cuda.jit
def SSIM_from_interpolated(SSIM, ori_var_img, int_var_img, cov_img, ori_mean_img, int_mean_img, dim):
    w = dim[0]
    h = dim[1]
    n = w * h
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    while ty < h:
        while tx < w:
            numerator_0 = (2 * ori_mean_img[ty,tx] * int_mean_img[ty,tx]) + c1
            numerator_1 = (2 * cov_img[ty,tx]) + c2
            denominator_0 = (ori_mean_img[ty,tx]*ori_mean_img[ty,tx]) + (int_mean_img[ty,tx]*int_mean_img[ty,tx]) + c1
            denominator_1 = (ori_var_img[ty,tx]*ori_var_img[ty,tx]) + (int_var_img[ty,tx]*int_var_img[ty,tx]) + c2

            SSIM[ty,tx] = (numerator_0*numerator_1) / (denominator_0*denominator_1)
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.blockDim.x


@cuda.jit
def pixel_impact(img, out_img, size):
    w = size[1]
    h = size[0]
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    i = 0
    while tx < w and ty < h:
        tx = cuda.threadIdx.x + i*cuda.blockDim.x
        ty = cuda.blockIdx.y + i*cuda.blockDim.x
        i += 1
        if tx > 0 and tx < w-1 and ty < h:
            el = (img[ty, tx-1] + img[ty, tx+1]) / 2
            out_img[ty, tx] = (img[ty,tx] - el)*(img[ty,tx] - el) / (w*h)
            #out_img[ty, tx] = el





if __name__ == "__main__":

    stream = cuda.stream()

    # Read the input image
    original = cv2.imread("data/original0.png")
    #original = cv2.resize(original, None, fx=0.25, fy=0.25)
    original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0]
    original = original.astype(np.uint64) / 255
    #original = np.full_like(original, 1)
    dim = np.array([original.shape[1], original.shape[0]])
    print(original.shape)

    p = np.array([64]) # pixels to interpolate in row
    max_levels = np.array([int(math.log2(original.shape[1]))]) # summation levels in CUDA for logn performance
    template = np.zeros_like(original)


    # Create images in GPU
    d_original = cuda.to_device(np.ascontiguousarray(original), stream=stream)
    d_interpolated = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)
    d_p = cuda.to_device(np.ascontiguousarray(p), stream=stream)

    interpolate_image[512,512](d_original, d_p, d_dim, d_interpolated)


""" FOR PSNR
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

    plt.plot(psnr_results)
    plt.title("PSNR score per row (64px stride interpolation)")
    plt.ylabel('PSNR')
    plt.xlabel("Image row")
    plt.axis([0, psnr_results.shape[0], 0, 100])
    plt.savefig("pixel_impact.png")

    interpolated = d_interpolated.copy_to_host()
    cv2.imwrite("interpolated.png", interpolated*255)

"""

    # Create the image means in the GPU
    ori_mean_img = np.full(original.shape, np.mean(original), original.dtype)
    d_ori_mean_img = cuda.to_device(np.ascontiguousarray(ori_mean_img), stream=stream)
    d_int_mean_img = cuda.to_device(np.ascontiguousarray(ori_mean_img), stream=stream)
    #means_from_interpolated[512,512](d_original, d_interpolated, d_p, d_int_mean_img,
    #                                    d_dim)
    means_from_interpolated_per_row[512,512](d_original, d_interpolated, d_p, d_int_mean_img,
                                        d_dim)


    # Create the image variances in the GPU
    ori_var, ori_xi, ori_xi_2 = compute_variance(original)
    ori_var = np.full(original.shape, ori_var, original.dtype)
    d_ori_var = cuda.to_device(np.ascontiguousarray(ori_var), stream=stream)
    d_int_var = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    d_ori_xi = cuda.to_device(np.ascontiguousarray(np.array([ori_xi])), stream=stream)
    d_ori_xi_2 = cuda.to_device(np.ascontiguousarray(np.array([ori_xi_2])), stream=stream)
    variance_from_interpolated[512,512](d_original, d_interpolated, d_int_var,
                                        d_ori_mean_img, d_ori_xi, d_ori_xi_2,
                                        d_dim)


    # Create the image covariance in the GPU
    _, _, _, int_xi, ori_yi, xiyi = compute_covariance(original, original)
    d_int_xi = cuda.to_device(np.ascontiguousarray(np.array([int_xi])), stream=stream)
    d_ori_yi = cuda.to_device(np.ascontiguousarray(np.array([ori_yi])), stream=stream)
    d_xiyi = cuda.to_device(np.ascontiguousarray(np.array([xiyi])), stream=stream)
    d_cov = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    covariance_from_interpolated[512,512](d_original, d_interpolated,
                                        d_cov, d_int_mean_img, d_ori_mean_img,
                                        d_int_xi, d_ori_yi, d_xiyi, d_dim)


    del(d_original)
    del(d_interpolated)
    del(d_ori_xi)
    del(d_ori_xi_2)
    del(d_int_xi)
    del(d_ori_yi)
    del(d_xiyi)

    # Compute SSIM in the GPU
    d_SSIM = cuda.to_device(np.ascontiguousarray(template), stream=stream)

    SSIM_from_interpolated[512,512](d_SSIM, d_ori_var, d_int_var, d_cov,
                                    d_ori_mean_img, d_int_mean_img, d_dim)


    ori_mean = np.mean(original)
    int_mean = np.mean(interpolated)
    ori_var = np.var(original)
    int_var = np.var(interpolated)
    cov = np.cov(original.flatten(), interpolated.flatten())[0,0]


    SSIM = d_SSIM.copy_to_host()
    SSIM = np.mean(SSIM, axis=1)
    print(SSIM.shape)
    print(SSIM)

    plt.clf()
    plt.ylabel("Row")
    plt.ylim(0,1.5)
    plt.plot(SSIM)
    plt.savefig('pixel_impact.png')

    cv2.imwrite("interpolated.png", interpolated*255)

    #mmeans = d_int_mean_img.copy_to_host()
    #print(np.var(mmeans))
    #print(np.min(mmeans), np.max(mmeans))
    diff = np.abs(original - interpolated)
    print("Mean: " + str(np.mean(diff)))
    print("Min: " + str(np.min(diff)), "  Max: " + str(np.max(diff)))
    #cv2.imwrite("means.png", mmeans*255)
