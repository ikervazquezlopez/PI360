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


def compute_variance(img):
    img = img.astype(np.int32)
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
def printer(inarray):
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    if tx < inarray.shape[0] and ty < inarray.shape[1]:
        inarray[tx, ty] = 0
    #cuda.syncthreads()

"""
@cuda.jit
def replace_mean_element(mean, n, removed_el, added_el):
    mean[0] = mean[0] - removed_el[0]/n[0] + added_el[0]/n[0]
"""

@cuda.jit
def interpolate_image(img, dim, out):
    w = dim[0]
    h = dim[1]
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    while ty < h:
        while tx < w:
            if tx == 0 or tx == w-1:
                pred = img[ty,tx]
            else:
                pred = (img[ty, tx-1] + img[ty,tx+1]) / 2
            out[ty,tx] = pred
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.blockDim.x


@cuda.jit
def means_from_interpolated(img, rec, mean, dim):
    w = dim[0]
    h = dim[1]
    n = w * h
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    while ty < h:
        while tx < w:
            mean[ty,tx] = mean[ty, tx] - img[ty,tx]/n + rec[ty,tx]/n
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
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
            """
            l = (2 * ori_mean_img[ty,tx] * int_mean_img[ty,tx] + c1) / (ori_mean_img[ty,tx]*ori_mean_img[ty,tx] + int_mean_img[ty,tx]*int_mean_img[ty,tx] + c1)
            c = (2 * cov_img[ty,tx] + c2) / (ori_var_img[ty,tx]*ori_var_img[ty,tx] + int_var_img[ty,tx]*int_var_img[ty,tx] + c2)
            s = (cov_img[ty,tx] + c2/2) / (ori_var_img[tx,ty]*int_var_img[tx,ty] + c2/2)
            SSIM[tx,ty] = 0.0# l * c * s
            """
            tx = tx + cuda.blockDim.x
        tx = cuda.threadIdx.x
        ty = ty + cuda.blockDim.x



"""
@cuda.jit
def compute_ssim()
"""

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
            """
            if  cuda.threadIdx.x == 1 and cuda.blockIdx.x == 1:
                print((img[ty,tx] - el) / (w*h))
            """
            out_img[ty, tx] = (img[ty,tx] - el)*(img[ty,tx] - el) / (w*h)
            #out_img[ty, tx] = el


"""
img1 = cv2.imread("original1.png")
print(img1.shape)
img1 = cv2.resize(img1, None, fx=0.25, fy=0.25)
img1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB))[0]
img1 = img1.astype(np.float32)


d_out = img1.copy() #np.zeros_like(img1)

size = np.array([img1.shape[1], img1.shape[0]])

stream = cuda.stream()

d_img1 = cuda.to_device(np.ascontiguousarray(img1), stream=stream)
d_out = cuda.to_device(np.ascontiguousarray(d_out), stream=stream)
d_size = cuda.to_device(np.ascontiguousarray(size), stream=stream)

pixel_impact[512, 512](d_img1, d_out, d_size)

cuda.synchronize()

#h_out = np.ascontiguousarray(np.zeros_like(img1))
h_out = d_out.copy_to_host()
print(h_out)

#print(cv2.PSNR(img1, h_out))
#means = np.mean(h_out, axis=0)
h_out  = h_out + 0.0000001
print(np.mean(10*np.log10((255*255) / h_out), axis=0))
#print(np.max(np.sum(h_out, axis=0)))

plt.clf()
plt.xlabel("Row")
plt.ylabel("PSNR")
plt.ylim(bottom=0, top=90)
plt.plot(np.mean(10*np.log10((255*255) / h_out), axis=0))
plt.savefig('pixel_impact_cuda.png')

#cv2.imwrite("out.png", img1_d)

stream = cuda.stream()

img2 = img1.copy()
img2[0,0] = 128

_mean = np.mean(img1)
_var, _xi, _xi_2 = compute_variance(img1)
_cov, _mean_x, _mean_y, _xi, _yi, _xiyi = compute_covariance(img1, img2)
print(_cov)



n = np.zeros((1)); n[0] = img1.shape[0]*img1.shape[1]
mean_x = np.zeros((1)); mean_x[0] = _mean_x
mean_y = np.zeros((1)); mean_y[0] = _mean_y
xi = np.zeros((1)); xi[0] = _xi
yi = np.zeros((1)); yi[0] = _yi
xiyi = np.zeros((1)); xiyi[0] = _xiyi
removed_el = np.zeros((1)); removed_el[0] = img2[0,0]
added_el = np.zeros((1)); added_el[0] = 128
mirror_el = np.zeros((1)); mirror_el[0] = img1[0,0]
cov = np.zeros((1))

d_img2 = cuda.to_device(np.ascontiguousarray(img2), stream=stream)
d_mean_x = cuda.to_device(np.ascontiguousarray(mean_x), stream=stream)
d_mean_y = cuda.to_device(np.ascontiguousarray(mean_y), stream=stream)
d_xi = cuda.to_device(np.ascontiguousarray(xi), stream=stream)
d_yi = cuda.to_device(np.ascontiguousarray(yi), stream=stream)
d_xiyi = cuda.to_device(np.ascontiguousarray(xiyi), stream=stream)
d_n = cuda.to_device(np.ascontiguousarray(n), stream=stream)
d_removed_el = cuda.to_device(np.ascontiguousarray(removed_el), stream=stream)
d_added_el = cuda.to_device(np.ascontiguousarray(added_el), stream=stream)
d_mirror_el = cuda.to_device(np.ascontiguousarray(mirror_el), stream=stream)
d_cov = cuda.to_device(np.ascontiguousarray(cov), stream=stream)

recompute_covariance[1,1](d_cov, d_mean_x, d_mean_y, d_xi, d_yi, d_xiyi, d_n, d_removed_el, d_added_el, d_mirror_el)
cov = d_cov.copy_to_host()

print("======")
print(np.mean(np.cov(img1.flatten(), img2.flatten())))
print(cov)

"""





if __name__ == "__main__":

    stream = cuda.stream()

    # Read the input image
    original = cv2.imread("data/original0.png")
    #original = cv2.resize(original, None, fx=0.25, fy=0.25)
    original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0]
    original = original.astype(np.float32) / 255
    dim = np.array([original.shape[1], original.shape[0]])
    print(original.shape)

    template = np.zeros_like(original)


    # Create images in GPU
    d_original = cuda.to_device(np.ascontiguousarray(original), stream=stream)
    d_interpolated = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    d_dim = cuda.to_device(np.ascontiguousarray(dim), stream=stream)

    interpolate_image[512,512](d_original, d_dim, d_interpolated)

    interpolated = d_interpolated.copy_to_host()

    # Create the image means in the GPU
    ori_mean_img = np.full(original.shape, np.mean(original), original.dtype)
    d_ori_mean_img = cuda.to_device(np.ascontiguousarray(ori_mean_img), stream=stream)
    d_int_mean_img = cuda.to_device(np.ascontiguousarray(ori_mean_img), stream=stream)
    means_from_interpolated[512,512](d_original, d_interpolated, d_int_mean_img,
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

    """
    _, int_xi, int_xi_2 = compute_variance(interpolated)
    d_int_var = cuda.to_device(np.ascontiguousarray(template), stream=stream)
    d_int_xi = cuda.to_device(np.ascontiguousarray(np.array([int_xi])), stream=stream)
    d_int_xi_2 = cuda.to_device(np.ascontiguousarray(np.array([int_xi_2])), stream=stream)
    variance_from_interpolated[512,512](d_original, d_interpolated, d_int_var,
                                        d_int_mean_img, d_int_xi, d_int_xi_2,
                                        d_dim)
    """


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


    """
    numerator_0 = 2 * ori_mean * int_mean + c1
    numerator_1 = 2 * cov + c2
    denominator_0 = (ori_mean*ori_mean) + (int_mean*int_mean) + c1
    denominator_1 = (ori_var*ori_var) + (int_var*int_var) + c2
    """


    SSIM = d_SSIM.copy_to_host()
    SSIM = np.mean(SSIM, axis=1)
    print(SSIM.shape)
    print(SSIM)

    plt.clf()
    plt.ylabel("Row")
    plt.ylim(0,1.5)
    plt.plot(SSIM)
    plt.savefig('pixel_impact.png')

    cv2.imwrite("out.png", SSIM)
