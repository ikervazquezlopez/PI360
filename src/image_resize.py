import sys
import os
from os import listdir
from os.path import isfile, join, isdir

import cv2

import multiprocessing as mp
from multiprocessing import set_start_method, get_context


divisor_x = 0.5
divisor_y = 0.5


def resize_image(f, in_dir, out_dir):
    img = cv2.imread(join(in_dir,f))
    img = cv2.resize(img, None, fx=divisor_x, fy=divisor_y, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(join(out_dir, f), img)

if __name__ == '__main__':
    set_start_method("spawn")
    mp.freeze_support()

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]


    with get_context("spawn").Pool(mp.cpu_count()-2) as pool:

        onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

        pool.starmap(resize_image, [(f, in_dir, out_dir) for f in onlyfilespaths])

        pool.close()
        pool.join()
