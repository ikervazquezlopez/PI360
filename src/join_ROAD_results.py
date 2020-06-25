import sys
import os
import cv2
from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import numpy as np



if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_file = sys.argv[2]

    onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    tmp_img = cv2.imread(join(in_dir, onlyfilespaths[0]))
    tmp_img = cv2.split(tmp_img)[0].astype(np.float64) / 255

    for f in tqdm(onlyfilespaths[1:]):
        new_img = cv2.imread(join(in_dir, f))
        new_img = cv2.split(new_img)[0].astype(np.float64) / 255
        tmp_img = tmp_img + new_img


    out_img = tmp_img / len(onlyfilespaths)
    out_img = (out_img*255).astype(np.uint8)
    cv2.imwrite(out_file, out_img)
