import sys
import os
from os import listdir
from os.path import isfile, join, isdir

import cv2


divisor_x = 0.33
divisor_y = 0.33

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

    for f in onlyfilespaths:
        img = cv2.imread(join(in_dir,f))
        img = cv2.resize(img, None, fx=divisor_x, fy=divisor_y, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(join(out_dir, f), img)
