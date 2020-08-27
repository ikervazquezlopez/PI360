import sys
import os
from os import listdir
from os.path import isfile, join, isdir
import multiprocessing as mp
from multiprocessing import set_start_method, get_context



def SSIM(filename, in_dir, out_dir):
    os.system("python SSIM_impact.py {f} {in_dir} {out_dir}".format(f=filename, in_dir=in_dir, out_dir=out_dir))


if __name__ == '__main__':
    set_start_method("spawn")
    mp.freeze_support()

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    with get_context("spawn").Pool(15) as pool:

        onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

        pool.starmap(SSIM, [(f, in_dir, out_dir) for f in onlyfilespaths])

        pool.close()
        pool.join()
