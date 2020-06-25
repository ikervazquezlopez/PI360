import sys
import os
from os import listdir
from os.path import isfile, join, isdir
import multiprocessing as mp
from multiprocessing import set_start_method, get_context



def ROAD(filename, in_dir, out_dir):
    os.system("python ROAD_impact.py {f} {in_dir} {out_dir}".format(f=filename, in_dir=in_dir, out_dir=out_dir))


if __name__ == '__main__':
    set_start_method("spawn")
    mp.freeze_support()

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    with get_context("spawn").Pool(mp.cpu_count()-2) as pool:

        onlyfilespaths = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

        pool.starmap(ROAD, [(f, in_dir, out_dir) for f in onlyfilespaths])

        pool.close()
        pool.join()
