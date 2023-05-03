# copy file from nas_path to data_path, if the folder is exist, then skip, skip all h5 files
import os
import shutil
import time
import glob
nas_path = "/home/caixin/nas/data/VIPLGaze538/origin"
data_path = "/home1/caixin/GazeData/VIPLGaze538/origin"
for person in os.listdir(nas_path):
    if os.path.exists(os.path.join(data_path, person)):
        continue
    else:
        shutil.copytree(os.path.join(nas_path, person), os.path.join(data_path, person), ignore=shutil.ignore_patterns('*.h5',".*"))
        print("copy %s to %s"%(person, data_path))
        time.sleep(1)
