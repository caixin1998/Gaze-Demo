import cv2 as cv
import numpy as np
import os 
import sys
# sys.path.append("src")
sys.path.append("../src")
sys.path.append("..")
import pickle
import h5py
from monitor import monitor
from face import Face
from landmarks import landmarks
from undistorter import Undistorter
import multiprocessing
landmarks_detector = landmarks()
face = Face()
# videos_path = "/home/caixin/nas_data/VIPLIIGaze/calibration/3"
# img_path = "/home/caixin/nas_data/data/VIPLIIGaze/images"
# output_path = "/home/caixin/nas_data/data/VIPLIIGaze/supps"
img_path = "/home1/caixin/GazeData/VIPLIIGaze538/images"
intrinsic_path = "/home1/caixin/Gaze-Demo/intrinsic/538"
external_path = "/home1/caixin/Gaze-Demo/external/538"
output_path = "/home1/caixin/GazeData/VIPLGaze538/supps"
os.makedirs(output_path, exist_ok = True)
# videodata = skvideo.io.vread(os.path.join(video_path, "cam0.mp4"))

persons = os.listdir(img_path)
persons.sort()

out_dict = {}
for person in persons:
    out_dict[person] = {}

def add_key(person, key, value):
    to_write = out_dict[person]
    if key not in to_write:
        to_write[key] = [value]
    else:
        to_write[key].append(value)
undistorter = Undistorter(cam_calib['mtx'], cam_calib['dist'])
def write_preh5(person):
    if os.path.isfile(os.path.join(output_path, person + ".h5")):
        continue
    cams = os.listdir(os.path.join(img_path,person))
    for cam in cams:
        if cam == "color":
            cam_id = "cam0"
        else:
            cam_id = "cam" + str(int(cam[-1]) + 3)
            
        if cam_id != "cam1":
            continue
        intrinsic = os.path.join(intrinsic_path, "%s.pkl"%cam_id)
        if os.path.exists(intrinsic):
            cam_calib = pickle.load(open(intrinsic, "rb"))
        camera_parameters = cam_calib["mtx"]
        distortion_parameters = cam_calib["dist"]
        undistorter = Undistorter(cam_calib['mtx'], cam_calib['dist'])
        ext = os.path.join(external_path, cam_id,  "opt.txt")
        
        videos = os.listdir(os.path.join(img_path,person,cam))
        if len(videos) != 18:
            print("len(videos) != 18", os.path.join(img_path,person,cam))
            
        for video in videos:
            imgs_path = os.path.join(img_path, person , cam, video)
            print(imgs_path)
            labels = np.loadtxt(os.path.join(imgs_path, "labels.txt"))
            # position = np.loadtxt(os.path.join(imgs_path, "position.txt"))
            if labels.shape[1] == 2:
                mon = monitor(ext)

            person = video.split("_")[0]
            imgs = os.listdir(imgs_path)
            if len(imgs) != 692:
                print("len(imgs) != 692", imgs_path)
                continue
            for i, img in enumerate(imgs):
                if not img.endswith("png"):
                    break
                file_name = os.path.join(person, cam, video, img)
                if labels.shape[1] == 2:
                    label = mon.monitor_to_camera(labels[i][0], labels[i][1])  
                else:
                    label = labels[i]  
                print(file_name)
                frame = cv.imread(os.path.join(img_path, file_name))
                frame = undistorter.apply(frame)
                face_location = face.detect(frame,scale=0.25, use_max='SIZE')
                if len(face_location) == 4:
                    add_key(person, "face_valid", 1) 
                    pts = landmarks_detector.detect(face_location, frame)
                else:
                    add_key(person, "face_valid", 0)
                    pts = np.zeros((68,2))
                add_key(person, "landmarks", pts)
                add_key(person, "3d_gaze_target", label)
                add_key(person, "file_name", file_name)
                # use video's name as position
                add_key(person, "position", int(video))
                add_key(person, "camera_parameters", camera_parameters)
                add_key(person, "distortion_parameters", distortion_parameters)
            
        

        
    with h5py.File(os.path.join(output_path, person + ".h5"), "w") as f:     
        for key, values in out_dict[person].items():
            values = np.array(values)
            print(key, values.shape)
            if key == "file_name":
                ds = f.create_dataset(
                    key, values.shape,
                dtype = h5py.special_dtype(vlen=str)
                )
                ds[:] = values
            else:
                f.create_dataset(
                    key, data=values,
                    chunks=(
                        tuple([1] + list(values.shape[1:]))
                        if isinstance(values, np.ndarray)
                        else None
                    ),
                    compression='lzf',
                )

                # print(key, np.array(out_dict[person][key]).shape)
            



if __name__ == "__main__":
    num_processes = 8
    process_pool = multiprocessing.Pool(processes=num_processes)
    result_list = process_pool.map(write_preh5, persons)
    # for video in videos:
    #     cut_video(video)
    process_pool.close()
    process_pool.join()