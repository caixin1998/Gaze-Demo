import cv2 as cv
import numpy as np
import os 
import sys
sys.path.append("src")
# sys.path.append("../src")
# sys.path.append("..")
import pickle
import h5py
from monitor import monitor
from face import Face
from landmarks import landmarks
from undistorter import Undistorter
import multiprocessing
from tqdm import tqdm 
landmarks_detector = landmarks()
face = Face()
# videos_path = "/home/caixin/nas_data/VIPLIIGaze/calibration/3"
# img_path = "/home/caixin/nas_data/data/VIPLIIGaze/images"
# output_path = "/home/caixin/nas_data/data/VIPLIIGaze/supps"
img_path = "/home1/caixin/GazeData/VIPLGaze538/images"
intrinsic_path = "/home1/caixin/Gaze-Demo/intrinsic/538"
external_path = "/home1/caixin/Gaze-Demo/external/538"
output_path = "/home1/caixin/GazeData/VIPLGaze538/supps"
os.makedirs(output_path, exist_ok = True)
# videodata = skvideo.io.vread(os.path.join(video_path, "cam0.mp4"))
listdirs = lambda x: [f for f in os.listdir(x) if not f.endswith('DS_Store')] # type: ignore
listimgs = lambda x: [f for f in os.listdir(x) if f.endswith('png')]

persons = listdirs(img_path)
persons.sort()

def add_key(to_write, key, value):
    if key not in to_write:
        to_write[key] = [value]
    else:
        to_write[key].append(value)

def ramdom_sample(data_num, cut_num = 15, total_num = 690):
    sampled_list = []
    assert data_num <= cut_num
    assert total_num % cut_num == 0

    for i in range(0,total_num,cut_num):
        sampled_list += list(np.random.choice(list(range(i, i+cut_num)), data_num, replace = False))
    return sampled_list

def write_preh5(person):
    # if os.path.isfile(os.path.join(output_path, person + ".h5")):
    #     os.remove(os.path.join(output_path, person + ".h5"))
    if os.path.isfile(os.path.join(output_path, person + ".h5")):
        return
    with h5py.File(os.path.join(output_path, person + ".h5"), "w") as f:     
        cams = listdirs(os.path.join(img_path,person))
        for cam in cams:
            if cam != "cam0":
                continue
            # print(cam)
            if cam == "color":
                cam_id = "cam0"
            else:
                cam_id = "cam" + str(int(cam[-1]) + 3)
                
            cam_group = f.create_group(cam)
            intrinsic = os.path.join(intrinsic_path, "calib_%s.pkl"%cam_id)
            # if os.path.exists(intrinsic):
            cam_calib = pickle.load(open(intrinsic, "rb"))
            undistorter = Undistorter(cam_calib['mtx'], cam_calib['dist'])
            camera_parameters = cam_calib["mtx"]
            distortion_parameters = cam_calib["dist"]
            undistorter = Undistorter(cam_calib['mtx'], cam_calib['dist'])
            ext = os.path.join(external_path, cam_id,  "opt.txt")
            
            videos = listdirs(os.path.join(img_path,person,cam))
            if len(videos) != 18:
                print("len(videos) != 18", os.path.join(img_path,person,cam))
                
            for video in videos:
                imgs_path = os.path.join(img_path, person, cam, video)
                # print(imgs_path)
                try:
                    labels = np.loadtxt(os.path.join(imgs_path, "labels.txt"))
                except:
                    print("labels.txt not exist", imgs_path)
                    continue    
                # position = np.loadtxt(os.path.join(imgs_path, "position.txt"))
                imgs = listimgs(imgs_path)
                imgs.sort()
                mon = monitor(ext)
                if len(imgs) != 690:
                    print("len(imgs) != 690", imgs_path)
                    continue
                video_group = cam_group.create_group(video)
                out_dict = {}
                ramdom_sample_list = ramdom_sample(2)
                for i in ramdom_sample_list:
                    img = imgs[i]
                    file_name = os.path.join(person, cam, video, img)
                    # print(file_name)
                    if labels.shape[1] == 2:
                        label = mon.monitor_to_camera(labels[i][0], labels[i][1])  
                    else:
                        label = labels[i]  
                    # print(file_name)
                    frame = cv.imread(os.path.join(img_path, file_name))
                    frame = undistorter.apply(frame)
                    cv.imwrite(os.path.join(img_path, file_name), frame)
                    face_location = face.detect(frame,scale=0.25, use_max='SIZE')
                    if len(face_location) == 4:
                        add_key(out_dict, "face_valid", 1) 
                        pts = landmarks_detector.detect(face_location, frame)
                    else:
                        add_key(out_dict, "face_valid", 0)
                        pts = np.zeros((68,2))
                    add_key(out_dict, "landmarks", pts)
                    add_key(out_dict, "3d_gaze_target", label)
                    add_key(out_dict, "file_name", file_name)
                    # use video's name as position
                    add_key(out_dict, "position", int(video))
                    add_key(out_dict, "camera_parameters", camera_parameters)
                    add_key(out_dict, "distortion_parameters", distortion_parameters)
                
        
                for key, values in out_dict.items():
                    values = np.array(values)
                    # print(key, values.shape)
                    if key == "file_name":
                        ds = video_group.create_dataset(
                            key, values.shape,
                        dtype = h5py.special_dtype(vlen=str)
                        )
                        ds[:] = values
                    else:
                        video_group.create_dataset(
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
    num_processes = 4
    multiprocessing.set_start_method('spawn')
    process_pool = multiprocessing.Pool(processes=num_processes)
    result_list = process_pool.map(write_preh5, persons)
    # for person in tqdm(persons):
    #     write_preh5(person)

    process_pool.close()
    process_pool.join()