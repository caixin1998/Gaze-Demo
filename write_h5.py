import sys
sys.path.append("src")
from normalization import normalize
from utils import draw_gaze
import torch
import cv2 as cv
import numpy as np 
import os 
import pickle
import h5py
import copy
from undistorter import Undistorter
from landmarks import landmarks
from head import HeadPoseEstimator
import multiprocessing
data_path = "/home1/caixin/GazeData/VIPLGaze538/supps"
dataset_path = "/home1/caixin/GazeData/VIPLGaze538/images"
out_path = "/home1/caixin/GazeData/VIPLGaze538/data"
visualize_path = "/home1/caixin/GazeData/VIPLGaze538/visual"
os.makedirs(out_path, exist_ok = True)
def add_key(to_write, key, value):
    if key not in to_write:
        to_write[key] = [value]
    else:
        to_write[key].append(value)
        
#sample rand_num of data from every cut_num data in range(total_num)
def ramdom_sample(data_num, cut_num = 15, total_num = 690):
    sampled_list = []
    assert data_num <= cut_num
    assert total_num % cut_num == 0

    for i in range(0,total_num,cut_num):
        sampled_list += list(np.random.choice(list(range(i, i+cut_num)), data_num, replace = False))
    return sampled_list

persons = os.listdir(dataset_path)
persons.sort()

head_pose_estimator = HeadPoseEstimator()
landmarks_detector = landmarks()
def write_h5(person):
    if os.path.isfile(os.path.join(visualize_path, person + ".h5")):
        return
    if not os.path.isfile(os.path.join(data_path, person + ".h5")):
        return
    os.makedirs(os.path.join(visualize_path, person), exist_ok = True)
    total_index = 0
    with h5py.File(os.path.join(data_path, person + ".h5"),"r") as f:
        with h5py.File(os.path.join(out_path, person + ".h5"),"w") as g:
            for cam_id, value in f.items(): # key is camera_idx
                cam = g.create_group(cam_id)
                for sub_key, sub_value in value.items(): # sub_key is video_idx / position_idx
                    video = cam.create_group(sub_key)
                    num_entries = next(iter(sub_value.values())).shape[0]
                    # img_list = ramdom_sample(2, total_num = num_entries)
                    to_write = {}
                    # print(sub_value['file_name'][:])
                    # print(sub_value['3d_gaze_target'][:])

                    for i in range(num_entries):
                        img_path = '%s/%s' % (dataset_path,
                                sub_value['file_name'][i])
                        camera_parameters = sub_value['camera_parameters'][i]
                        distortion_parameters = sub_value['distortion_parameters'][i]
                        img = cv.imread(img_path, cv.IMREAD_COLOR)
                        undistorter = Undistorter(camera_parameters, distortion_parameters)
                        img = undistorter.apply(img)
                        entry = {}
                        entry["full_frame"] = img
                        entry["camera_parameters"] = camera_parameters
                        rvec, tvec, o_3d = head_pose_estimator(img, sub_value["landmarks"][i], camera_parameters)
                        head_pose = (rvec, tvec)
                        entry["head_pose"] = head_pose
                        entry["o_3d"] = o_3d
                        entry["3d_gaze_target"] = sub_value["3d_gaze_target"][i]
                        normalized_entry = normalize(entry, patch_type = "face")
                        add_key(to_write, "face_patch", normalized_entry["patch"].astype(np.uint8))
                        add_key(to_write, "face_gaze", normalized_entry["normalized_gaze"])
                        add_key(to_write, "face_pose", normalized_entry["normalized_pose"])
                        add_key(to_write, "gaze_origin", normalized_entry["gaze_cam_origin"])
                        add_key(to_write, "gaze_target", normalized_entry["gaze_cam_target"])
                       
                        face = copy.deepcopy(normalized_entry["patch"])
                        nor_pts = landmarks_detector.detect([0,0,224,224], cv.cvtColor(face, cv.COLOR_RGB2BGR))
                        add_key(to_write, "nor_pts", nor_pts)
                        total_index = total_index + 1

                        if total_index % 100 == 0:
                # print(nor_pts.shape)
                            face = landmarks_detector.plot_markers(face, nor_pts.astype(int))
                            vis_img = draw_gaze(face, normalized_entry["normalized_gaze"] ,color = (255,0,100))
                            cv.imwrite(os.path.join(visualize_path, person, '%s_%s_%s'%(cam_id,sub_key,os.path.basename(sub_value["file_name"][i]))), cv.cvtColor(vis_img, cv.COLOR_RGB2BGR))

                    for key, values in to_write.items():
                        values = np.array(values)
                        video.create_dataset(
                            key, data=values,
                            chunks=(
                                tuple([1] + list(values.shape[1:]))
                                if isinstance(values, np.ndarray)
                                else None
                            ),
                            compression='lzf',
                        )
    # exit(0)
if __name__ == "__main__":
    num_processes = 4
    multiprocessing.set_start_method('spawn')
    process_pool = multiprocessing.Pool(processes=num_processes)
    result_list = process_pool.map(write_h5, persons)
    # for person in persons:
    #     write_h5(person)

    process_pool.close()
    process_pool.join()