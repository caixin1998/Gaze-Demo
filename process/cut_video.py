import cv2 as cv
import numpy as np
import os 
from shutil import copyfile
import multiprocessing
id = 3
cam_idx = 6
videos_path = "/home1/caixin/GazeData/VIPLGaze538/origin"
output_path = "/home1/caixin/GazeData/VIPLGaze538/images"
# videodata = skvideo.io.vread(os.path.join(video_path, "cam0.mp4"))
cut_num = 15



def cut_video(video):
    person = video.split("+")[0]
    video_path = os.path.join(videos_path, video)
    position = video.split("_")[1]
    try:
        labels = np.loadtxt(os.path.join(video_path, "calib_target.txt"))
        times = np.loadtxt(os.path.join(video_path, "times.txt"))
    except:
        return
    print(video_path)
    assert len(labels) == len(times)
    # save_labels = np.repeat(labels, cut_num, axis = 0)
    for cam in os.listdir(video_path):
        save_labels = []
        
        if not cam.endswith(".mp4"):
            continue
        cam_name = cam.split(".")[0]
        if cam_name == "color":
            time0 = np.loadtxt(os.path.join(video_path, "time_realsense.txt"))
        elif cam_name[:3] == "cam":
            time0 = np.loadtxt(os.path.join(video_path, "time%c.txt"%cam_name[3]))
        else:
            raise ValueError("Invalid value!")
        
        if os.path.exists(os.path.join(output_path, person, cam_name, position)):
            continue
        os.makedirs(os.path.join(output_path, person, cam_name, position), exist_ok = True)
    
        frame_idxs = []
        for i,time in enumerate(times):
            idx = np.argmin(abs(time0 - time))
            if idx >= cut_num:
                frame_idxs += list(range(idx - cut_num + 1, idx + 1))  
                save_labels += [np.repeat(labels[i][None], cut_num, axis = 0)]
            else:
                frame_idxs += list(range(0, idx + 1))  
                save_labels += [np.repeat(labels[i][None], idx + 1, axis = 0)]
        if len(frame_idxs) == 0:
            continue    
        cap = cv.VideoCapture(os.path.join(video_path, cam))
        i = 0
        idx = 0
        ret, frame = cap.read()
        num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if num_frames != len(time0):
            print(num_frames, len(time0), os.path.join(video_path, cam))
                    
        while ret and i < len(frame_idxs):
            if idx == frame_idxs[i]:
                i = i + 1
                if not os.path.exists("%s/%s/%s/%s/%05d.png"%(output_path, person, cam_name, position, i)):
                    cv.imwrite("%s/%s/%s/%s/%05d.png"%(output_path, person, cam_name, position, i), frame)
                # print("%s/%s/%s/%s/%05d.png"%(output_path, person, cam_name, position, i))
                # print("%s/%s/%05d.png"%(output_path,video, i))
            ret, frame = cap.read()
            idx += 1
        
        save_labels = np.concatenate(save_labels, axis = 0)    
        # print(ret, i, len(frame_idxs), len(save_labels))
        assert i == len(save_labels)
        # print(os.path.join(output_path, person, cam_name, position))
        cap.release()

        np.savetxt(os.path.join(output_path, person, cam_name, position, "labels.txt"), save_labels)
        copyfile(os.path.join(videos_path, video, "position.txt"), os.path.join(output_path, person, cam_name, position,"position.txt"))
        # copyfile("/home/caixin/tnm-opencv/data/%s/cam%s/opt.txt"%(id, cam_idx), os.path.join(output_path, person, video, video,"ext.txt"))

if __name__ == "__main__":
    videos = os.listdir(videos_path)
    videos.sort()
    num_processes = 8
    process_pool = multiprocessing.Pool(processes=num_processes)
    result_list = process_pool.map(cut_video, videos)
    # for video in videos:
    #     cut_video(video)
    process_pool.close()
    process_pool.join()