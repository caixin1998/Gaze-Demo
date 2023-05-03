#read depth and rgb from depth folder
import h5py
import cv2 as cv
import numpy as np

depth_scale = 0.0010000000474974513

clipping_distance_in_meters = 0.7 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
        
def read_depth_rgb(depth_file_name, rgb_file_name, idx):
    with h5py.File(depth_file_name, 'r') as depth_file:
        depth = depth_file["depth"]
        depth_image = np.array(depth)
  
        depth_image_clip = np.where((depth_image > clipping_distance),clipping_distance, depth_image)
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image_clip, alpha=0.2), cv.COLORMAP_JET)

    rgb = cv.imread(rgb_file_name)
    # resize depth and rgb to the same size
    depth_colormap = cv.resize(depth_colormap, (rgb.shape[1], rgb.shape[0]))
    #concatenate depth and rgb
    depth_rgb = np.concatenate((rgb, depth_colormap), axis=1)
    #save depth_rgb
    cv.imwrite('outputs/%s_depth_rgb.png'%idx, depth_rgb)

if __name__ == '__main__':
    depth_file_name = 'frames/lmy_7_15.h5'
    rgb_file_name = 'frames/lmy_7_15.png'
    read_depth_rgb(depth_file_name, rgb_file_name, 0)