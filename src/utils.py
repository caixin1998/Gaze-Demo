


import numpy as np
import cv2 as cv
import torch
def get_rect(points , ratio = 1.0): # ratio = w:h
    x = points[:,0]
    y = points[:,1]

    x_expand = 0.1*(max(x)-min(x))
    y_expand = 0.1*(max(y)-min(y))
    
    
    x_max, x_min = max(x)+x_expand, min(x)-x_expand
    y_max, y_min = max(y)+y_expand, min(y)-y_expand

    #h:w=1:2
    if (y_max-y_min)*ratio < (x_max-x_min):
        h = (x_max-x_min)/ratio
        pad = (h-(y_max-y_min))/2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max-y_min)
        pad = (h*ratio-(x_max-x_min))/2
        x_max += pad
        x_min -= pad
    return int(x_min),int(x_max),int(y_min),int(y_max)

def R_x(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    return np.array([
        [1., 0., 0.],
        [0., cos_, -sin_],
        [0., sin_, cos_]
     ]).astype(np.float32)

def R_y(phi):
    sin_ = np.sin(phi)
    cos_ = np.cos(phi)
    return np.array([
        [cos_, 0., sin_],
        [0., 1., 0.],
        [-sin_, 0., cos_]
    ]).astype(np.float32)

def calculate_rotation_matrix(e):
     return np.matmul(R_y(e[1]), R_x(e[0]))


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in.copy()
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)

    return image_out

def cropImage(img, bbox):
    bbox = np.array(bbox, int)
    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))
    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)
    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)    
    res[aDst[1]:bDst[1],aDst[0]:bDst[0],:] = img[aSrc[1]:bSrc[1],aSrc[0]:bSrc[0],:]
    return res

def preprocess_image(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
    image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
    # cv.imshow('processed patch', ima
    image = np.transpose(image, [2, 0, 1])  # CxHxW
    image = 2.0 * image / 255.0 - 1
    return image[None]

def pitchyaw_to_vector(pitchyaw):
    vector = np.zeros((3, 1))
    vector[0, 0] = np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
    vector[1, 0] = np.sin(pitchyaw[0])
    vector[2, 0] = np.cos(pitchyaw[0]) * np.cos(pitchyaw[1])
    return vector

def apply_transformation(T, vec):
    if vec.shape[0] == 2:
        vec = pitchyaw_to_vector(vec)
    h_vec = np.array([*vec,1]).reshape(4, 1)
    return np.matmul(T, h_vec)[:3, 0]

def apply_rotation(T, vec):
    if vec.shape[0] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(3, 1)
    R = T[:3, :3]
    return np.matmul(R, vec).reshape(3, 1)

def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    d = - o[2] / g[2] 
    x = o[0] + d * g[0]
    y = o[1] + d * g[1] 
    # print(d,x,y,g)
    #[x,y,0] - o = k * g 
    return np.array([x,y])

def to_screen_coordinates(origin, direction, rotation, monitor):
    inv_rotation = np.transpose(rotation)
    direction = pitchyaw_to_vector(direction)
    direction = np.matmul(inv_rotation, direction)
    direction = apply_rotation(monitor.inv_camera_transformation, direction)
    origin = apply_transformation(monitor.inv_camera_transformation, origin)
    recovered_target_2D = get_intersect_with_zero(origin, direction)
    # print(origin, direction,recovered_target_2D)

    PoG_mm = recovered_target_2D
    ppm_w = monitor.pixels_per_millimeter[0]
    ppm_h = monitor.pixels_per_millimeter[1]
    PoG_px = np.array([recovered_target_2D[0] * ppm_w, recovered_target_2D[1] * ppm_h])
    return PoG_mm, PoG_px
# def preprocess_image(image):
#     ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
#     ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
#     image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
#     # cv.imshow('processed patch', ima
#     image = np.transpose(image, [2, 0, 1])  # CxHxW
#     image = 2.0 * image / 255.0 - 1
#     return image

def polyfit(src_x, src_y, order = 1):
    assert src_x.shape == src_y.shape and src_x.shape[0] > 0 and src_x.shape[1] > 0 and order >= 1

    bias = np.ones((src_x.shape[0], 1), dtype=np.float32)
    input_x = np.zeros((src_x.shape[0], order*src_x.shape[1]), dtype=np.float32)

    for i in range(1, order+1):
        copy = np.power(src_x, i)
        input_x[:, (i-1)*src_x.shape[1]:i*src_x.shape[1]] = copy

    new_mat = np.concatenate((input_x, bias), axis=1)
    matrix = cv.solve(new_mat, src_y, flags=cv.DECOMP_NORMAL)
    return matrix[1]

def polymat(src_x, matrix, order = 1):
    bias = np.ones((src_x.shape[0], 1), dtype=np.float32)
    input_x = np.zeros((src_x.shape[0], order*src_x.shape[1]), dtype=np.float32)
    for i in range(1, order+1):
        copy = np.power(src_x, i)
        input_x[:, (i-1)*src_x.shape[1]:i*src_x.shape[1]] = copy
    new_mat = np.concatenate((input_x, bias), axis=1)
    calibrated = new_mat @ matrix
    
    return calibrated

def polymat4tensor(src_x, matrix, order=1):
    bias = torch.ones(src_x.shape[0], 1)
    input_x = torch.zeros(src_x.shape[0], order*src_x.shape[1])
    for i in range(1, order+1):
        copy = torch.pow(src_x, i)
        input_x[:, (i-1)*src_x.shape[1]:i*src_x.shape[1]] = copy
    new_mat = torch.cat((input_x, bias), dim=1)
    calibrated = torch.mm(new_mat, matrix)
    return calibrated