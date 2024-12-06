import numpy as np
import os
import json
import cv2
from scipy.ndimage import map_coordinates

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def process_frame(room_name:str,
                  frame_id:int,
                  pixels_yx: np.array, #pixels with shape N, 2, [y(row), x(column)]
                  replica_root_dir:str = '/mnt/sdb1/home/kbotashev/iros_paper/replica_dataset/'):

    replica_room_source = os.path.join(replica_root_dir, 'scenes', room_name, room_name + '_base', room_name + '_base_raw')

    with open(os.path.join(replica_room_source, 'cam_params.json')) as fp:
        camera_params = json.load(fp)['camera']

    with open(os.path.join(replica_room_source, 'transforms.json')) as fp:
        transforms = json.load(fp)    

    K = np.array([[camera_params['fx'], 0, camera_params['cx']],
                [0, camera_params['fy'], camera_params['cy']],
                [0, 0, 1]])
    
    frame_info = transforms['frames'][frame_id]

    image_path = frame_info['file_path']
    depth_path = frame_info['depth_path']

    image_num_str = image_path.split('image.')[-1].split('.png')[0]
    assert int(image_num_str) == frame_id, "Frame id int is not the same as img_name"

    image_orig = cv2.cvtColor(cv2.imread(os.path.join(replica_room_source, image_path)), cv2.COLOR_BGR2RGB)/255
    depth = cv2.imread(os.path.join(replica_room_source, depth_path), cv2.IMREAD_ANYDEPTH)/camera_params['scale']

    cam2world = np.array(frame_info['transform_matrix']) @ np.diag([1,-1,-1,1])
    world2cam = np.linalg.inv(cam2world)

    tvec = world2cam[:-1,-1]
    rot = world2cam[:3,:3]
    qvec = rotmat2qvec(rot)

    depth_interp = map_coordinates(depth, pixels_yx.T)

    pixels_yx = pixels_yx.reshape(-1,2)

    pixs = np.concatenate([pixels_yx, np.expand_dims(np.ones_like(pixels_yx[:,0]),1)], -1)

    Kinv_P = np.einsum('ij,cj->ci', np.linalg.inv(K), pixs[:,[1,0,2]]) 
    D_Kinv_P = np.concatenate([np.multiply(np.expand_dims(depth_interp,-1), Kinv_P), np.expand_dims(np.ones_like(depth_interp),-1)], -1).squeeze()
    pcd = np.einsum('ij,cj->ci', cam2world, D_Kinv_P)[:,:-1]
    colors = image_orig[tuple((pixels_yx.T).astype(int))]

    return qvec, tvec, pcd, colors