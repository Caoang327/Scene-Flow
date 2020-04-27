## util function 

import numpy as np
import imageio
from collections import OrderedDict
import cv2
import os

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def load_calib_rigid(filepath):
    data = read_calib_file(filepath)
    return transform_from_rot_trans(data['R'], data['T'])

def load_calib_cam_to_cam(velo_to_cam_file, cam_to_cam_file):

    data = {}

    # Load the rigid transformation from velodyne coordinates
    # to unrectified cam0 coordinates

    T_cam0unrect_velo = load_calib_rigid(velo_to_cam_file)
    data['T_cam0_velo_unrect'] = T_cam0unrect_velo

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(cam_to_cam_file)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
    P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
    P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
    P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Create 4x4 matrices from the rectifying rotation matrices
    R_rect_00 = np.eye(4)
    R_rect_00[0:3, 0:3] = np.reshape(filedata['R_rect_00'], (3, 3))
    R_rect_10 = np.eye(4)
    R_rect_10[0:3, 0:3] = np.reshape(filedata['R_rect_01'], (3, 3))
    R_rect_20 = np.eye(4)
    R_rect_20[0:3, 0:3] = np.reshape(filedata['R_rect_02'], (3, 3))
    R_rect_30 = np.eye(4)
    R_rect_30[0:3, 0:3] = np.reshape(filedata['R_rect_03'], (3, 3))

    data['R_rect_00'] = R_rect_00
    data['R_rect_10'] = R_rect_10
    data['R_rect_20'] = R_rect_20
    data['R_rect_30'] = R_rect_30

    # Compute the rectified extrinsic from cam0 to camN
    T0 = np.eye(4)
    T0[0, 3] = P_rect_00[0, 3] / P_rect_00[0, 0]
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = T0.dot(R_rect_00.dot(T_cam0unrect_velo))
    data['T_cam1_velo'] = T1.dot(R_rect_00.dot(T_cam0unrect_velo))
    data['T_cam2_velo'] = T2.dot(R_rect_00.dot(T_cam0unrect_velo))
    data['T_cam3_velo'] = T3.dot(R_rect_00.dot(T_cam0unrect_velo))

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3] / 1000
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    data['f'] = data['P_rect_30'][0,0] / 1000

    pik00 = P_rect_00.dot(R_rect_00.dot(data['T_cam0_velo']))
    pik10 = P_rect_10.dot(R_rect_10.dot(data['T_cam1_velo']))
    pik20 = P_rect_20.dot(R_rect_20.dot(data['T_cam2_velo']))
    pik30 = P_rect_30.dot(R_rect_30.dot(data['T_cam3_velo']))

    data['Pik00'] = pik00
    data['Pik10'] = pik10
    data['Pik20'] = pik20
    data['Pik30'] = pik30

    return data


def make_colorwheel():
    '''
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
    '''
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param u: np.ndarray, input horizontal flow
    :param v: np.ndarray, input vertical flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    '''

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi

    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1

        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range?

        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    '''
    Expects a two dimensional flow image of shape [H,W,2]
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: np.ndarray of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :return:
    '''

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_compute_color(u, v, convert_to_bgr)

def get_gt_kitti(instance_img_path):
    """ get the goundtruth from the kitti instance image file
    Args:
    --instance_img_path: the ground truth instance image file path
    
    Return:
    
    --gt: dict, with 'masks': NxHxW, 'semantic id', 'instance id'
    """
    instance_img = imageio.imread(instance_img_path)
    
    kitti_semantic = instance_img // 256 ## semantic id
    kitti_instance = instance_img % 256  ## instance id 
    
    instance_map = kitti_instance > 0.0

    masks = []
    semantic_ids = [] 
    instance_ids = []
    for semantic_id in np.unique(instance_map * kitti_semantic):
        if semantic_id == 0:
            continue
        semantic_ids.append(semantic_id)
        semantic_mask = (kitti_semantic == semantic_id)
        for instance_id in np.unique(kitti_instance * semantic_mask):
            if instance_id == 0:
                continue
            instance_ids.append(instance_id)
            instance_mask = (kitti_instance*semantic_mask) == instance_id
            masks.append(np.expand_dims(instance_mask,0))
    masks = np.concatenate(masks,0)
    semantic_ids = np.array(semantic_ids)
    instance_ids = np.array(instance_ids)
    gt = OrderedDict()
    gt.update({"masks":masks, "semantic_id":semantic_ids, "instance_ids":instance_ids})
    return gt

def _get_gt_kitti_flow(path):
    # img_name = "/content/training/flow_occ/000000_10.png"
    flow_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float)
    # 20: the network was pre-trained with this factor (magic number:)
    # data = {}
    flow = (flow_raw[:, :, :2] - 2**15) / 64.0
    mask = flow_raw[:, :, 2].astype(bool)
    return flow, mask

def _get_gt_kitti_disparity_single_file(path):
    # Disparity
    # first_img_name = "/content/training/disp_occ_0/000000_10.png"
    # second_img_name = "/content/training/disp_occ_1/000000_10.png"

    disp_first = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float) / 256.0
    # disp_second = cv2.imread(second_img_name, cv2.IMREAD_UNCHANGED).astype(np.float) / 256.0
    # np.savez("disparity.npz", first=disp_first, second=disp_second)
    return disp_first

def get_gt(category, idx, root_path="./", mode="occ"):
    """Helper function to obtain ground truth
    
    Parameters
    ----------
    category - str
        choose from ["disparity", "flow"]
    idx - int
        index of the image whose ground truth will be retrived, limit 0 <= idx < 200
    root_path - str
        the path to the folder `training`, the directory of training is as follows:        
            training
            ├── disp_noc_0
            ├── disp_noc_1
            ├── disp_occ_0
            ├── disp_occ_1
            ├── flow_noc
            ├── flow_occ
            ├── image_2
            ├── image_3
            ├── obj_map
            ├── viz_flow_occ
            └── viz_flow_occ_dilate_1      

    mode - str
        (optional) choose from ["occ" (default), "ncc"] 

    Returns
    -------
    data - dict<string, numpy.ndarray>
        if category = "flow", keys = {"flow", "mask"}
        if category = "disparity", keys = {"first", "second"}
    """
    
    # validation        
    if idx < 0 or idx >= 200:
        raise ValueError("idx out of range: [0, 200)")

    data = {}

    if category is "disparity":
        first_path = os.path.join(root_path, "disp_{}_0".format(mode))
        second_path = os.path.join(root_path, "disp_{}_1".format(mode))
        file_name = "{:06}_10.png".format(idx)
        first_path = os.path.join(first_path, file_name)
        second_path = os.path.join(second_path, file_name)
        if not os.path.exists(first_path):
            raise FileNotFoundError("Not found: {}".format(first_path))
        if not os.path.exists(second_path):
            raise FileNotFoundError("Not found: {}".format(second_path))
             
        first = _get_gt_kitti_disparity_single_file(first_path)
        second = _get_gt_kitti_disparity_single_file(second_path)
        data = {"first": first, "second": second}
    elif category is "flow":
        directory = os.path.join(root_path, "flow_{}".format(mode))
        file_name = "{:06}_10.png".format(idx)
        directory = os.path.join(directory, file_name)
        if not os.path.exists(directory):
            raise FileNotFoundError("Not found: {}".format(directory))

        flow, mask = _get_gt_kitti_flow(directory)
        data['flow'] = flow
        data['mask'] = mask

    else:
        raise ValueError("Unknown category {}, choose from ['disparity', 'flow;]".format(category))    
    
    return data
