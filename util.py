## util function 

import numpy as np
import imageio
from collections import OrderedDict
import cv2
import os
import scipy
import scipy.ndimage

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
    data['K_cam2'] = P_rect_20[0:3, 0:3]
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

    data['f'] = data['P_rect_30'][0,0]

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

def get_mask_gt(masks,alpha_p):
    mask_bg = np.ones_like(masks[0]) == 1
    for mask in masks:
        mask_bg[mask] = False
        mask = mask & alpha_p
        points = np.argwhere(mask)
        points[:, [0, 1]] = points[:, [1, 0]]

    mask_bg = mask_bg & alpha_p
    return mask_bg

def disp_error(D_gt, D_est, tau, mask):
    E = np.abs(D_gt - D_est)
    n_err = np.sum((mask == 1) & (E>tau[0]) & ((E/np.abs(D_gt))>tau[1]))
    n_total = np.sum(mask == 1)
    d_err = n_err / n_total
    return d_err


def flow_err(F_gt, F_est, flow_mask, tau):
    E, F_val = flow_error_map(F_gt, F_est, flow_mask)

    F_mag = np.sqrt(F_gt[0,:,:] * F_gt[0,:,:] + F_gt[1,:,:] * F_gt[1,:,:])
    n_err = np.sum(F_val & (E>tau[0]) & ((E/F_mag)>tau[1]))
    n_total = np.sum(F_val)

    f_err = n_err/n_total
    return f_err 


def flow_error_map(F_gt, F_est, flow_mask):
    F_gt_du = F_gt[0, :,:]
    F_gt_dv = F_gt[1, :,:]
    F_gt_val = flow_mask 

    F_est_du = F_est[0,:,:]
    F_est_dv = F_est[1,:,:]

    E_du = F_gt_du - F_est_du
    E_dv = F_gt_dv - F_est_dv 
    E = np.sqrt(E_du * E_du + E_dv * E_dv)
    E[flow_mask ==0] = 0
    return E, F_gt_val

def sf_error(D1_gt, D1_est, D2_gt, D2_est, F_gt, F_est, tau, mask):
    E_f, F_val_f = flow_error_map(F_gt, F_est, mask)
    F_mag = np.sqrt(F_gt[0,:,:] * F_gt[0,:,:] + F_gt[1,:,:] * F_gt[1,:,:])

    E_d1 = np.abs(D1_gt - D1_est)
    E_d2 = np.abs(D2_gt - D2_est)

    err_d1 = ((mask == 1) & (E_d1>tau[0]) & ((E_d1/np.abs(D1_gt))>tau[1]))
    err_d2 = ((mask == 1) & (E_d2>tau[0]) & ((E_d2/np.abs(D2_gt))>tau[1]))
    err_f = F_val_f & (E_f>tau[0]) & ((E_f/F_mag)>tau[1])

    n_err = np.sum(err_d1 | err_d2 | err_f)
    n_total = np.sum(mask == 1)

    f_err = n_err / n_total
    return f_err

def warp_flow_fast(im2, u, v):
    """
    im2 warped according to (u,v).
    This is a helper function that is used to warp an image to make it match another image
    (in this case, I2 is warped to I1).
    Assumes im1[y, x] = im2[y + v[y, x], x + u[y, x]]
    """
    # this code is confusing because we assume vx and vy are the negative
    # of where to send each pixel, as in the results by ce's siftflow code
    y, x = np.mgrid[:im2.shape[0], :im2.shape[1]]
    dy = (y + v).flatten()[np.newaxis, :]
    dx = (x + u).flatten()[np.newaxis, :]
    # this says: a recipe for making im1 is to make a new image where im[y, x] = im2[y + flow[y, x, 1], x + flow[y, x, 0]]
    return np.concatenate([scipy.ndimage.map_coordinates(im2[..., i], np.concatenate([dy, dx])).reshape(im2.shape[:2] + (1,)) \
                           for i in range(im2.shape[2])], axis = 2)


def outlier_warpper(category: str, path_gt: str, path_output: str, path_seg: str, mode: int = 0, tau=[3, 0.05], path_flow=None):
    # mode: choose from 0 or 1 if category is "disparity"
    # path_flow: if category = "disparity" and mode=1, path_flow is necessary
    assert os.path.exists(path_gt)
    assert os.path.exists(path_output)

    tot = 0
    outlier_all = 0.
    outlier_fg = 0.
    outlier_bg = 0.

    if category is "flow":
        for idx in range(200):
            gt_file_name = "{:06}_10.png".format(idx)
            flow_gt, mask_gt = _get_gt_kitti_flow(os.path.join(path_gt, gt_file_name))
            output_file_name = "{:06}_10.npy".format(idx)
            output_flow = np.load(os.path.join(path_output, output_file_name))

            outlier_this_all = flow_err(flow_gt, output_flow, tau, mask_gt)

            # retrieve object mask
            obj_file_name = "{:06}_10.npy".format(idx)
            obj_mask = get_gt_kitti(os.path.join(path_seg, "image_2", obj_file_name))
            mask_fg = np.sum(obj_mask, axis=2) > 0
            mask_bg = np.logical_not(mask_fg)

            outlier_this_fg = flow_err(flow_gt, output_flow, tau, np.logical_and(mask_fg, mask_gt))
            outlier_this_bg = flow_err(flow_gt, output_flow, tau, np.logical_and(mask_bg, mask_gt))

            # for debug purpose
            print("Image #{}: all: {}, fg: {}, bg: {}".format(idx, outlier_this_all, outlier_this_fg, outlier_this_bg))
            outlier_all += outlier_this_all
            outlier_bg += outlier_this_bg
            outlier_fg += outlier_this_fg
            tot += 1
    elif category is "disparity":
        if mode == 1:
            assert path_flow is not None and os.path.exists(path_flow)

        for idx in range(200):
            if mode == 0:
                gt_file_name = "{:06}_10.png".format(idx)
                output_file_name = "{:06}_10.npy".format(idx)
                output_disp = np.load(os.path.join(path_output, output_file_name))
            elif mode == 1:
                gt_file_name = "{:06}_11.png".format(idx)
                output_file_name = "{:06}_11.npy".format(idx)
                output_disp = np.load(os.path.join(path_output, output_file_name))
                # warp to figure 1
                # read in flow
                gt_flow_file_name = "{:06}_10.png".format(idx)
                flow_gt, _ = _get_gt_kitti_flow(os.path.join(path_flow, gt_flow_file_name))
                u = flow_gt[:, :, 0]
                v = flow_gt[:, :, 1]
                output_disp = warp_flow_fast(output_disp[:, :, np.newaxis], u, v).squeeze()
            else:
                raise ValueError("mode not known: {}".format(mode))

            disp_gt = _get_gt_kitti_disparity_single_file(os.path.join(path_gt, gt_file_name))
            # output_disp = np.load(os.path.join(path_output, output_file_name))
            mask_gt = (disp_gt > 0).astype(int)
            outlier_this_all = disp_error(disp_gt, output_disp, mask_gt)

            # retrieve object mask
            obj_file_name = "{:06}_10.npy".format(idx)
            obj_mask = get_gt_kitti(os.path.join(path_seg, "image_2", obj_file_name))
            mask_fg = np.sum(obj_mask, axis=2) > 0
            mask_bg = np.logical_not(mask_fg)

            outlier_this_fg = disp_error(disp_gt, output_disp, tau, np.logical_and(mask_fg, mask_gt))
            outlier_this_bg = flow_err(disp_gt, output_disp, tau, np.logical_and(mask_bg, mask_gt))

            # for debug purpose
            print("Image #{}: all: {}, fg: {}, bg: {}".format(idx, outlier_this_all, outlier_this_fg, outlier_this_bg))
            outlier_all += outlier_this_all
            outlier_bg += outlier_this_bg
            outlier_fg += outlier_this_fg
            tot += 1
    else:
        raise ValueError("category not known {}".format(category))

    print("Summary:\n"
          "Outlier ratio per image: {}".format(outlier_all / tot))
    return outlier_all / tot


def sf_warpper(gt_path, output_path, tau=[3, 0.05]):

    error_sum = 0.0
    tot = 0
    for idx in range(200):
        disp1_gt_filename = "{:06}_10.png".format(idx)
        disp2_gt_filename = "{:06}_10.png".format(idx)
        flow_gt_filename = "{:06}_10.png".format(idx)

        D1_gt = _get_gt_kitti_disparity_single_file(os.path.join(gt_path, "disp_occ_0", disp1_gt_filename))
        D2_gt = _get_gt_kitti_disparity_single_file(os.path.join(gt_path, "disp_occ_1", disp2_gt_filename))
        F_gt, mask_F = _get_gt_kitti_flow(os.path.join(gt_path, "flow_occ", flow_gt_filename))

        disp1_est_filename = "{:06}_10.npy".format(idx)
        disp2_est_filename = "{:06}_11.npy".format(idx)
        flow_est_filename = "{:06}_10.npy".format(idx)

        D1_est = np.load(os.path.join(output_path, "disparity_0", disp1_est_filename))
        D2_est = np.load(os.path.join(output_path, "disparity_1", disp2_est_filename))
        # warp disparity with flow GT
        u = F_gt[:, :, 0]
        v = F_gt[:, :, 1]
        D2_est = warp_flow_fast(D2_est[:, :, np.newaxis], u, v).squeeze()
        F_est = np.load(os.path.join(output_path, "flow", flow_est_filename))

        mask_d1 = D1_gt > 0
        mask_d2 = D2_gt > 0
        mask = np.logical_and(mask_d1, mask_d2, mask_F)
        err = sf_error(D1_gt, D1_est, D2_gt, D2_est, F_gt, F_est, tau, mask)

        print("image #{}: {}".format(idx, err))

        error_sum += err
        tot += 1
    print("Summary:\n"
          "Error per image: {}".format(error_sum/tot))
    return error_sum / tot