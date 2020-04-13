## util function 

import numpy as np

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

    # Compute the rectified extrinsics from cam0 to camN
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

    data['f'] = data['R_rect_30'][0,0]

    pik00 = P_rect_00.dot(R_rect_00.dot(data['T_cam0_velo']))
    pik10 = P_rect_10.dot(R_rect_10.dot(data['T_cam1_velo']))
    pik20 = P_rect_20.dot(R_rect_20.dot(data['T_cam2_velo']))
    pik30 = P_rect_30.dot(R_rect_30.dot(data['T_cam3_velo']))

    data['Pik00'] = pik00
    data['Pik10'] = pik10
    data['Pik20'] = pik20
    data['Pik30'] = pik30

    return data