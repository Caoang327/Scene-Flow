import numpy as np
import solver_numpy as solver
import pickle
import util
import PIL

# Load data
data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
pi_k, T, f = data['K_cam2'], data['b_rgb'], data['f']
masks = np.load("pmask_left_t0.npy")
flow = np.load('flow.npy').transpose(2, 0, 1)
disparity1 = np.load("disparity1.npy")
disparity2 = np.load("disparity2.npy")
L0 = np.array(PIL.Image.open("left_000000_10.png")).astype(np.float32) * (1.0 / 255.0)
L1 = np.array(PIL.Image.open("left_000000_11.png")).astype(np.float32) * (1.0 / 255.0)

# Load ground truth
disparity_data = np.load("disparity.npz")
flow_data = np.load("flow.npz")
disparity1_gt, disparity2_gt = disparity_data["first"], disparity_data["second"]
disparity1_gt[disparity1_gt<=0] = 1
flow_gt, mask_valid = flow_data["flow"].transpose(2, 0, 1) * 20, flow_data["mask"] > 0
with open('img2_000000_10.pickle', 'rb') as handle:
    masks_gt = pickle.load(handle)['masks']

# Initialize thresholds for counting the outliers
threshold_flow = 10
threshold_d1 = 10
threshold_d2 = 10
threshold_m = 10
threshold_p = 10

# Initialization
alpha_p = np.ones_like(disparity1).astype(int)
motion_0 = np.ones(6)
best_motions = []

# Initialize variable to save results
flow_ini = np.zeros_like(flow)
flow_result = np.zeros_like(flow)
motion_map = np.zeros((L0.shape[0], L0.shape[1], 6))
point_map = np.zeros((L0.shape[0], L0.shape[1], 3))
d2_map = np.zeros_like(L0)

# Inference for outputs of network
point_set = solver.get_pointset(masks)
for t, p in enumerate(point_set):
    # Get optimal motion through RANSAC and Gaussian Newton solver
    motion, p_inlier = solver.ransac(p, disparity1, disparity2, flow, alpha_p)
    trans_matrix, motion = solver.gaussian_newton(p_inlier, disparity1, disparity2, flow, alpha_p, L0, L1, motion)
    best_motions.append(motion)

    # Record motion
    motion_map[p[:, 1], p[:, 0], :] = motion

    # Calculate 3d error
    p_3d = solver.inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
    flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
    q = p + flow_q.T
    disparity_q = solver.bilinear_interpolate_numpy(disparity2, q, alpha_p, p)  # of shape (N, )
    q_3d = solver.inverse_project(q, pi_k, disparity_q, T, f)

    # delete invalid points
    p_3d = p_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
    q_3d = q_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
    p = p[alpha_p[p[:, 1], p[:, 0]] == 1, :]

    N = p.shape[0]
    p_homo = trans_matrix.dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
    p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
    point_map[p[:, 1], p[:, 0], :] = p3d_t2.T

    # Calculate the instance-wise rigid flow estimation
    p_homo = pi_k.dot(p3d_t2)
    p_t2 = p_homo[:2, :] / p_homo[2:3, :]
    flow_rigid = p_t2 - p.T
    flow_result[:, p[:, 1], p[:, 0]] = flow_rigid
    flow_ini[:, p[:, 1], p[:, 0]] = flow[:, p[:, 1], p[:, 0]]

    # Calculate the estimation of disparity2
    d2_map[p[:, 1], p[:, 0]] = f * T / p3d_t2[2, :]  # of shape (N, )

# Inference for ground truth
motion_gt, motion_map_gt, point_map_gt, alpha_p_gt = solver.get_groundtruth(masks_gt, flow_gt, disparity1_gt, disparity2_gt, mask_valid, pi_k, T, f)

# Calculate errors
err_flow = np.sqrt(np.sum((flow_result-flow_gt)**2, axis=0)) * (mask_valid & alpha_p)
err_d1 = np.sqrt((disparity1-disparity1_gt)**2) * (mask_valid & alpha_p)
err_d2 = np.sqrt((d2_map-disparity2_gt)**2) * (mask_valid & alpha_p)
err_m = np.sqrt(np.sum((motion_map-motion_map_gt)**2, axis=2))
err_p = np.sqrt(np.sum((point_map-point_map_gt)**2, axis=2))

# Count outliers



