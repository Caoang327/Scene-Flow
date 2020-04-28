import numpy as np
import solver_numpy as solver
import pickle
import util
import PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from scipy.optimize import least_squares
import scipy.optimize

res_outlier_d2 = []
res_outlier_f = []
res_outlier_sf = []
res_outlier_d2_ls = []
res_outlier_f_ls = []
res_outlier_sf_ls = []

res_outlier_d2_fg = []
res_outlier_f_fg = []
res_outlier_sf_fg = []

res_outlier_d2_bg = []
res_outlier_f_bg = []
res_outlier_sf_bg = []

for i in range(200):
    # Initialize file path
    calib_path1 = 'data_scene_flow_calib/training/calib_velo_to_cam/' + str(i).zfill(6) + '.txt'
    calib_path2 = 'data_scene_flow_calib/training/calib_cam_to_cam/' + str(i).zfill(6) + '.txt'
    mask_path = 'masks/' + str(i).zfill(6) + '_10.npy'
    flow_path = 'flow/' + str(i).zfill(6) + '_10.npy'
    d1_path = 'disparity_0/' + str(i).zfill(6) + '_10.npy'
    d2_path = 'disparity_1/' + str(i).zfill(6) + '_11.npy'
    L0_path = 'data_scene_flow/training/image_2/' + str(i).zfill(6) + '_10.png'
    L1_path = 'data_scene_flow/training/image_2/' + str(i).zfill(6) + '_11.png'
    mask_gt_path = 'data_semantics/training/instance/' + str(i).zfill(6) + '_10.png'

    # Load data
    data = util.load_calib_cam_to_cam(calib_path1, calib_path2)
    pi_k, T, f = data['K_cam2'], data['b_rgb'], data['f']
    masks = np.load(mask_path)
    flow = np.load(flow_path).transpose(2, 0, 1)
    disparity1 = np.load(d1_path)
    disparity2 = np.load(d2_path)
    L0 = np.array(PIL.Image.open(L0_path)).astype(np.float32) * (1.0 / 255.0)
    L1 = np.array(PIL.Image.open(L1_path)).astype(np.float32) * (1.0 / 255.0)

    # Load ground truth
    data_disparity = util.get_gt('disparity', i, root_path='data_scene_flow/training/')
    data_flow = util.get_gt('flow', i, root_path='data_scene_flow/training/')
    disparity1_gt, disparity2_gt = data_disparity["first"], data_disparity["second"]
    # disparity1_gt[disparity1_gt <= 0] = 1
    flow_gt, mask_valid = data_flow["flow"].transpose(2, 0, 1), data_flow["mask"] > 0
    mask_valid = (mask_valid & (disparity1_gt > 0) & (disparity2_gt > 0))
    masks_gt = util.get_gt_kitti(mask_gt_path)['masks']

    # Initialize thresholds for counting the outliers
    threshold_flow = 10
    threshold_d1 = 10
    threshold_d2 = 10
    threshold_m = 10
    threshold_p = 10

    # Initialization
    alpha_p = (np.ones_like(disparity1).astype(int) == 1)
    best_motions = []

    # Initialize variable to save results
    flow_ini = np.zeros_like(flow)
    flow_result = np.zeros_like(flow)
    flow_ls = np.zeros_like(flow)
    motion_map = np.zeros((L0.shape[0], L0.shape[1], 6))
    point_map = np.zeros((L0.shape[0], L0.shape[1], 3))
    d2_map = np.zeros((L0.shape[0], L0.shape[1]))
    d2_map_before = np.zeros((L0.shape[0], L0.shape[1]))
    motion_map_before = np.zeros_like(motion_map)
    point_map_before = np.zeros_like(point_map)

    # Inference for outputs of network
    point_set = solver.get_pointset(masks)
    for t, p in enumerate(point_set):
        # Get optimal motion through RANSAC and Gaussian Newton solver
        motion, p_inlier = solver.ransac(p, disparity1, disparity2, flow, alpha_p, pi_k, T, f)

        if np.sum(alpha_p[p[:, 1], p[:, 0]]) == 0:
            continue

        # Calculate errors before Gaussian Newton solver
        motion_map_before[p[:, 1], p[:, 0], :] = motion
        motion_before = motion

        trans_matrix, motion = solver.gaussian_newton(p_inlier, disparity1, disparity2, flow, alpha_p, L0, L1, motion, pi_k, T, f)
        best_motions.append(motion)

        # Record motion
        motion_map[p[:, 1], p[:, 0], :] = motion

        # Calculate 3d points
        p_3d = solver.inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)

        # delete invalid points
        p_3d = p_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
        p = p[alpha_p[p[:, 1], p[:, 0]] == 1, :]

        N = p.shape[0]
        p_homo = trans_matrix.dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
        p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
        point_map[p[:, 1], p[:, 0], :] = p3d_t2.T

        p_homo = solver.exp_map(motion_before).dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
        p3d_t2_before = p_homo[:3, :] / p_homo[3:4, :]
        point_map_before[p[:, 1], p[:, 0], :] = p3d_t2_before.T

        # Calculate the instance-wise rigid flow estimation
        p_homo = pi_k.dot(p3d_t2)
        p_t2 = p_homo[:2, :] / p_homo[2:3, :]
        flow_rigid = p_t2 - p.T
        flow_result[:, p[:, 1], p[:, 0]] = flow_rigid
        flow_ini[:, p[:, 1], p[:, 0]] = flow[:, p[:, 1], p[:, 0]]
        p_homo = pi_k.dot(p3d_t2_before)
        p_t2_before = p_homo[:2, :] / p_homo[2:3, :]
        flow_ls[:, p[:, 1], p[:, 0]] = p_t2_before - p.T

        # Calculate the estimation of disparity2
        d2_map[p[:, 1], p[:, 0]] = f * T / p3d_t2[2, :]  # of shape (N, )
        d2_map_before[p[:, 1], p[:, 0]] = f * T / p3d_t2_before[2, :]


    # Inference for ground truth
    motion_gt, motion_map_gt, point_map_gt, alpha_p_gt, mask_bg_gt = solver.get_groundtruth(masks_gt, flow_gt, disparity1_gt, disparity2_gt, L0, L1, mask_valid, pi_k, T, f)

    # Calculate errors
    # All the errors should have a shape of (m, n)
    err_flow = np.sqrt(np.sum((flow_result-flow_gt)**2, axis=0)) * (mask_valid & alpha_p)
    err_d1 = np.sqrt((disparity1-disparity1_gt)**2) * (mask_valid & alpha_p)
    err_d2 = np.sqrt((d2_map-disparity2_gt)**2) * (mask_valid & alpha_p)
    err_d2_before = np.sqrt((d2_map_before-disparity2_gt)**2) * (mask_valid & alpha_p)
    err_m = np.sqrt(np.sum((motion_map-motion_map_gt)**2, axis=2)) * (alpha_p & alpha_p_gt)
    err_p = np.sqrt(np.sum((point_map-point_map_gt)**2, axis=2)) * (alpha_p & alpha_p_gt)
    err_m_before = np.sqrt(np.sum((motion_map_before - motion_map_gt) ** 2, axis=2)) * (alpha_p & alpha_p_gt)
    err_p_before = np.sqrt(np.sum((point_map_before - point_map_gt) ** 2, axis=2)) * (alpha_p & alpha_p_gt)
    err_flow_before = np.sqrt(np.sum((flow_ls-flow_gt)**2, axis=0)) * (mask_valid & alpha_p)

    # Save results
    results = {'flow_result': flow_result, 'flow_ini': flow_ini, 'flow_gt': flow_gt, 'mask_valid': mask_valid, 'alpha_p': alpha_p,
               'disparity1': disparity1, 'disparity1_gt': disparity1_gt, 'd2_map': d2_map, 'disparity2_gt': disparity2_gt,
               'motion_map': motion_map, 'motion_map_gt': motion_map_gt, 'alpha_p_gt': alpha_p_gt, 'point_map': point_map,
               'point_map_gt': point_map_gt, 'motion_map_before': motion_map_before, 'point_map_before': point_map_before,
               'd2_map_before': d2_map_before, 'flow_ls': flow_ls}

    save_path = 'results/' + str(i).zfill(6) + '_result.pickle'
    with open(save_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Count outliers
    tau = [3, 0.05]
    # All
    disparity1_gt[disparity1_gt == 0] = -1
    disparity2_gt[disparity2_gt == 0] = -1
    flow_gt[flow_gt == 0] = -1

    mask_all = mask_valid & alpha_p
    outlier_d2 = util.disp_error(disparity2_gt, d2_map, tau, mask_all)
    outlier_f = util.flow_err(flow_gt, flow_result, mask_all, tau)
    outlier_sf = util.sf_error(disparity1_gt, disparity1, disparity2_gt, d2_map, flow_gt, flow_result, tau, mask_all)
    outlier_d2_ls = util.disp_error(disparity2_gt, d2_map_before, tau, mask_all)
    outlier_f_ls = util.flow_err(flow_gt, flow_ls, mask_all, tau)
    outlier_sf_ls = util.sf_error(disparity1_gt, disparity1, disparity2_gt, d2_map_before, flow_gt, flow_ls, tau, mask_all)

    res_outlier_d2.append(outlier_d2)
    res_outlier_f.append(outlier_f)
    res_outlier_sf.append(outlier_sf)
    res_outlier_d2_ls.append(outlier_d2_ls)
    res_outlier_f_ls.append(outlier_f_ls)
    res_outlier_sf_ls.append(outlier_sf_ls)

    # Foreground
    mask_fg = mask_valid & alpha_p & (mask_bg_gt == False)
    outlier_d2_fg = util.disp_error(disparity2_gt, d2_map, tau, mask_fg)
    outlier_f_fg = util.flow_err(flow_gt, flow_result, mask_fg, tau)
    outlier_sf_fg = util.sf_error(disparity1_gt, disparity1, disparity2_gt, d2_map, flow_gt, flow_result, tau, mask_fg)

    res_outlier_d2_fg.append(outlier_d2_fg)
    res_outlier_f_fg.append(outlier_f_fg)
    res_outlier_sf_fg.append(outlier_sf_fg)

    # Background
    mask_bg = mask_valid & alpha_p & (mask_bg_gt == True)
    outlier_d2_bg = util.disp_error(disparity2_gt, d2_map, tau, mask_bg)
    outlier_f_bg = util.flow_err(flow_gt, flow_result, mask_bg, tau)
    outlier_sf_bg = util.sf_error(disparity1_gt, disparity1, disparity2_gt, d2_map, flow_gt, flow_result, tau, mask_bg)

    res_outlier_d2_bg.append(outlier_d2_bg)
    res_outlier_f_bg.append(outlier_f_bg)
    res_outlier_sf_bg.append(outlier_sf_bg)

    # Figures
    cols = np.array([[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
                     [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
                     [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
                     [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
                     [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
                     [3 / 3.0, 6 / 3.0, 254, 224, 144],
                     [6 / 3.0, 12 / 3.0, 253, 174, 97],
                     [12 / 3.0, 24 / 3.0, 244, 109, 67],
                     [24 / 3.0, 48 / 3.0, 215, 48, 39],
                     [48 / 3.0, float('inf'), 165, 0, 38]])

    val_range = np.array([0, 0.19, 0.38, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0])
    color = cols[:, 2:] / 255

    cmap = mpl.colors.ListedColormap(color)
    norm = mpl.colors.BoundaryNorm(val_range, cmap.N)

    cmap.set_bad('k', 1.0)

    # Errors of motion
    # plt.figure(1)
    # result = np.sqrt(np.sum((motion_map-motion_map_gt)**2, axis=2) / np.sum(motion_map_gt**2, axis=2))
    # result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    # plt.imshow(result, cmap='bwr', vmin=0, vmax=1)
    # plt.axis('off')
    # plt.title('Normalized root square errors of motion')
    # plt.savefig('results/' + str(i).zfill(6) + '_err_m_norm.png')
    # plt.clf()

    plt.figure(2)
    result = np.sqrt(np.sum((motion_map-motion_map_gt)**2, axis=2))
    result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title('Root square errors of motion')
    plt.savefig('results/' + str(i).zfill(6) + '_err_m.png')
    plt.clf()

    # Errors of points
    # plt.figure(3)
    # result = np.sqrt(np.sum((point_map-point_map_gt)**2, axis=2) / np.sum(point_map_gt**2, axis=2))
    # result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    # plt.imshow(result, cmap='bwr', vmin=0, vmax=1)
    # plt.axis('off')
    # plt.colorbar(orientation='horizontal', pad=0.05)
    # plt.title('Normalized root square errors of 3D coordinates')
    # plt.savefig('results/' + str(i).zfill(6) + '_err_p_norm.png')
    # plt.clf()

    plt.figure(4)
    result = np.sqrt(np.sum((point_map-point_map_gt)**2, axis=2))
    result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title('Root square errors of 3D coordinates')
    plt.savefig('results/' + str(i).zfill(6) + '_err_p.png')
    plt.clf()

    # Flow
    plt.figure(5)
    result = util.flow_to_color(flow_result.transpose(1, 2, 0))
    plt.imshow(result)
    plt.axis('off')
    plt.title('Flow after (DRISF)')
    plt.savefig('results/' + str(i).zfill(6) + '_flow_after.png')
    plt.clf()

    plt.figure(6)
    result = util.flow_to_color(flow_ini.transpose(1, 2, 0))
    plt.imshow(result)
    plt.axis('off')
    plt.title('Flow before (PWC)')
    plt.savefig('results/' + str(i).zfill(6) + '_flow_before.png')
    plt.clf()

    plt.figure(7)
    result = np.ma.masked_where((alpha_p & mask_valid)==False, err_flow)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title(' Flow errors After (DRISF)')
    plt.savefig('results/' + str(i).zfill(6) + '_err_f_after.png')
    plt.clf()

    plt.figure(8)
    flow_error = np.sqrt(np.sum((flow_ini - flow_gt)**2, axis=0))
    result = np.ma.masked_where((alpha_p & mask_valid)==False, flow_error)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title('Flow errors before (PWC)')
    plt.savefig('results/' + str(i).zfill(6) + '_err_f_before.png')
    plt.clf()

    # Disparity
    plt.figure(9)
    plt.imshow(d2_map)
    plt.axis('off')
    plt.title('D2 after (DRISF)')
    plt.savefig('results/' + str(i).zfill(6) + '_D_after.png')
    plt.clf()

    plt.figure(10)
    result = np.ma.masked_where((alpha_p & mask_valid)==False, err_d2)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title(' D2 errors after (DRISF)')
    plt.savefig('results/' + str(i).zfill(6) + '_err_d_after.png')
    plt.clf()

    plt.figure(11)
    result = np.ma.masked_where((alpha_p & mask_valid) == False, err_d2_before)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title(' D2 errors before (LS)')
    plt.savefig('results/' + str(i).zfill(6) + '_err_d_before.png')
    plt.clf()

    plt.figure(12)
    # result = np.sqrt(np.sum((motion_map_before - motion_map_gt) ** 2, axis=2) / np.sum(motion_map_gt ** 2, axis=2))
    # result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    # plt.imshow(result, cmap='bwr', vmin=0, vmax=1)
    # plt.axis('off')
    # plt.colorbar(orientation='horizontal', pad=0.05)
    # plt.title('Normalized root square errors of motion (LS)')
    # plt.savefig('results/' + str(i).zfill(6) + '_err_m_before_norm.png')
    # plt.clf()

    plt.figure(13)
    result = np.sqrt(np.sum((motion_map_before - motion_map_gt) ** 2, axis=2))
    result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title('Root square errors of motion (LS)')
    plt.savefig('results/' + str(i).zfill(6) + '_err_m_before.png')
    plt.clf()

    plt.figure(14)
    # result = np.sqrt(np.sum((point_map_before - point_map_gt) ** 2, axis=2) / np.sum(point_map_gt ** 2, axis=2))
    # result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    # plt.imshow(result, cmap='bwr', vmin=0, vmax=1)
    # plt.axis('off')
    # plt.colorbar(orientation='horizontal', pad=0.05)
    # plt.title('Normalized root square errors of 3D coordinates (LS)')
    # plt.savefig('results/' + str(i).zfill(6) + '_err_p_before_norm.png')
    # plt.clf()

    plt.figure(15)
    result = np.sqrt(np.sum((point_map_before - point_map_gt) ** 2, axis=2))
    result = np.ma.masked_where((alpha_p & alpha_p_gt) == False, result)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title('Root square errors of 3D coordinates (LS)')
    plt.savefig('results/' + str(i).zfill(6) + '_err_p_before.png')
    plt.clf()

    plt.figure(16)
    result = np.ma.masked_where((alpha_p & mask_valid) == False, err_flow_before)
    plt.imshow(result, cmap=cmap, norm=norm)
    plt.axis('off')
    plt.title(' Flow errors before (LS)')
    plt.savefig('results/' + str(i).zfill(6) + '_err_f_ls.png')
    plt.clf()

    # print('flow:', outlier_flow, outlier_flow_bg, outlier_flow_fg)
    # print('d1:', outlier_d1, outlier_d1_bg, outlier_d1_fg)
    # print('d2:', outlier_d2, outlier_d2_bg, outlier_d2_fg)
    # print('motion:', outlier_m, outlier_m_bg, outlier_m_fg)
    # print('points:', outlier_p, outlier_p_bg, outlier_p_fg)

# Save results
res_outliers = {'outlier_flow': res_outlier_f, 'outlier_flow_fg': res_outlier_f_fg, 'outlier_flow_bg': res_outlier_f_bg,
                'outlier_d2': res_outlier_d2, 'outlier_d2_fg': res_outlier_d2_fg, 'outlier_d2_bg': res_outlier_d2_bg,
                'outlier_sf': res_outlier_sf, 'outlier_sf_fg': res_outlier_sf_fg, 'outlier_sf_bg': res_outlier_sf_bg,
                'outlier_flow_ls': res_outlier_f_ls, 'outlier_d2_ls': res_outlier_d2_ls, 'outlier_sf_ls': res_outlier_sf_ls}

with open('results/res_outlier.pickle', 'wb') as handle:
    pickle.dump(res_outliers, handle, protocol=pickle.HIGHEST_PROTOCOL)


