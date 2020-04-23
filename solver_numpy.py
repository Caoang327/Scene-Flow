import numpy as np
import util
import matplotlib.pyplot as plt
import copy
from scipy.optimize import least_squares
import PIL
import PIL.Image


def exp_map(motion):
    """
    Map the rigid motion from se(3) to SE(3)

    Inputs:
    - motion: rigid motion representation in se(3), array of shape (6,)

    Outputs:
    - T: the corresponding transform matrix in SE(3), array of shape (4, 4)
    """
    motion = motion.reshape(-1, 1)
    u = motion[:3]
    w = motion[3:]
    theta = np.sqrt(w.T.dot(w))
    A = np.sin(theta) / theta
    B = (1 - np.cos(theta)) / theta ** 2
    C = (1 - A) / theta ** 2
    w_x = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + A * w_x + B * w_x.dot(w_x)
    V = np.eye(3) + B * w_x + C * w_x.dot(w_x)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3:4] = V.dot(u)
    T[3, 3] = 1

    return T


def get_pointset(masks):
    """
    Get the instance points set from the masks obtained from mask-rcnn

    Inputs:
    - masks: masks obtained from mask-rcnn, array of shape (N, m, n)

    Outputs:
    - p_set: list of the point set of all instances, list<array of shape (N, 2)>
    """
    p_set = []
    for mask in masks:
        points = np.argwhere(mask)
        points[:, [0, 1]] = points[:, [1, 0]]
        p_set.append(points)

    return p_set


def inverse_project(p, pi_k, d, T, f):
    """
    Get 3d point in the world coordinate from pixel and disparity

    Inputs:
    - p: image coordinates of the points, array of shape (N, 2)
    - pi_k: perspective project function, array of shape (3, 3)
    - d: the corresponding disparity, array of shape(N,)
    - T: length of baseline, scalar
    - f: focal length, scalar

    Outputs:
    - p_3d: 3d point in the world coordinate, tensor of shape (3, N)
    """
    depth = f * T / d  # of shape (N, 1)
    N = p.shape[0]
    p_3d = np.linalg.inv(pi_k).dot(np.concatenate((p.T, np.ones((1, N))), axis=0)) * depth.reshape(1, N)

    return p_3d


def bilinear_interpolate_numpy(im, q, alpha_p=None, p=None, flag=True):
    """
    Implement the bilinear interpolation
    revised version
    original codes: https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

    Inputs:
    - im: the value matrix, can be img/disparity, array of shape (m, n, )
    - q: the point set at time t+1, array of shape (N, 2)
    - alpha_p: indicator, array of shape (m, n)
    - p: the point set at time t, array of shape (N, 2)
    - flag: if check result==0, the default value is True

    Outputs:
    - result: of shape (N,)
    """

    x = q[:, 0]
    y = q[:, 1]

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1

    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)
    x = np.clip(x, 0, im.shape[1] - 1)
    y = np.clip(y, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    if len(im.shape) == 2:
        result = Ia * wa + Ib * wb + Ic * wc + Id * wd
    elif len(im.shape) == 3:
        result = Ia * wa.reshape(-1, 1) + Ib * wb.reshape(-1, 1) + Ic * wc.reshape(-1, 1) + Id * wd.reshape(-1, 1)
    # print(p[result==0])
    if flag:
        alpha_p[p[result == 0, 1], p[result == 0, 0]] = 0
        result[result == 0] = 1
        alpha_p[p[Ia == 0, 1], p[Ia == 0, 0]] = 0
        alpha_p[p[Ib == 0, 1], p[Ib == 0, 0]] = 0
        alpha_p[p[Ic == 0, 1], p[Ic == 0, 0]] = 0
        alpha_p[p[Id == 0, 1], p[Id == 0, 0]] = 0
    return result
    # return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)


def cost_function(motion, p, disparity1, disparity2, flow, alpha_p):
    """
    Calculate the E_rigid

    Inputs:
    - motion: rigid motion representation in se(3), array of shape (6,)
    - p: point set of one instance in 1st img, array of shape (N, 2)
    - disparity1: disparity map of time t, array of shape (m, n)
    - disparity2: disparity map of time t+1, array of shape (m, n)
    - flow: flow matrix of 1st img, array of shape (2, m, n)
    - alpha_p: indicators representing if it is a outlier, array of shape (m, n)

    Outputs:
    - E: value of E_rigid, array fo shape (N,)
    """
    # alpha = 0.225
    alpha = 0.5
    epsilon = 1e-5
    N = p.shape[0]
    data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
    pi_k, T, f = data['K_cam2'], data['b_rgb'], data['f']
    p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
    flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
    q = p + flow_q.T
    disparity_q = bilinear_interpolate_numpy(disparity2, q, alpha_p, p)  # of shape (N, )
    q_3d = inverse_project(q, pi_k, disparity_q, T, f)
    p_homo = exp_map(motion).dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
    p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
    residual = p3d_t2 - q_3d
    # print(np.sum(np.abs(residual), axis=0))
    # print(residual)
    E = ((np.sum(residual ** 2, axis=0) + epsilon ** 2) ** alpha) * alpha_p[p[:, 1], p[:, 0]]

    return E


def ransac(p, disparity1, disparity2, flow, alpha_p):
    """
    Implement RANSAC algorithm to find the optimal motion

    Inputs:
    - p: point set of one instance in 1st img, array of shape (N, 2)
    - disparity1: disparity map of time t, array of shape (m, n)
    - disparity2: disparity map of time t+1, array of shape (m, n)
    - flow: flow matrix of 1st img, array of shape (2, m, n)
    - alpha_p: indicators representing if it is a outlier, array of shape (m, n)

    Outputs:
    - best_motion: the optimal motion, array of shape (6,)
    """
    num_iters = 5
    epsilon = 5
    N = p.shape[0]
    num_inliers = 0
    idxset_best = []
    motion_0 = np.ones(6)
    for i in range(num_iters):
        idx = np.random.choice(N, N//15, replace=False)
        res = least_squares(cost_function, motion_0, args=(p[idx, :], disparity1, disparity2, flow, alpha_p))
        motion = res.x

        # calculate residual
        data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
        pi_k, T, f = data['K_cam2'], data['b_rgb'], data['f']
        p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
        flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
        q = p + flow_q.T
        disparity_q = bilinear_interpolate_numpy(disparity2, q, alpha_p, p)  # of shape (N, )
        q_3d = inverse_project(q, pi_k, disparity_q, T, f)
        p_homo = exp_map(motion).dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
        p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
        residual = np.sqrt(np.sum((p3d_t2 - q_3d)**2, axis=0))
        if num_inliers < np.sum(residual < epsilon):
            idxset_best = np.argwhere(residual < epsilon)[:, 0]
            num_inliers = np.sum(residual < epsilon)
    best_res = least_squares(cost_function, motion_0, args=(p[idxset_best, :], disparity1, disparity2, flow, alpha_p))
    best_motion = best_res.x
    # alpha_p[p[residual > epsilon, 1], p[residual > epsilon, 0]] = 0
    p = p[idxset_best, :]
    # print(best_res.cost)
    # print(num_inliers, N)
    # print(residual)
    # print(cost_function(best_motion, p, disparity1, disparity2, flow, alpha_p))

    return best_motion, p


def jacobian_rigidfit(p_3d, q_3d):
    """
    Calculate the Jacobian of the Rigid Fitting term

    Inputs:
    - p_3d: the 3D coordinate of pixels after inverse depth warping and
    applying rigid transform, array of shape (3, N)
    - q_3d: the 3D coordinate of pixels at time t+1, array of shape (3, N)

    Outputs:
    - JWJ: the result of J^{T}WJ, array of shape (N, 6, 6)
    - JWr: the result of J^{T}Wr, array of shape (N, 6, 1)
    """
    alpha = 0.45
    epsilon = 1e-5
    N = p_3d.shape[1]
    J = np.zeros((N, 3, 6))
    J[:, :3, :3] = np.eye(3)
    J[:, 0, 4] = p_3d[2, :]
    J[:, 0, 5] = -p_3d[1, :]
    J[:, 1, 3] = -p_3d[2, :]
    J[:, 1, 5] = p_3d[0, :]
    J[:, 2, 3] = p_3d[1, :]
    J[:, 2, 4] = -p_3d[0, :]

    r = p_3d - q_3d
    Lp = np.sum(r**2, axis=0).reshape(-1, 1)
    W = alpha * (Lp + epsilon**2)**(alpha - 1)

    JWJ = W.reshape(N, 1, 1) * np.matmul(J.transpose((0, 2, 1)), J)
    JWr = W.reshape(N, 1, 1) * np.matmul(J.transpose((0, 2, 1)), r.T.reshape(N, 3, 1))

    return JWJ, JWr


def jacobian_flow(p_3d, p, fx, fy, pi_k, flow):
    """
    Calculate the Jacobian of the Flow Consistency term

    Inputs:
    - p_3d: the 3D coordinate of pixels after inverse depth warping and
    applying rigid transform, array of shape (3, N)
    - p: the 2D points, array of shape (N, 2)
    - f_x, f_y: paramaters in intrinsic K, scalar
    - pi_k: perspective projection matrix, array of shape (3, 3)
    - flow: flow matrix, array of shape (2, m, n)

    Outputs:
    - JWJ: the result of J^{T}WJ, array of shape (N, 6, 6)
    - JWr: the result of J^{T}Wr, array of shape (N, 6, 1)
    """
    alpha = 0.45
    epsilon = 1e-5
    N = p_3d.shape[1]
    J1 = np.zeros((N, 2, 3))
    J2 = np.zeros((N, 3, 6))
    J1[:, 0, 0] = fx / p_3d[2, :]
    J1[:, 0, 2] = -p_3d[0, :] * fx / p_3d[2, :]**2
    J1[:, 1, 1] = fy / p_3d[2, :]
    J1[:, 1, 2] = -p_3d[1, :] * fy / p_3d[2, :]**2
    J2[:, :3, :3] = np.eye(3)
    J2[:, 0, 4] = p_3d[2, :]
    J2[:, 0, 5] = -p_3d[1, :]
    J2[:, 1, 3] = -p_3d[2, :]
    J2[:, 1, 5] = p_3d[0, :]
    J2[:, 2, 3] = p_3d[1, :]
    J2[:, 2, 4] = -p_3d[0, :]
    J = np.matmul(J1, J2)

    p_homo = pi_k.dot(p_3d)
    p_t2 = p_homo[:2, :] / p_homo[2:3, :]
    r = p_t2 - p.T - flow[:, p[:, 1], p[:, 0]]
    Lp = np.sum(r**2, axis=0).reshape(-1, 1)
    W = alpha * (Lp + epsilon**2)**(alpha - 1)

    JWJ = W.reshape(N, 1, 1) * np.matmul(J.transpose((0, 2, 1)), J)
    JWr = W.reshape(N, 1, 1) * np.matmul(J.transpose((0, 2, 1)), r.T.reshape(N, 2, 1))
    return JWJ, JWr


def jacobian_photometric(p_3d, p, L0, L1, fx, fy, pi_k):
    """
    Calculate the Jacobian of the Photometric Error term

    Inputs:
    - p_3d: the 3D coordinate of pixels after inverse depth warping and
    applying rigid transform, array of shape (3, N)
    - p: the 2D points, array of shape (N, 2)
    - L0, L1: left image at time t & t+1, array of shape (m, n, 3)
    - f_x, f_y: parameters in intrinsic K, scalar
    - pi_k: perspective projection matrix, array of shape (3, 3)

    Outputs:
    - JWJ: the result of J^{T}WJ, array of shape (N, 6, 6)
    - JWr: the result of J^{T}Wr, array of shape (N, 6, 1)
    """
    alpha = 0.45
    epsilon = 1e-5
    N = p.shape[0]

    p_homo = pi_k.dot(p_3d)
    p_t2 = p_homo[:2, :] / p_homo[2:3, :]
    x = np.floor(p_t2[0, :]).astype(int)
    y = np.floor(p_t2[1, :]).astype(int)
    x0 = np.ceil(p_t2[0, :] - 1).astype(int)
    x1 = np.floor(p_t2[0, :] + 1).astype(int)
    y0 = np.ceil(p_t2[1, :] - 1).astype(int)
    y1 = np.floor(p_t2[1, :] + 1).astype(int)

    x0 = np.clip(x0, 0, L1.shape[1] - 1)
    x1 = np.clip(x1, 0, L1.shape[1] - 1)
    y0 = np.clip(y0, 0, L1.shape[0] - 1)
    y1 = np.clip(y1, 0, L1.shape[0] - 1)
    x = np.clip(x, 0, L1.shape[1] - 1)
    y = np.clip(y, 0, L1.shape[0] - 1)

    J1 = np.zeros((N, 3, 2))
    J2 = np.zeros((N, 2, 3))
    J3 = np.zeros((N, 3, 6))
    J1[:, :, 0] = (L1[y, x1] - L1[y, x0]).reshape(-1, 3)
    J1[:, :, 1] = (L1[y1, x] - L1[y0, x]).reshape(-1, 3)
    J2[:, 0, 0] = fx / p_3d[2, :]
    J2[:, 0, 2] = -p_3d[0, :] * fx / p_3d[2, :] ** 2
    J2[:, 1, 1] = fy / p_3d[2, :]
    J2[:, 1, 2] = -p_3d[1, :] * fy / p_3d[2, :] ** 2
    J3[:, :3, :3] = np.eye(3)
    J3[:, 0, 4] = p_3d[2, :]
    J3[:, 0, 5] = -p_3d[1, :]
    J3[:, 1, 3] = -p_3d[2, :]
    J3[:, 1, 5] = p_3d[0, :]
    J3[:, 2, 3] = p_3d[1, :]
    J3[:, 2, 4] = -p_3d[0, :]

    J = np.matmul(np.matmul(J1, J2), J3)

    r = bilinear_interpolate_numpy(L1, p_t2.T, flag=False) - L0[p[:, 1], p[:, 0], :]  # of shape (N, 3)
    Lp = np.sum(r**2, axis=1)
    W = alpha * (Lp + epsilon ** 2) ** (alpha - 1)

    JWJ = W.reshape(N, 1, 1) * np.matmul(J.transpose((0, 2, 1)), J)
    JWr = W.reshape(N, 1, 1) * np.matmul(J.transpose((0, 2, 1)), r.reshape(N, 3, 1))

    return JWJ, JWr


def gaussian_newton(p,  disparity1, disparity2, flow, alpha_p, L0, L1, motion_0):
    """
    Implement Gaussian Newton solver

    Inputs:
    - p: point set of one instance in 1st img, array of shape (N, 2)
    - disparity1: disparity map of time t, array of shape (m, n)
    - disparity2: disparity map of time t+1, array of shape (m, n)
    - flow: flow matrix of 1st img, array of shape (2, m, n)
    - alpha_p: indicators representing if it is a outlier, array of shape (m, n)
    - L0, L1: left image at time t & t+1, array of shape (m, n, 3)
    - motion_0: initialization of 3D motion, array of shape (6,)

    Outputs:
    - trans_matrix: the exponential mapping of the optimal motion, array of shape (4, 4)
    - motion : the optimal motion, array of shape (4, 4)
    """
    trans_matrix = exp_map(motion_0)
    motion = motion_0
    num_iters = 30
    data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
    pi_k, T, f = data['K_cam2'], data['b_rgb'], data['f']
    fx = fy = f  # temporary
    for i in range(num_iters):
        N = p.shape[0]
        p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)
        p_homo = trans_matrix.dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
        p_3d_t2 = p_homo[:3, :] / p_homo[3:4, :]

        flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
        q = p + flow_q.T
        disparity_q = bilinear_interpolate_numpy(disparity2, q, alpha_p, p)  # of shape (N, )
        q_3d = inverse_project(q, pi_k, disparity_q, T, f)

        # delete the invalid points
        p_3d_t2 = p_3d_t2[:, alpha_p[p[:, 1], p[:, 0]] == 1]
        q_3d = q_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
        p = p[alpha_p[p[:, 1], p[:, 0]] == 1, :]

        JWJ1, JWr1 = jacobian_photometric(p_3d_t2, p, L0, L1, fx, fy, pi_k)
        JWJ2, JWr2 = jacobian_rigidfit(p_3d_t2, q_3d)
        JWJ3, JWr3 = jacobian_flow(p_3d_t2, p, fx, fy, pi_k, flow)

        dmotion = -np.linalg.inv(np.sum(JWJ1+JWJ2+JWJ3, axis=0)).dot(np.sum(JWr1+JWr2+JWr3, axis=0))  # of shape (6, 1)
        # dmotion = -np.linalg.inv(np.sum(JWJ2, axis=0)).dot(np.sum(JWr2, axis=0))  # of shape (6, 1)
        # print(dmotion)

        motion = motion + dmotion.reshape(6)
        trans_matrix = exp_map(motion)
        # trans_matrix = trans_matrix.dot(exp_map(dmotion.reshape(6)))

    return trans_matrix, motion


def get_groundtruth(masks, flow, disparity1, disparity2, alpha_p):
    """
    Find the optimal motion using the ground truth of flow, mask, and disparity

    Inputs:
    - masks: the ground truth of masks obtained from mask-rcnn, array of shape (N, m, n)
    - flow: the ground truth of flow, array of shape (2, m, n)
    - disparity1: the ground truth of disparity map at time t, array of shape (m, n)
    - disparity2: the ground truth of disparity map at time t+1, array of shape (m, n)
    - alpha_p: the mask matrix indicating if the point is valid

    Outputs:
    - motions: the list of optimal motions, list<array of shape (6, )>
    """
    p_set = []
    for mask in masks:
        mask = mask & alpha_p
        points = np.argwhere(mask)
        points[:, [0, 1]] = points[:, [1, 0]]
        p_set.append(points)

    motion_0 = np.ones(6)
    motions = []
    t = 0
    for p in p_set[1:2]:
        res = least_squares(cost_function, motion_0, args=(p, disparity1, disparity2, flow, alpha_p))
        motion_gt = res.x
        motions.append(motion_gt)

        # # test errors
        trans_matrix = exp_map(motion_gt)
        data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
        pi_k, T, f = data['K_cam2'], data['b_rgb'], data['f']
        p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
        flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
        q = p + flow_q.T
        disparity_q = bilinear_interpolate_numpy(disparity2, q, alpha_p, p)  # of shape (N, )
        q_3d = inverse_project(q, pi_k, disparity_q, T, f)

        # delete invalid points
        p_3d = p_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
        q_3d = q_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
        p = p[alpha_p[p[:, 1], p[:, 0]] == 1, :]

        N = p.shape[0]
        p_homo = trans_matrix.dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))

        p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
        residual = p3d_t2 - q_3d
        print(t)
        t += 1
        print(np.sum(np.abs(residual), axis=0))
        print(np.sum(np.abs(residual)))
        print(np.median(np.sum(np.abs(residual), axis=0)))

    return motions


masks = np.load("pmask_left_t0.npy")
flow = np.load("pwc_image_2.npy")
disparity = np.load("disparity(1).npz")
disparity1, disparity2 = disparity["first"], disparity["second"]
# disparity1 = np.load("psm_1(1).npy")
# disparity2 = np.load("psm_2(1).npy")
L0 = np.array(PIL.Image.open("left_000000_10.png")).astype(np.float32) * (1.0 / 255.0)
L1 = np.array(PIL.Image.open("left_000000_11.png")).astype(np.float32) * (1.0 / 255.0)

result = util.flow_to_color(flow.transpose(1, 2, 0))
plt.imshow(result, cmap='gray')
plt.show()

disparity_data = np.load("disparity.npz")
flow_data = np.load("flow.npz")

disparity1_gt, disparity2_gt = disparity_data["first"], disparity_data["second"]
disparity1_gt[disparity1_gt<=0] = 1

flow_gt, mask_gt = flow_data["flow"].transpose(2, 0, 1), flow_data["mask"] > 0

# result = util.flow_to_color(flow_gt.transpose(1, 2, 0))
# plt.imshow(disparity1)
# plt.show()


# flow_result = copy.deepcopy(flow)
flow_result = np.zeros_like(flow)
alpha_p = np.ones_like(disparity1).astype(int)
# motion_0 = 5 * np.random.randn(6)
motion_0 = np.ones(6)
t = 0
best_motions = []
for p in get_pointset(masks)[1:2]:
    # print(flow[:, p[:, 1], p[:, 0]])
    # print(disparity1[p[:, 1], p[:, 0]])

    motion, p_inlier = ransac(p, disparity1, disparity2, flow, alpha_p)
    # p = p_inlier

    # res = least_squares(cost_function, motion_0, args=(p, disparity1, disparity2, flow, alpha_p))
    # motion = res.x
    # motion = np.array([-1.33551372e+00, 1.41114219e+00, 5.61324558e+00, -4.55258878e-02, -8.33163654e-02, 2.22574094e-03])  # 1
    # motion = np.array([0.59305141, 0.9614412, 1.81942165, -0.33966084, 0.02325005, 0.56535765])  # 15
    # motion = motion_0
    # print(motion)

    trans_matrix, motion = gaussian_newton(p, disparity1, disparity2, flow, alpha_p, L0, L1, motion)
    # trans_matrix, motion = gaussian_newton(p_inlier, disparity1, disparity2, flow, alpha_p, L0, L1, motion)
    best_motions.append(motion)
    # trans_matrix = exp_map(motion)
    # print(trans_matrix)
    # calculate 3d error
    data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
    pi_k, T, f = data['K_cam2'], data['b_rgb'], data['f']
    # print(pi_k, T)
    p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
    flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
    q = p + flow_q.T
    disparity_q = bilinear_interpolate_numpy(disparity2, q, alpha_p, p)  # of shape (N, )
    q_3d = inverse_project(q, pi_k, disparity_q, T, f)

    # delete invalid points
    p_3d = p_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
    q_3d = q_3d[:, alpha_p[p[:, 1], p[:, 0]] == 1]
    p = p[alpha_p[p[:, 1], p[:, 0]] == 1, :]

    N = p.shape[0]
    p_homo = trans_matrix.dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))

    p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
    residual = p3d_t2 - q_3d
    print(t)
    t += 1
    print(np.sum(np.abs(residual), axis=0))
    print(np.sum(np.abs(residual)))
    print(np.median(np.sum(np.abs(residual), axis=0)))
#     # print(residual)
#     # print(residual.shape)
#
#
#     # calculate the instance-wise rigid flow estimation
#     p_3d_t2 = trans_matrix.dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
#     p_3d_t2 = p_3d_t2[:3, :] / p_3d_t2[3:4, :]
#     p_homo = pi_k.dot(p_3d_t2)
#     p_t2 = p_homo[:2, :] / p_homo[2:3, :]
#     flow_rigid = p_t2 - p.T
#     print(np.sum(np.abs(flow[:, p[:, 1], p[:, 0]] - flow_rigid))/N)
#     print(np.sum(np.abs(flow_gt[:, p[:, 1], p[:, 0]] - flow_rigid)) / N)
#     # print(flow_result[:, p[:, 1], p[:, 0]])
#     # print(flow_rigid)
#     flow_result[:, p[:, 1], p[:, 0]] = flow_rigid
#
# result = util.flow_to_color(flow_result.transpose(1, 2, 0))
# plt.imshow(result)
# plt.show()

motion_gt = get_groundtruth(masks, flow, disparity1, disparity2, np.ones_like(disparity1)==1)
for i in range(len(motion_gt)):
    if len(motion_gt[i]):
        print(i, "gt", motion_gt[i])
        print(i, "estimated", best_motions[i])
        print(i, np.sqrt(np.sum((motion_gt[i] - best_motions[i])**2)))
        print(i, np.sqrt(np.sum((motion_gt[i] - best_motions[i])**2) / np.sum(motion_gt[i]**2)))
# print(np.sum((motion_gt-motion)**2))
# print(np.sqrt(np.sum((motion - motion_gt)**2) / np.sum(motion_gt**2)))
# motion = motion / np.sqrt(np.sum(motion**2))
# motion_gt = motion_gt / np.sqrt(np.sum(motion_gt**2))
#
# print(np.sum((motion_gt-motion)**2))

# print(motion)
