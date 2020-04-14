import numpy as np
import util
import matplotlib.pyplot as plt
import copy
from scipy.optimize import least_squares


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
    - mask: masks obtained from mask-rcnn, array of shape (N, m, n)

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
    - pi_k: perspective project function, array of shape (3, 4)
    - d: the corresponding disparity, array of shape(N,)
    - T: length of baseline, scalar
    - f: focal length, scalar

    Outputs:
    - p_3d: 3d point in the world coordinate, tensor of shape (3, N)
    """
    depth = f * T / d  # of shape (N, 1)
    N = p.shape[0]
    p_homo = np.linalg.pinv(pi_k).dot(np.concatenate((p.T, np.ones((1, N))), axis=0)) * depth.reshape(1, N)
    p_3d = p_homo[:3, :] / p_homo[3:4, :]

    return p_3d


def bilinear_interpolate_numpy(im, q, alpha_p, p):
    """
    Implement the bilinear interpolation
    revised version
    original codes: https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

    Inputs:
    - im: the value matrix, can be img/disparity, array of shape (m, n)
    - q: the point set at time t+1, array of shape (N, 2)
    - alpha_p: indicator, array of shape (m, n)
    - p: the point set at time t, array of shape (N, 2)

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

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    result = Ia * wa + Ib * wb + Ic * wc + Id * wd
    # print(p[result==0])
    alpha_p[p[result == 0, 1], p[result == 0, 0]] = 0
    result[result == 0] = 1

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
    - E: value of E_rigid, scalar
    """
    alpha = 0.225
    epsilon = 1e-5
    N = p.shape[0]
    data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
    pi_k, T, f = data['Pik20'], data['b_rgb'], data['f']
    p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
    flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
    q = p + flow_q.T
    disparity_q = bilinear_interpolate_numpy(disparity2, q, alpha_p, p)  # of shape (N, )
    q_3d = inverse_project(q, pi_k, disparity_q, T, f)
    p_homo = exp_map(motion).dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
    p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
    residual = p3d_t2 - q_3d
    # print(torch.sum(residual))
    E = ((np.sum(residual ** 2, axis=0) + epsilon ** 2) ** alpha) * alpha_p[p[:, 1], p[:, 0]]

    return E


masks = np.load("pmask_left_t0.npy")
flow = np.load("pwc_image_2.npy")
disparity1 = np.load("psm_1(1).npy")
disparity2 = np.load("psm_2(1).npy")

result = util.flow_to_color(flow.transpose(1, 2, 0))
plt.imshow(result)
plt.show()

flow_result = copy.deepcopy(flow)
for p in get_pointset(masks)[1:2]:
    alpha_p = np.ones_like(disparity1)
    motion_0 = np.ones(6)
    # motion_0 = np.array([0.1158, 0.2223, -0.4711, 3.7559, 3.7559, 3.7559])
    # motion_0 = np.array([ 0.44899609, 0.06546864, -0.49421121, 3.53099094, 3.28189738, 4.32275508])
    res = least_squares(cost_function, motion_0, args=(p, disparity1, disparity2, flow, alpha_p))
    motion = res.x
    print(res.cost)

    # calculate the instance-wise rigid flow estimation
    data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
    pi_k, T, f = data['Pik20'], data['b_rgb'], data['f']
    p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
    N = p_3d.shape[1]
    p_3d_t2 = exp_map(motion).dot(np.concatenate((p_3d, np.ones((1, N))), axis=0))
    p_3d_t2 = p_3d_t2 / p_3d_t2[3:4, :]
    p_homo = pi_k.dot(p_3d_t2)
    p_t2 = p_homo[:2, :] / p_homo[2:3, :]
    flow_rigid = p_t2 - p.T
    print(np.sum(np.abs(flow_result[:, p[:, 1], p[:, 0]] - flow_rigid))/N)
    print(flow_result[:, p[:, 1], p[:, 0]])
    print(flow_rigid)
    flow_result[:, p[:, 1], p[:, 0]] = flow_rigid

result = util.flow_to_color(flow_result.transpose(1, 2, 0))
plt.imshow(result)
plt.show()

print(motion)