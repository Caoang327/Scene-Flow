import cv2
import torch
import numpy as np
import util
import matplotlib.pyplot as plt
import copy
from scipy.optimize import least_squares

device = "cpu"
dtype = torch.FloatTensor

def exp_map(motion):
    """
    Map the rigid motion from se(3) to SE(3)

    Inputs:
    - motion: rigid motion representation in se(3), tensor of shape (6, 1)
    
    Outputs:
    - T: the corresponding transform matrix in SE(3), tensor of shape (4, 4)
    """

    u = motion[:3]
    w = motion[3:]
    theta = torch.sqrt(w.t().matmul(w))
    A = torch.sin(theta) / theta
    B = (1 - torch.cos(theta)) / theta**2
    C = (1 - A) / theta**2
    w_x = torch.tensor([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]]).to(device)
    R = torch.eye(3).to(device) + A * w_x + B * w_x.matmul(w_x)
    V = torch.eye(3).to(device) + B * w_x + C * w_x.matmul(w_x)
    T = torch.zeros((4,4)).to(device)
    T[:3, :3] = R
    T[:3, 3:4] = V.matmul(u)
    T[3, 3] = 1

    return T


def get_pointset(masks):
    """
    Get the instance points set from the masks obtained from mask-rcnn

    Inputs:
    - mask: masks obtained from mask-rcnn, tensor of shape (N, m, n)

    Outputs:
    - p_set: list of the point set of all instances, list<tensor of shape (N, 2)>
    """
    p_set = []
    for mask in masks:
        points = mask.nonzero()
        points = torch.index_select(points, 1, torch.LongTensor([1, 0]))
        p_set.append(points)

    return p_set


def inverse_project(p, pi_k, d, T, f):
    """
    Get 3d point in the world coordinate from pixel and disparity

    Inputs:
    - p: image coordinates of the points, tensor of shape (N, 2)
    - pi_k: perspective project function, tensor of shape (3, 4)
    - d: the corresponding disparity, tensor of shape(N,)
    - T: length of baseline, scalar
    - f: focal length, scalar

    Outputs:
    - p_3d: 3d point in the world coordinate, tensor of shape (3, N)
    """
    depth = f * T / d  # of shape (N, 1)
    N = p.shape[0]
    p_homo = torch.pinverse(pi_k).matmul(torch.cat((p.t().to(pi_k), torch.ones(1, N).to(pi_k)), 0)) * depth.view(1, N).to(pi_k)
    p_3d = p_homo[:3, :] / p_homo[3:4, :]

    return p_3d


def bilinear_interpolate_torch(im, q, alpha_p, p):
    """
    Implement the bilinear interpolation
    revised version
    original codes: https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

    Inputs:
    - im: the value matrix, can be img/disparity, tensor of shape (m, n)
    - q: the point set at time t+1, tensor of shape (N, 2)
    - alpha_p: indicator, tensor of shape (m, n)
    - p: the point set at time t, tensor of shape (N, 2)

    Outputs:
    - result: of shape (N,)
    """
    dtype = torch.FloatTensor
    # dtype_long = torch.FloatTensor
    dtype_long = torch.LongTensor

    x = q[:, 0]
    y = q[:, 1]

    x0 = torch.floor(x).type(dtype_long).to(device)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long).to(device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type(dtype).to(device) - x) * (y1.type(dtype).to(device) - y)
    wb = (x1.type(dtype).to(device) - x) * (y - y0.type(dtype).to(device))
    wc = (x - x0.type(dtype).to(device)) * (y1.type(dtype).to(device) - y)
    wd = (x - x0.type(dtype).to(device)) * (y - y0.type(dtype).to(device))

    result = Ia * wa + Ib * wb + Ic * wc + Id * wd
    # print(p[result==0])
    alpha_p[p[result==0, 1], p[result==0, 0]] = 0
    result[result==0] = 1

    return result
    # return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)


def cost_function(motion, p, disparity1, disparity2, flow, alpha_p):
    """
    Calculate the E_rigid

    Inputs:
    - motion: rigid motion representation in se(3), tensor of shape (6, 1)
    - p: point set of one instance in 1st img, tensor of shape (N, 2)
    - disparity1: disparity map of time t, tensor of shape (m, n)
    - disparity2: disparity map of time t+1, tensor of shape (m, n)
    - flow: flow matrix of 1st img, tensor of shape (2, m, n)
    - alpha_p: indicators representing if it is a outlier, tensor of shape (m, n)

    Outputs:
    - E: value of E_rigid, scalar
    """
    alpha = 0.45
    epsilon = 1e-5
    N = p.shape[0]
    data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
    pi_k, T, f = torch.from_numpy(data['Pik20']).type(dtype).to(device), data['b_rgb'], data['f']
    p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
    flow_q = flow[:, p[:, 1], p[:, 0]]  # of shape (2, N)
    q = p.to(flow_q) + flow_q.t()
    # q = p.to(flow_q) + torch.cat((flow_q.t()[:, 1:2], flow_q.t()[:, 0:1]), 1)
    disparity_q = bilinear_interpolate_torch(disparity2, q, alpha_p, p)  # of shape (N, )
    q_3d = inverse_project(q, pi_k, disparity_q, T, f)
    p_homo = exp_map(motion).matmul(torch.cat((p_3d, torch.ones(1, N).to(p_3d)), 0))
    p3d_t2 = p_homo[:3, :] / p_homo[3:4, :]
    residual = p3d_t2 - q_3d
    # print(torch.sum(residual))
    E = torch.sum(((torch.sum(residual**2, dim=0) + epsilon**2)**alpha) * alpha_p[p[:, 1], p[:, 0]])

    return E


masks = torch.from_numpy(np.load("pmask_left_t0.npy").astype(np.uint8)).type(dtype).to(device)
flow = torch.from_numpy(np.load("pwc_image_2.npy")).type(dtype).to(device)
disparity1 = torch.from_numpy(np.load("psm_1(1).npy")).type(dtype).to(device)
disparity2 = torch.from_numpy(np.load("psm_2(1).npy")).type(dtype).to(device)

result = util.flow_to_color(flow.numpy().transpose(1, 2, 0))
plt.imshow(result)
plt.show()

flow_result = copy.deepcopy(flow)
for p in get_pointset(masks)[1:2]:
    alpha_p = torch.ones_like(disparity1)
    motion = (2 * torch.ones(6, 1).to(device)).requires_grad_()
    learning_rate = 1e-6
    # weight_decay = 1e-5
    optimizer = torch.optim.SGD([motion], lr=learning_rate)
    num_iters = 2000

    # res = least_squares(fx, motion_0, args=(p, disparity1, disparity2, flow, alpha_p))
    # motion = res.x
    # motion = torch.from_numpy(motion)
    # print(res.cost)
    # print(res.grad)

    for i in range(num_iters):
        loss = cost_function(motion, p, disparity1, disparity2, flow, alpha_p)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

    # calculate the instance-wise rigid flow estimation
    data = util.load_calib_cam_to_cam("velo_to_cam000010.txt", "./cam_to_cam000010.txt")
    pi_k, T, f = torch.from_numpy(data['Pik20']).type(dtype).to(device), data['b_rgb'], data['f']
    p_3d = inverse_project(p, pi_k, disparity1[p[:, 1], p[:, 0]], T, f)  # of shape (3, N)
    N = p_3d.shape[1]
    p_3d_t2 = exp_map(motion).matmul(torch.cat((p_3d, torch.ones(1, N)), 0))

    p_3d_t2 = p_3d_t2 / p_3d_t2[3:4, :]
    p_homo = pi_k.matmul(p_3d_t2)
    p_t2 = p_homo[:2, :] / p_homo[2:3, :]
    flow_rigid = p_t2 - p.to(p_t2).t()
    print(torch.sum(torch.abs(flow_result[:, p[:, 1], p[:, 0]] - flow_rigid))/N)
    flow_result[:, p[:, 1], p[:, 0]] = flow_rigid

result = util.flow_to_color(flow_result.detach().numpy().transpose(1, 2, 0))
plt.imshow(result)
plt.show()

print(torch.sum(alpha_p))
print(motion)
