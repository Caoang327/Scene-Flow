import argparse
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
# from models.PSMnet import PSMNet
# from models.smoothloss import SmoothL1Loss
from utils.KITTI2015_loader import KITTI2015_stereo, RandomCrop, ToTensor, Normalize, Pad
import math

import tensorboardX as tX

import matplotlib
import matplotlib.pyplot as plt
import gc
import cv2


# Arguments & Hyper-parameters
class myargs:
    def __init__(self):
        self.maxdisp = 192
        self.logdir = "log/runs"
        self.datadir = "."
        self.cuda = 0
        self.batch_size = 2
        self.validate_batch_size = 2
        self.log_per_step = 20
        self.save_per_epoch = 5
        self.model_dir = "/content/drive/My Drive/EECS 504/project proposal ideas/PSMNet_Checkpoints"
        self.model_path = None
        self.lr = 0.001
        self.num_epochs = 300
        self.num_workers = 1


args = myargs()

# Constants

# image net
mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]

device_ids = [0]
writer = tX.SummaryWriter(log_dir=args.logdir, comment='FSMNet')
device = torch.device('cuda')
print(device)

"""## Networks

### CostNet

#### Part 1. CNN Block
"""


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, downsample=False):
        super().__init__()

        self.net = nn.Sequential(
            Conv2dBn(in_channels, out_channels, kernel_size, stride, padding, dilation, use_relu=True),
            Conv2dBn(out_channels, out_channels, kernel_size, 1, padding, dilation, use_relu=False)
        )

        self.downsample = None
        if downsample:
            self.downsample = Conv2dBn(in_channels, out_channels, 1, stride, use_relu=False)

    def forward(self, inputs):
        out = self.net(inputs)
        if self.downsample:
            inputs = self.downsample(inputs)
        out = out + inputs

        return out


class StackedBlocks(nn.Module):

    def __init__(self, n_blocks, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()

        if stride == 1 and in_channels == out_channels:
            downsample = False
        else:
            downsample = True
        net = [ResidualBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, downsample)]

        for i in range(n_blocks - 1):
            net.append(ResidualBlock(out_channels, out_channels, kernel_size, 1, padding, dilation, downsample=False))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
            Conv2dBn(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, use_relu=True),  # downsample
            Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True),
            Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)
        )

        self.conv1 = StackedBlocks(n_blocks=3, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                                   dilation=1)
        self.conv2 = StackedBlocks(n_blocks=16, in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,
                                   dilation=1)  # downsample
        self.conv3 = StackedBlocks(n_blocks=3, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2,
                                   dilation=2)  # dilated
        self.conv4 = StackedBlocks(n_blocks=3, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4,
                                   dilation=4)  # dilated

    def forward(self, inputs):
        conv0_out = self.conv0(inputs)
        conv1_out = self.conv1(conv0_out)  # [B, 32, 1/2H, 1/2W]
        conv2_out = self.conv2(conv1_out)  # [B, 64, 1/4H, 1/4W]
        conv3_out = self.conv3(conv2_out)  # [B, 128, 1/4H, 1/4W]
        conv4_out = self.conv4(conv3_out)  # [B, 128, 1/4H, 1/4W]

        return conv2_out, conv4_out


"""#### Part 2. SPP Module"""


class Conv2dBn(nn.Module):
    """ Conv Block: Conv2d + BatchedNorm2d + (optional)ReLu """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm2d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class SPP(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch1 = self.__make_branch(kernel_size=64, stride=64)
        self.branch2 = self.__make_branch(kernel_size=32, stride=32)
        self.branch3 = self.__make_branch(kernel_size=16, stride=16)
        self.branch4 = self.__make_branch(kernel_size=8, stride=8)

    def forward(self, inputs):
        out_size = inputs.size(2), inputs.size(3)
        branch1_out = F.upsample(self.branch1(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        # print('branch1_out')
        # print(branch1_out[0, 0, :3, :3])
        branch2_out = F.upsample(self.branch2(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch3_out = F.upsample(self.branch3(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch4_out = F.upsample(self.branch4(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        out = torch.cat([branch4_out, branch3_out, branch2_out, branch1_out], dim=1)  # [B, 128, 1/4H, 1/4W]

        return out

    @staticmethod
    def __make_branch(kernel_size, stride):
        branch = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride),
            Conv2dBn(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)
            # kernel size maybe 1
        )
        return branch


"""#### Part 3. CostNet"""


class CostNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = CNN()
        self.spp = SPP()
        self.fusion = nn.Sequential(
            Conv2dBn(in_channels=320, out_channels=128, kernel_size=3, stride=1, padding=1, use_relu=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, inputs):
        conv2_out, conv4_out = self.cnn(inputs)  # [B, 64, 1/4H, 1/4W], [B, 128, 1/4H, 1/4W]

        spp_out = self.spp(conv4_out)  # [B, 128, 1/4H, 1/4W]
        out = torch.cat([conv2_out, conv4_out, spp_out], dim=1)  # [B, 320, 1/4H, 1/4W]
        out = self.fusion(out)  # [B, 32, 1/4H, 1/4W]

        return out


"""### Stacked Hourglass

#### Part 1. Hourglass
"""


class Conv3dBn(nn.Module):
    """ Conv Block: Conv3d + BatchNorm + (optional)ReLu"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm3d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class Hourglass(nn.Module):

    def __init__(self):
        super().__init__()

        self.net1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.net2 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.net3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.BatchNorm3d(num_features=64)
            # nn.ReLU(inplace=True)
        )
        self.net4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1,
                               bias=False),
            nn.BatchNorm3d(num_features=32)
        )

    def forward(self, inputs, scale1=None, scale2=None, scale3=None):
        net1_out = self.net1(inputs)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale1 is not None:
            net1_out = F.relu(net1_out + scale1, inplace=True)
        else:
            net1_out = F.relu(net1_out, inplace=True)

        net2_out = self.net2(net1_out)  # [B, 64, 1/16D, 1/16H, 1/16W]
        net3_out = self.net3(net2_out)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale2 is not None:
            net3_out = F.relu(net3_out + scale2, inplace=True)
        else:
            net3_out = F.relu(net3_out + net1_out, inplace=True)

        net4_out = self.net4(net3_out)

        if scale3 is not None:
            net4_out = net4_out + scale3

        return net1_out, net3_out, net4_out


"""#### Part 2. Helper: DisparityRegression"""


class DisparityRegression(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.disp_score = torch.range(0, max_disp - 1)  # [D]
        self.disp_score = self.disp_score.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, D, 1, 1]

    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        out = torch.sum(disp_score * prob, dim=1)  # [B, H, W]
        return out


"""#### Part 3. StackedHourglass"""


class StackedHourglass(nn.Module):
    '''
    inputs --- [B, 64, 1/4D, 1/4H, 1/4W]
    '''

    def __init__(self, max_disp):
        super().__init__()

        self.conv0 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.conv1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.hourglass1 = Hourglass()
        self.hourglass2 = Hourglass()
        self.hourglass3 = Hourglass()

        self.out1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out2 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out3 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.regression = DisparityRegression(max_disp)

    def forward(self, inputs, out_size):
        conv0_out = self.conv0(inputs)  # [B, 32, 1/4D, 1/4H, 1/4W]
        conv1_out = self.conv1(conv0_out)
        conv1_out = conv0_out + conv1_out  # [B, 32, 1/4D, 1/4H, 1/4W]

        hourglass1_out1, hourglass1_out3, hourglass1_out4 = self.hourglass1(conv1_out, scale1=None, scale2=None,
                                                                            scale3=conv1_out)
        hourglass2_out1, hourglass2_out3, hourglass2_out4 = self.hourglass2(hourglass1_out4, scale1=hourglass1_out3,
                                                                            scale2=hourglass1_out1, scale3=conv1_out)
        hourglass3_out1, hourglass3_out3, hourglass3_out4 = self.hourglass3(hourglass2_out4, scale1=hourglass2_out3,
                                                                            scale2=hourglass1_out1, scale3=conv1_out)

        out1 = self.out1(hourglass1_out4)  # [B, 1, 1/4D, 1/4H, 1/4W]
        out2 = self.out2(hourglass2_out4) + out1
        out3 = self.out3(hourglass3_out4) + out2

        cost1 = F.upsample(out1, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost2 = F.upsample(out2, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost3 = F.upsample(out3, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]

        prob1 = F.softmax(-cost1, dim=1)  # [B, D, H, W]
        prob2 = F.softmax(-cost2, dim=1)
        prob3 = F.softmax(-cost3, dim=1)

        disp1 = self.regression(prob1)
        disp2 = self.regression(prob2)
        disp3 = self.regression(prob3)

        return disp1, disp2, disp3


"""### PSM Net"""


# PSM Net Model
class PSMNet(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.cost_net = CostNet()
        self.stackedhourglass = StackedHourglass(max_disp)
        self.D = max_disp

        self.__init_params()

    def forward(self, left_img, right_img):
        original_size = [self.D, left_img.size(2), left_img.size(3)]

        left_cost = self.cost_net(left_img)  # [B, 32, 1/4H, 1/4W]
        right_cost = self.cost_net(right_img)  # [B, 32, 1/4H, 1/4W]
        # cost = torch.cat([left_cost, right_cost], dim=1)  # [B, 64, 1/4H, 1/4W]
        # B, C, H, W = cost.size()

        # print('left_cost')
        # print(left_cost[0, 0, :3, :3])

        B, C, H, W = left_cost.size()

        cost_volume = torch.zeros(B, C * 2, self.D // 4, H, W).type_as(left_cost)  # [B, 64, D, 1/4H, 1/4W]

        # for i in range(self.D // 4):
        #     cost_volume[:, :, i, :, i:] = cost[:, :, :, i:]

        for i in range(self.D // 4):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_cost[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_cost[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_cost
                cost_volume[:, C:, i, :, :] = right_cost

        disp1, disp2, disp3 = self.stackedhourglass(cost_volume, out_size=original_size)

        return disp1, disp2, disp3

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


"""### Loss Function"""


# L1 Loss
class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, disp1, disp2, disp3, target):
        loss1 = F.smooth_l1_loss(disp1, target)
        loss2 = F.smooth_l1_loss(disp2, target)
        loss3 = F.smooth_l1_loss(disp3, target)

        return loss1, loss2, loss3


"""Training, Validation and Inference"""


def validate(model, validate_loader, epoch):
    '''
    validate 40 image pairs
    '''
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)

    avg_error = 0.0

    left_save = None
    disp_save = None

    for i, batch in enumerate(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        with torch.no_grad():
            _, _, disp = model(left_img, right_img)

        delta = torch.abs(disp[mask] - target_disp[mask])
        cr1 = (delta >= 3.0).int()
        cr2 = (delta >= 0.05 * (target_disp[mask])).int()
        error_mat = ((cr1 + cr2) == 2)
        # error_mat = (((delta >= 3.0) + (delta >= 0.05 * (target_disp[mask]))) == 2)
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100

        avg_error += error
        if i == idx:
            left_save = left_img
            disp_save = disp

    avg_error = avg_error / num_batches
    print('epoch: {:03} | 3px-error: {:.5}%'.format(epoch, avg_error))
    writer.add_scalar('error/3px', avg_error, epoch)
    save_image(left_save[0], disp_save[0], epoch)

    return avg_error


def save_image(left_image, disp, epoch):
    for i in range(3):
        left_image[i] = left_image[i] * std[i] + mean[i]
    b, r = left_image[0], left_image[2]
    left_image[0] = r  # BGR --> RGB
    left_image[2] = b
    # left_image = torch.from_numpy(left_image.cpu().numpy()[::-1])

    disp_img = disp.detach().cpu().numpy()
    fig = plt.figure(figsize=(12.84, 3.84))
    plt.axis('off')  # hide axis
    plt.imshow(disp_img)
    plt.colorbar()

    writer.add_figure('image/disp', fig, global_step=epoch)
    writer.add_image('image/left', left_image, global_step=epoch)


def train(model, train_loader, optimizer, criterion, step):
    """
    train one epoch
    """
    for batch in train_loader:
        step += 1
        optimizer.zero_grad()

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        disp1, disp2, disp3 = model(left_img, right_img)
        loss1, loss2, loss3 = criterion(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
        total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

        total_loss.backward()
        optimizer.step()

        # print(step)

        if step % args.log_per_step == 0:
            writer.add_scalar('loss/loss1', loss1, step)
            writer.add_scalar('loss/loss2', loss2, step)
            writer.add_scalar('loss/loss3', loss3, step)
            writer.add_scalar('loss/total_loss', total_loss, step)
            print('step: {:05} | total loss: {:.5} | loss1: {:.5} | loss2: {:.5} | loss3: {:.5}'.format(step,
                                                                                                        total_loss.item(),
                                                                                                        loss1.item(),
                                                                                                        loss2.item(),
                                                                                                        loss3.item()))

    return step


def adjust_lr(optimizer, epoch):
    if epoch == 200:
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save(model, optimizer, epoch, step, error, best_error):
    path = os.path.join(args.model_dir, '{:03}.ckpt'.format(epoch))
    # torch.save(model.state_dict(), path)
    # model.save_state_dict(path)

    state = {}
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['error'] = error
    state['epoch'] = epoch
    state['step'] = step

    torch.save(state, path)
    print('save model at epoch{}'.format(epoch))

    if error < best_error:
        best_error = error
        best_path = os.path.join(args.model_dir, 'best_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_path)
        print('best model in epoch {}'.format(epoch))

    return best_error


def load_model(model, path):
    state = torch.load(path)
    # assert: device_ids == 1
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state['state_dict'].items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    state['state_dict'] = new_state_dict
    model.load_state_dict(state['state_dict'])

    # Logs
    print('load model from {}'.format(path))
    print('epoch: {}'.format(state['epoch']))
    print('3px-error: {}%'.format(state['error']))

    return model


def inference(model, path_img1, path_img2):
    left = cv2.imread(path_img1)
    right = cv2.imread(path_img2)
    disp = np.zeros_like(left)  # functions as a placeholder: necessary for Pad
    # W, H = left.shape

    pairs = {'left': left, 'right': right, 'disp': disp}
    transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    # transform = T.Compose([Normalize(mean, std), ToTensor()])
    pairs = transform(pairs)
    left = pairs['left'].to(device).unsqueeze(0)
    right = pairs['right'].to(device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        _, _, disp = model(left, right)

    disp = disp.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(12.84, 3.84))
    plt.axis('off')
    plt.imshow(disp)
    plt.colorbar()
    plt.show()
    # plt.savefig(args.save_path, dpi=100)
    return disp

    # print('save diparity map in {}'.format(args.save_path))


if __name__ is "__main__":
    model = PSMNet(args.maxdisp).to(device)
    path = "/content/PSMNet/best_model.ckpt"
    model = load_model(model, path)

    path_left = "/content/PSMNet/training/image_2/000000_10.png"
    path_right = "/content/PSMNet/training/image_3/000000_10.png"
    disp = inference(model, path_left, path_right)
    np.save("psm_1.npy", disp)
