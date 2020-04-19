from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from os.path import join
import cv2
import numpy as np
# from PIL import Image


class KITTI2015_stereo(Dataset):

    def __init__(self, directory, mode, validate_size=40, occ=True, transform=None):
        super().__init__()

        self.mode = mode
        self.transform = transform

        if mode == 'train' or mode == 'validate':
            self.dir = join(directory, 'training')
        elif mode == 'test':
            self.dir = join(directory, 'testing')

        left_dir = join(self.dir, 'image_2')
        right_dir = join(self.dir, 'image_3')
        left_imgs = list()
        right_imgs = list()

        if mode == 'train':
            imgs_range = range(200 - validate_size)
        elif mode == 'validate':
            imgs_range = range(200 - validate_size, 200)
        elif mode == 'test':
            imgs_range = range(200)

        fmt = '{:06}_10.png'

        for i in imgs_range:
            left_imgs.append(join(left_dir, fmt.format(i)))
            right_imgs.append(join(right_dir, fmt.format(i)))

        self.left_imgs = left_imgs
        self.right_imgs = right_imgs

        # self.disp_imgs = None
        if mode == 'train' or mode == 'validate':
            disp_imgs = list()
            if occ:
                disp_dir = join(self.dir, 'disp_occ_0')
            else:
                disp_dir = join(self.dir, 'disp_noc_0')
            disp_fmt = '{:06}_10.png'
            for i in imgs_range:
                disp_imgs.append(join(disp_dir, disp_fmt.format(i)))

            self.disp_imgs = disp_imgs

    def __len__(self):
        return len(self.left_imgs)

    def __getitem__(self, idx):
        data = {}

        # bgr mode
        data['left'] = cv2.imread(self.left_imgs[idx])
        data['right'] = cv2.imread(self.right_imgs[idx])
        if self.mode != 'test':
            data['disp'] = cv2.imread(self.disp_imgs[idx])[:, :, 0]

        if self.transform:
            data = self.transform(data)

        return data


class KITTI2015_flow(Dataset):

    def __init__(self, directory, mode, validate_size=40, occ=True, transform=None):
        super().__init__()

        self.mode = mode
        self.transform = transform

        if mode == 'train' or mode == 'validate':
            self.dir = join(directory, 'training')
        elif mode == 'test':
            self.dir = join(directory, 'testing')

        left_dir = join(self.dir, 'image_2')
        left_imgs_01 = list()
        left_imgs_11 = list()

        if mode == 'train':
            imgs_range = range(200 - validate_size)
        elif mode == 'validate':
            imgs_range = range(200 - validate_size, 200)
        elif mode == 'test':
            imgs_range = range(200)

        fmt_10 = '{:06}_10.png'
        fmt_11 = '{:06}_11.png'

        for i in imgs_range:
            left_imgs_01.append(join(left_dir, fmt_10.format(i)))
            left_imgs_11.append(join(left_dir, fmt_11.format(i)))
            # right_imgs.append(join(right_dir, fmt.format(i)))

        self.left_imgs_first = left_imgs_01
        self.left_imgs_second = left_imgs_11

        # self.disp_imgs = None
        if mode == 'train' or mode == 'validate':
            flow_imgs = list()
            if occ:
                flow_dir = join(self.dir, 'flow_occ')
            else:
                flow_dir = join(self.dir, 'flow_noc')
            flow_fmt = '{:06}_10.png'
            for i in imgs_range:
                flow_imgs.append(join(flow_dir, flow_fmt.format(i)))

            self.flow_imgs = flow_imgs

    def __len__(self):
        return len(self.left_imgs_first)

    def __getitem__(self, idx):
        data = {}

        # bgr mode, note: 'left' --> first, 'right' --> second, names to reuse transforms
        data['left'] = cv2.imread(self.left_imgs_first[idx])[:, :, ::-1].astype(np.float)  # RGB to BGR
        data['right'] = cv2.imread(self.left_imgs_second[idx])[:, :, ::-1].astype(np.float)  # RGB to BGR

        if self.transform:
            data = self.transform(data)

        if self.mode != 'test':
            flow = cv2.imread(self.flow_imgs[idx], cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float)
            # 20: the network was pre-trained with this factor (magic number:)
            data['flow'] = (flow[:, :, :2] - 2**15) / 64.0 / 20
            data['mask'] = flow[:, :, 2]

        return data


class RandomCrop():

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = self.output_size
        h, w, _ = sample['left'].shape
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for key in sample:
            sample[key] = sample[key][top: top + new_h, left: left + new_w]

        return sample


class Normalize():
    '''
    RGB mode
    '''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = sample['left'] / 255.0
        sample['right'] = sample['right'] / 255.0

        sample['left'] = self.__normalize(sample['left'])
        sample['right'] = self.__normalize(sample['right'])

        return sample

    def __normalize(self, img):
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img


class ToTensor():

    def __call__(self, sample):
        left = sample['left']
        right = sample['right']

        # H x W x C ---> C x H x W
        sample['left'] = torch.from_numpy(left.transpose([2, 0, 1])).type(torch.FloatTensor)
        sample['right'] = torch.from_numpy(right.transpose([2, 0, 1])).type(torch.FloatTensor)

        if 'disp' in sample:
            sample['disp'] = torch.from_numpy(sample['disp']).type(torch.FloatTensor)

        return sample


class Pad():
    def __init__(self, H, W):
        self.w = W
        self.h = H

    def __call__(self, sample):
        pad_h = self.h - sample['left'].size(1)
        pad_w = self.w - sample['left'].size(2)

        left = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        left = F.pad(left, pad=(0, pad_w, 0, pad_h))
        right = sample['right'].unsqueeze(0)  # [1, 3, H, W]
        right = F.pad(right, pad=(0, pad_w, 0, pad_h))

        if 'disp' in sample:
            disp = sample['disp'].unsqueeze(0)  # [1, 2, H, W]
            disp = F.pad(disp, pad=(0, pad_w, 0, pad_h))
        if 'flow' in sample:
            flow = sample['flow'].unsqueeze(0).unsqueeze(1)
            flow = F.pad(flow, pad=(0, pad_w, 0, pad_h))
            mask = sample['mask'].unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
            mask = F.pad(mask, pad=(0, pad_w, 0, pad_h))

        sample['left'] = left.squeeze()
        sample['right'] = right.squeeze()
        if 'disp' in sample:
            sample['disp'] = disp.squeeze()
        if 'flow' in sample:
            sample['mask'] = mask.squeeze()
            sample['flow'] = flow.squeeze()

        return sample


if __name__ == '__main__':
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    # BGR
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    train_transform = T.Compose([RandomCrop([256, 512]), ToTensor()])
    train_dataset = KITTI2015_stereo('D:/dataset/data_scene_flow', mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset)
    print(len(train_loader))

    # test_transform = T.Compose([ToTensor()])
    # test_dataset = KITTI2015_stereo('D:/dataset/data_scene_flow', mode='test', transform=test_transform)

    # validate_transform = T.Compose([ToTensor()])
    # validate_dataset = KITTI2015_stereo('D:/dataset/data_scene_flow', mode='validate', transform=validate_transform)

    # datasets = [train_dataset, test_dataset, validate_dataset]

    # for i, dataset in enumerate(datasets):
    #     a = dataset[0]['right'].numpy().transpose([1, 2, 0])
    #     plt.subplot(3, 1, i + 1)
    #     plt.imshow(a)
    # plt.show()
