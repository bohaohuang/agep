"""
Download data from: https://susanqq.github.io/UTKFace/
Unzip them, remove the space in part3/24_0_1_20170116220224657 .jpg 24
The labels is embedded in the file name formatted as:
[age]_[gender]_[race]_[date&time].jpg
"""


# Built-in
import os
from glob import glob

# Libs
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import io, color, transform

# Pytorch
import torch
import torchvision.transforms as transforms
from torch.utils import data

# Settings
RANDOM_SEED = 0


def make_dataset_hdf5(data_dir, valid_percent=0.2):
    # make files
    hdf5_train = h5py.File(os.path.join(data_dir, 'train.hdf5'), mode='w')
    hdf5_valid = h5py.File(os.path.join(data_dir, 'valid.hdf5'), mode='w')

    # read all image files
    img_files = []
    for idx in range(1, 4):
        img_files.extend(glob(os.path.join(data_dir, 'part{}'.format(idx), '*.jpg')))

    # shuffle and get valid data
    valid_num = int(np.floor(len(img_files) * valid_percent))
    np.random.seed(RANDOM_SEED)
    img_files = np.random.permutation(img_files)
    valid_files = sorted(img_files[:valid_num])
    train_files = sorted(img_files[valid_num:])

    hdf5_train.create_dataset('img', (len(train_files), 256, 256, 3), np.uint8)
    hdf5_train.create_dataset('age', (len(train_files), ), np.int)
    hdf5_valid.create_dataset('img', (len(valid_files), 256, 256, 3), np.uint8)
    hdf5_valid.create_dataset('age', (len(valid_files),), np.int)

    # log entries
    for cnt, f in enumerate(tqdm(valid_files, desc='Prepare valid set')):
        img = io.imread(f)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        age = int(os.path.splitext(os.path.basename(f))[0].split('_')[0])
        hdf5_valid['img'][cnt, ...] = transform.resize(img, (256, 256), anti_aliasing=True,
                                                       preserve_range=True)[..., :3]
        hdf5_valid['age'][cnt, ...] = age
    for cnt, f in enumerate(tqdm(train_files, desc='Prepare train set')):
        img = io.imread(f)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        age = int(os.path.splitext(os.path.basename(f))[0].split('_')[0])
        hdf5_train['img'][cnt, ...] = transform.resize(img, (256, 256), anti_aliasing=True,
                                                       preserve_range=True)[..., :3]
        hdf5_train['age'][cnt, ...] = age

    hdf5_train.close()
    hdf5_valid.close()


def make_dataset_folder(data_dir, age_thresh=6, valid_percent=0.2):
    np.random.seed(RANDOM_SEED)
    img_files = sorted(glob(os.path.join(data_dir, '*.chip.jpg')))
    h0_files, h1_files = [], []
    for img_file in img_files:
        age = int(os.path.splitext(os.path.basename(img_file))[0].split('_')[0])
        if age <= age_thresh:
            h1_files.append(img_file)
        else:
            h0_files.append(img_file)
    # shuffle and get valid data
    h0_files, h1_files = np.random.permutation(h0_files), np.random.permutation(h1_files)
    train_files, valid_files = [], []
    valid_num_h0, valid_num_h1 = int(np.floor(len(h0_files) * valid_percent)), int(np.floor(len(h1_files) * valid_percent))
    valid_files.extend(h0_files[:valid_num_h0])
    valid_files.extend(h1_files[:valid_num_h1])
    train_files.extend(h0_files[valid_num_h0:])
    train_files.extend(h1_files[valid_num_h1:])

    save_dir_train, save_dir_valid = os.path.join(data_dir, 'train'), os.path.join(data_dir, 'valid')
    if not os.path.exists(save_dir_train):
        os.makedirs(os.path.join(save_dir_train, '0'))
        os.makedirs(os.path.join(save_dir_train, '1'))
    if not os.path.exists(save_dir_valid):
        os.makedirs(os.path.join(save_dir_valid, '0'))
        os.makedirs(os.path.join(save_dir_valid, '1'))

    for cnt, f in enumerate(tqdm(valid_files, desc='Prepare valid set')):
        img = io.imread(f)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        img = transform.resize(img, (112, 112), anti_aliasing=True, preserve_range=True)[..., :3]
        age = int(os.path.splitext(os.path.basename(f))[0].split('_')[0])
        if age <= age_thresh:
            io.imsave(os.path.join(save_dir_valid, '1', '{:05d}.jpg'.format(cnt)), img)
        else:
            io.imsave(os.path.join(save_dir_valid, '0', '{:05d}.jpg'.format(cnt)), img)
    for cnt, f in enumerate(tqdm(train_files, desc='Prepare train set')):
        img = io.imread(f)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        img = transform.resize(img, (112, 112), anti_aliasing=True, preserve_range=True)[..., :3]
        age = int(os.path.splitext(os.path.basename(f))[0].split('_')[0])
        if age <= age_thresh:
            io.imsave(os.path.join(save_dir_train, '1', '{:05d}.jpg'.format(cnt)), img)
        else:
            io.imsave(os.path.join(save_dir_train, '0', '{:05d}.jpg'.format(cnt)), img)


def make_dataset_folder_multiclass(data_dir, age_thresh=6, valid_percent=0.2):
    np.random.seed(RANDOM_SEED)
    img_files = sorted(glob(os.path.join(data_dir, '*.chip.jpg')))
    # shuffle and get valid data
    img_files = np.random.permutation(img_files)
    train_files, valid_files = [], []
    valid_num = int(np.floor(len(img_files) * valid_percent))
    valid_files.extend(img_files[:valid_num])
    train_files.extend(img_files[valid_num:])

    save_dir_train, save_dir_valid = os.path.join(data_dir, 'train'), os.path.join(data_dir, 'valid')
    if not os.path.exists(save_dir_train):
        for i in range(5):
            os.makedirs(os.path.join(save_dir_train, '{}'.format(i)))
    if not os.path.exists(save_dir_valid):
        for i in range(5):
            os.makedirs(os.path.join(save_dir_valid, '{}'.format(i)))

    for cnt, f in enumerate(tqdm(valid_files, desc='Prepare valid set')):
        img = io.imread(f)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        img = transform.resize(img, (112, 112), anti_aliasing=True, preserve_range=True)[..., :3]
        age = int(os.path.splitext(os.path.basename(f))[0].split('_')[0])
        if age <= age_thresh:
            io.imsave(os.path.join(save_dir_valid, '1', '{:05d}.jpg'.format(cnt)), img)
        else:
            io.imsave(os.path.join(save_dir_valid, '0', '{:05d}.jpg'.format(cnt)), img)
    for cnt, f in enumerate(tqdm(train_files, desc='Prepare train set')):
        img = io.imread(f)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        img = transform.resize(img, (112, 112), anti_aliasing=True, preserve_range=True)[..., :3]
        age = int(os.path.splitext(os.path.basename(f))[0].split('_')[0])
        if age <= age_thresh:
            io.imsave(os.path.join(save_dir_train, '1', '{:05d}.jpg'.format(cnt)), img)
        else:
            io.imsave(os.path.join(save_dir_train, '0', '{:05d}.jpg'.format(cnt)), img)


class UTKDataLoader(data.Dataset):
    def __init__(self, parent_path, file_list, transforms=None, age_thresh=6, reorder=False):
        """
        A data reader for the UTKFace dataset
        The dataset storage structure should be like
        /parent_path
            /patches
                img0.png
                img1.png
            file_list.txt
        :param parent_path: path to the zipped UTKFace dataset
        :param file_list: a text file where each row contains path to image file and age separated by space
        :param transforms: transforms
        :param age_thresh: any age below this threshold will be considered as class 1
        """
        self.file_path = os.path.join(parent_path, file_list)
        self.transforms = transforms
        self.age_thresh = age_thresh
        self.dataset = None
        self.class_cnt = [0, 0]
        class_0_ids, class_1_ids = [], []
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = file['img'].shape[0]
            for i in range(self.dataset_len):
                if file['age'][i] < self.age_thresh:
                    self.class_cnt[1] += 1
                    class_1_ids.append(i)
                else:
                    self.class_cnt[0] += 1
                    class_0_ids.append(i)
        self.reorder = reorder
        self.idx = np.arange(self.dataset_len)
        if self.reorder:
            self.idx = np.arange(self.dataset_len)
            add_idx = np.random.choice(class_1_ids, self.class_cnt[0]-self.class_cnt[1])
            self.idx = np.concatenate([self.idx, add_idx])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        if not self.dataset:
            self.dataset = h5py.File(self.file_path, 'r')
        img = self.dataset['img'][self.idx[index], ...]
        age = self.dataset['age'][self.idx[index]]
        age = 1 if int(age) < self.age_thresh else 0
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(age)


def vis_data(img, lbl, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255),
             im_grid=(2, 5)):
    mean = [-a / b for a, b in zip(mean, std)]
    std = [1 / a for a in std]
    inv_normalize = transforms.Normalize(
        mean=mean,
        std=std
    )
    for i in range(img.shape[0]):
        img[i] = inv_normalize(img[i])
    img, lbl = img.numpy(), lbl.numpy()
    img = np.rollaxis(img, 1, 4)
    plt.figure(figsize=(12, 5))
    for cnt, (i, l) in enumerate(zip(img, lbl)):
        plt.subplot(*im_grid, cnt+1)
        plt.imshow(i)
        plt.axis('off')
        plt.title(l)
        if cnt == im_grid[0] * im_grid[1] - 1:
            break
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # make_dataset_hdf5(r'./data')

    '''dr = UTKDataLoader(r'./data', 'train.hdf5')
    for img, lbl in dr:
        lbl = lbl.cpu().numpy()
        if lbl == 1:
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()'''

    make_dataset_folder(r'./data/UTKFace')
