import logging
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.io as sio
import h5py
import os


def load_image(filename, flag='phi_unwrapped'):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    # 读取mat格式文件
    elif ext == '.mat':
        try:
            img_mat = sio.loadmat(filename)[flag]
            img_mat = np.array(img_mat)
        except:
            img_mat = h5py.File(filename, 'r')[flag] 
            img_mat = np.array(img_mat)# 读取mat文件中的phi_unwrapped numpy array W * H
            img_mat = img_mat.transpose(1, 0)
            # H * W 
        img_mat = Image.fromarray(img_mat) # 转换为PIL Image .size = (W, H)
        return img_mat
    else:
        # return Image.open(filename).convert('RGB')
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', img_use_path=None):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_dir_fenzi = os.path.join(mask_dir, 'fenzi_GT_mat')
        self.mask_dir_fenmu = os.path.join(mask_dir, 'fenmu_GT_mat')
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        # get img_num, set train_set test_set
        if img_use_path:
            img_use = [str(i) for i in list(np.load(img_use_path))]
            self.sub_dirs = [d for d in os.listdir(images_dir) 
                            if os.path.isdir(os.path.join(images_dir, d)) 
                            and not d.startswith('.')
                            and d in img_use]
        else:
            self.sub_dirs = [d for d in os.listdir(images_dir) 
                            if os.path.isdir(os.path.join(images_dir, d)) 
                            and not d.startswith('.')]
        # import pdb;pdb.set_trace()
                            
        if not self.sub_dirs:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.sub_dirs)} examples')
        logging.info('Scanning mask files to determine unique values')


    def __len__(self):
        return len(self.sub_dirs)

    @staticmethod 
    # flag 表示是图还是mat的mask，如果是mask则不应该归一化, 1表示是图。 0则是mat
    def preprocess(flag, pil_img, scale, is_mask):
        w, h = pil_img.size
        if scale != 1:
            assert scale == 1, 'Scale is not implemented yet'
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if flag:
            if (img > 1).any():
                img = img / 255.0

        return img

    def __getitem__(self, idx):
        name = self.sub_dirs[idx]
        # imgs
        img_path_1 = os.path.join(self.images_dir, name, f'{name}_1.bmp')
        img_path_2 = os.path.join(self.images_dir, name, f'{name}_2.bmp')
        img_path_3 = os.path.join(self.images_dir, name, f'{name}_3.bmp')
        img_path_4 = os.path.join(self.images_dir, name, f'{name}_4.bmp')
        # masks fenzi
        fenzi_mask_path_1 = os.path.join(self.mask_dir_fenzi, f'{name}-1.mat')
        fenzi_mask_path_2 = os.path.join(self.mask_dir_fenzi, f'{name}-2.mat')
        fenzi_mask_path_3 = os.path.join(self.mask_dir_fenzi, f'{name}-3.mat')
        fenzi_mask_path_4 = os.path.join(self.mask_dir_fenzi, f'{name}-4.mat')
        # mask fenmu
        fenmu_mask_path_1 = os.path.join(self.mask_dir_fenmu, f'{name}-1.mat')
        fenmu_mask_path_2 = os.path.join(self.mask_dir_fenmu, f'{name}-2.mat')
        fenmu_mask_path_3 = os.path.join(self.mask_dir_fenmu, f'{name}-3.mat')
        fenmu_mask_path_4 = os.path.join(self.mask_dir_fenmu, f'{name}-4.mat')

        img_1 = load_image(img_path_1)
        img_2 = load_image(img_path_2)
        img_3 = load_image(img_path_3)
        img_4 = load_image(img_path_4)

        fenzi_mask_1 = load_image(fenzi_mask_path_1, 'numerator')
        fenzi_mask_2 = load_image(fenzi_mask_path_2, 'numerator')
        fenzi_mask_3 = load_image(fenzi_mask_path_3, 'numerator')
        fenzi_mask_4 = load_image(fenzi_mask_path_4, 'numerator')
        fenmu_mask_1 = load_image(fenmu_mask_path_1, 'denominator')
        fenmu_mask_2 = load_image(fenmu_mask_path_2, 'denominator')
        fenmu_mask_3 = load_image(fenmu_mask_path_3, 'denominator')
        fenmu_mask_4 = load_image(fenmu_mask_path_4, 'denominator')
        
        img_1 = self.preprocess(1, img_1, self.scale, is_mask=False)
        img_2 = self.preprocess(1, img_2, self.scale, is_mask=False)
        img_3 = self.preprocess(1, img_3, self.scale, is_mask=False)
        img_4 = self.preprocess(1, img_4, self.scale, is_mask=False)

        fenzi_mask_1 = self.preprocess(0, fenzi_mask_1, self.scale, is_mask=False)
        fenzi_mask_2 = self.preprocess(0, fenzi_mask_2, self.scale, is_mask=False)
        fenzi_mask_3 = self.preprocess(0, fenzi_mask_3, self.scale, is_mask=False)
        fenzi_mask_4 = self.preprocess(0, fenzi_mask_4, self.scale, is_mask=False)

        fenmu_mask_1 = self.preprocess(0, fenmu_mask_1, self.scale, is_mask=False)
        fenmu_mask_2 = self.preprocess(0, fenmu_mask_2, self.scale, is_mask=False)
        fenmu_mask_3 = self.preprocess(0, fenmu_mask_3, self.scale, is_mask=False)
        fenmu_mask_4 = self.preprocess(0, fenmu_mask_4, self.scale, is_mask=False)


        assert img_1.shape == img_2.shape == img_3.shape == img_4.shape, "All images must have the same shape."
        assert fenzi_mask_1.shape == fenzi_mask_2.shape == fenzi_mask_3.shape == fenzi_mask_4.shape, "All fenzi masks must have the same shape."
        assert fenmu_mask_1.shape == fenmu_mask_2.shape == fenmu_mask_3.shape == fenmu_mask_4.shape, "All fenmu masks must have the same shape."
        assert img_1.shape == fenzi_mask_1.shape == fenmu_mask_1.shape, "All images and masks must have the same shape."

        img = np.concatenate([img_1, img_2, img_3, img_4], axis=0)
        fenzi_mask = np.concatenate([fenzi_mask_1, fenzi_mask_2, fenzi_mask_3, fenzi_mask_4], axis=0)
        fenmu_mask = np.concatenate([fenmu_mask_1, fenmu_mask_2, fenmu_mask_3, fenmu_mask_4], axis=0)
        # mask = np.concatenate([fenzi_mask_1, fenmu_mask_1, fenzi_mask_2, fenmu_mask_2, fenzi_mask_3, fenmu_mask_3, fenzi_mask_4, fenmu_mask_4], axis=0)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'fenzi_mask': torch.as_tensor(fenzi_mask.copy()).float().contiguous(),
            'fenmu_mask': torch.as_tensor(fenmu_mask.copy()).float().contiguous(),
            # 'mask': torch.as_tensor(mask.copy()).float().contiguous(),
            'name': name, 
            'idx': idx
        }


