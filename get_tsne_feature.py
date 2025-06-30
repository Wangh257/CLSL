"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import csv

import wandb

# import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
# from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter

from models.alexnet import MyAlexNetCMC
from models.resnet import MyResNetsCMC, MyResNetsLight_Singel_Encoder
from models.unet import MyUNet_Light_Singel_Encoder
from NCE.NCEAverage import NCEAverage, Feature_Dict, Feature_Dict_Singel_Encoder
from NCE.NCECriterion import NCECriterion
from NCE.NCECriterion import NCESoftmaxLoss

from dataset import BasicDataset

try:
    from apex import amp, optimizers
except ImportError:
    pass

import warnings 
warnings.filterwarnings('ignore')

# 离线运行
os.environ['WANDB_MODE'] = 'dryrun'

"""
TODO: python 3.6 ModuleNotFoundError
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=30, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50v2', choices=['alexnet',
                                                                         'resnet50v1', 'resnet101v1', 'resnet18v1',
                                                                         'resnet50v2', 'resnet101v2', 'resnet18v2',
                                                                         'resnet50v3', 'resnet101v3', 'resnet18v3', 'unet'])
    parser.add_argument('--softmax', action='store_true', help='using softmax contrastive loss rather than NCE')
    parser.add_argument('--nce_k', type=int, default=16384)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # dataset
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet'])

    # specify folder
    parser.add_argument('--data_folder', type=str, default='/home/lzy/data/ImageNet/imagenet', help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    # add new views
    parser.add_argument('--view', type=str, default='Lab', choices=['Lab', 'YCbCr'])

    # mixed precision setting
    parser.add_argument('--amp', action='store_true', help='using mixed precision')
    parser.add_argument('--opt_level', type=str, default='O2', choices=['O1', 'O2'])

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")
    parser.add_argument('--dir_img', type=str, required=True)
    parser.add_argument('--dir_mask', type=str, required=True)
    parser.add_argument('--global_step', type=int, default=0)

    opt = parser.parse_args()

    # if (opt.data_folder is None) or (opt.model_path is None) or (opt.tb_path is None):
    #     raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if opt.dataset == 'imagenet':
        if 'alexnet' not in opt.model:
            opt.crop_low = 0.08

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax' if opt.softmax else 'nce'
    opt.model_name = 'lr_{}_bsz_{}'.format(opt.learning_rate, opt.batch_size)

    if opt.amp:
        opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)

    # opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)

    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    # if not os.path.isdir(opt.data_folder):
    #     raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(args):
    """get the train loader"""
    # data_folder = os.path.join(args.data_folder, 'train')

    # if args.view == 'Lab':
    #     mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
    #     std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
    #     color_transfer = RGB2Lab()
    # elif args.view == 'YCbCr':
    #     mean = [116.151, 121.080, 132.342]
    #     std = [109.500, 111.855, 111.964]
    #     color_transfer = RGB2YCbCr()
    # else:
    #     raise NotImplemented('view not implemented {}'.format(args.view))
    # normalize = transforms.Normalize(mean=mean, std=std)

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     color_transfer,
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    dir_metal_1000 = '/home/wangh20/data/structure/metal_dataset/metal_1000_cera_800/metal_1000/images_GT'
    dir_mask_metal = '/home/wangh20/data/structure/metal_dataset/metal_1000_cera_800/metal_1000/'
    dir_cera_800 = '/home/wangh20/data/structure/metal_dataset/metal_1000_cera_800/cera_800/images_GT'
    dir_mask_cera = '/home/wangh20/data/structure/metal_dataset/metal_1000_cera_800/cera_800/'
    # try:
    #     train_dataset = BasicDataset(args.dir_img, args.dir_mask)
        # dataset_plat = BasicDataset(dir_img_plat, dir_mask_plat)
        # dataset_qiu1 = BasicDataset(dir_img_qiu1, dir_mask_qiu1)
        # dataset_qiu2 = BasicDataset(dir_img_qiu2, dir_mask_qiu2)
        # dataset = torch.utils.data.ConcatDataset([dataset, dataset_plat, dataset_qiu1, dataset_qiu2])
    # 使用img_use_path 决定训练的样本号，不使用则是默认用全部数据
    # cara_train_path = '/home/wangh20/projects/trans/tools/split_dataset/cara/100/train.npy'
    cara_train_path = '/home/wangh20/projects/trans/tools/split_dataset/cara/100/test.npy'

    metal_train_path = '/home/wangh20/projects/trans/tools/split_dataset/metal/all/test.npy'

    metal_dataset_1700 = BasicDataset(args.dir_img, args.dir_mask)

    metal_dataset_1000 = BasicDataset(dir_metal_1000, dir_mask_metal, img_use_path=metal_train_path)
    # metal_dataset_1000 = BasicDataset(dir_metal_1000, dir_mask_metal)

    cera_dataset_800 = BasicDataset(dir_cera_800, dir_mask_cera, img_use_path=cara_train_path)
    # cera_dataset_800 = BasicDataset(dir_cera_800, dir_mask_cera)

    # train_dataset = BasicDataset(args.dir_img, args.dir_mask)
    
    # train_dataset = torch.utils.data.ConcatDataset([metal_dataset_1700, metal_dataset_1000, cera_dataset_800])
    # train_dataset = torch.utils.data.ConcatDataset([metal_dataset_1700, metal_dataset_1000])
    train_dataset = metal_dataset_1000
    # train_dataset = cera_dataset_800
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = MyAlexNetCMC(args.feat_dim)
    elif args.model.startswith('resnet'):
        # model = MyResNetsCMC(args.model)
        model = MyResNetsLight_Singel_Encoder(args.model, mlp=True)
        # import pdb; pdb.set_trace()
        model = nn.DataParallel(model)
    elif args.model.startswith('unet'):
        model = MyUNet_Light_Singel_Encoder(args.model, mlp=True)
        model = nn.DataParallel(model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))

    contrast = Feature_Dict_Singel_Encoder(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
    # criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        # criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True
    return model, contrast, criterion_l


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def get_features(epoch, train_loader, model, contrast, criterion_l, optimizer, opt):
    """
    one epoch training
    """
    model.eval()
    contrast.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # losses = AverageMeter()
    # f_fenzi_loss_meter = AverageMeter()
    # f_fenmu_loss_meter = AverageMeter()
    # fenzi_f_loss_meter = AverageMeter()
    # fenzi_fenmu_loss_meter = AverageMeter()
    # fenmu_f_loss_meter = AverageMeter()
    # fenmu_fenzi_loss_meter = AverageMeter()

    features_f = []
    features_fenzi = []
    features_fenmu = []
    labels = []

    end = time.time()
    for idx, images in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)
        images_fringe = images['image']
        images_fenzi = images['fenzi_mask']
        images_fenmu = images['fenmu_mask']
        # images_mask = images['mask']
        index = images['idx']
        assert images_fringe.shape[1] == 4, f'Network has been designed to work with 1 channel images, but received {images_fringe.shape[1]} channels'

        bsz = images_fringe.size(0)
        images_fringe = images_fringe.float()
        images_fenzi = images_fenzi.float()
        images_fenmu = images_fenmu.float()

        if torch.cuda.is_available():
            index = index.cuda()
            images_fringe = images_fringe.cuda()
            images_fenzi = images_fenzi.cuda() 
            images_fenmu = images_fenmu.cuda()

        # ===================forward=====================
        fea_f, fea_fenzi, fea_fenmu = model(images_fringe, images_fenzi, images_fenmu)

        # save the feature 


        fea_f = fea_f.detach().cpu().numpy()
        fea_fenzi = fea_fenzi.detach().cpu().numpy()
        fea_fenmu = fea_fenmu.detach().cpu().numpy()
        label = index.detach().cpu().numpy()

        features_f.append(fea_f)
        features_fenzi.append(fea_fenzi)
        features_fenmu.append(fea_fenmu)
        labels.append(label)

    # import pdb;pdb.set_trace()
    features_f_all = np.concatenate(features_f, axis=0)        # N × D1
    features_fenzi_all = np.concatenate(features_fenzi, axis=0)  # N × D2
    features_fenmu_all = np.concatenate(features_fenmu, axis=0)  # N × D3
    labels_all = np.concatenate(labels, axis=0)                # N
    # 保存为 .npy 文件
    np.save(os.path.join(opt.model_folder, 'features_f.npy'), features_f_all)
    np.save(os.path.join(opt.model_folder, 'features_fenzi.npy'), features_fenzi_all)
    np.save(os.path.join(opt.model_folder, 'features_fenmu.npy'), features_fenmu_all)
    np.save(os.path.join(opt.model_folder, 'labels.npy'), labels_all)


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_l = set_model(args, n_data)

    # load checkpoints 
    # checkpoints_path = '/home/wangh20/projects/trans/NCE_Light_encoder/tools/checkpoints_curve/cera_800/lr_0.001_bsz_8/best_checkpoint.pth'
    checkpoints_path = '/home/wangh20/projects/trans/NCE_Light_encoder/tools/checkpoints_curve/metal_1000/lr_0.001_bsz_8/best_checkpoint.pth'

    checkpoint = torch.load(checkpoints_path)
    model.load_state_dict(checkpoint['model'])

    # set the optimizer
    optimizer = set_optimizer(args, model)

    args.start_epoch = 1

    # routine
    for epoch in range(args.start_epoch, args.start_epoch + 1):
        # adjust_learning_rate(epoch, args, optimizer)
        print("==> get_features...")
        time1 = time.time()
        get_features(epoch, train_loader, model, contrast, criterion_l, optimizer, args)
        time2 = time.time()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
