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
    cara_train_path = '/home/wangh20/projects/trans/tools/split_dataset/cara/all/train.npy'

    metal_train_path = '/home/wangh20/projects/trans/tools/split_dataset/metal/all/train.npy'

    metal_dataset_1700 = BasicDataset(args.dir_img, args.dir_mask)

    # metal_dataset_1000 = BasicDataset(dir_metal_1000, dir_mask_metal, img_use_path=metal_train_path)
    metal_dataset_1000 = BasicDataset(dir_metal_1000, dir_mask_metal)

    # cera_dataset_800 = BasicDataset(dir_cera_800, dir_mask_cera, img_use_path=cara_train_path)
    cera_dataset_800 = BasicDataset(dir_cera_800, dir_mask_cera)

    # train_dataset = BasicDataset(args.dir_img, args.dir_mask)
    
    # train_dataset = torch.utils.data.ConcatDataset([metal_dataset_1700, metal_dataset_1000, cera_dataset_800])
    # train_dataset = torch.utils.data.ConcatDataset([metal_dataset_1700, metal_dataset_1000])
    # train_dataset = metal_dataset_1000
    train_dataset = cera_dataset_800
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
        # import pdb; pdb.set_trace()
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


def train(epoch, train_loader, model, contrast, criterion_l, experiment, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f_fenzi_loss_meter = AverageMeter()
    f_fenmu_loss_meter = AverageMeter()
    fenzi_f_loss_meter = AverageMeter()
    fenzi_fenmu_loss_meter = AverageMeter()
    fenmu_f_loss_meter = AverageMeter()
    fenmu_fenzi_loss_meter = AverageMeter()
    # ab_loss_meter = AverageMeter()
    # l_prob_meter = AverageMeter()
    # ab_prob_meter = AverageMeter()

    end = time.time()
    for idx, images in enumerate(train_loader):
        data_time.update(time.time() - end)
        images_fringe = images['image']
        images_fenzi = images['fenzi_mask']
        images_fenmu = images['fenmu_mask']
        # images_mask = images['mask']
        index = images['idx']
        assert images_fringe.shape[1] == 4, f'Network has been designed to work with 1 channel images, but received {images_fringe.shape[1]} channels'

        bsz = images_fringe.size(0)
        images_fringe = images_fringe.float()
        # images_mask = images_mask.float()
        images_fenzi = images_fenzi.float()
        images_fenmu = images_fenmu.float()

        if torch.cuda.is_available():
            index = index.cuda()
            images_fringe = images_fringe.cuda()
            images_fenzi = images_fenzi.cuda() 
            images_fenmu = images_fenmu.cuda()

        # ===================forward=====================
        fea_f, fea_fenzi, fea_fenmu = model(images_fringe, images_fenzi, images_fenmu)
        f_fenzi, f_fenmu, fenzi_f, fenzi_fenmu, fenmu_f, fenmu_fenzi = contrast(fea_f, fea_fenzi, fea_fenmu, index)
        # import pdb; pdb.set_trace()
        f_fenzi_loss = criterion_l(f_fenzi)
        f_fenmu_loss = criterion_l(f_fenmu)
        fenzi_f_loss = criterion_l(fenzi_f)
        fenzi_fenmu_loss = criterion_l(fenzi_fenmu)
        fenmu_f_loss = criterion_l(fenmu_f)
        fenmu_fenzi_loss = criterion_l(fenmu_fenzi)

        # l_prob = out_f[:, 0].mean()
        # ab_prob = out_p[:, 0].mean()
        loss = f_fenzi_loss + f_fenmu_loss + fenzi_f_loss + fenzi_fenmu_loss + fenmu_f_loss + fenmu_fenzi_loss
        # ===================backward=====================
        optimizer.zero_grad()
        if opt.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        f_fenzi_loss_meter.update(f_fenzi_loss.item(), bsz)
        f_fenmu_loss_meter.update(f_fenmu_loss.item(), bsz)
        fenzi_f_loss_meter.update(fenzi_f_loss.item(), bsz)
        fenzi_fenmu_loss_meter.update(fenzi_fenmu_loss.item(), bsz)
        fenmu_f_loss_meter.update(fenmu_f_loss.item(), bsz)
        fenmu_fenzi_loss_meter.update(fenmu_fenzi_loss.item(), bsz)
        # l_loss_meter.update(l_loss.item(), bsz)
        # l_prob_meter.update(l_prob.item(), bsz)
        # ab_loss_meter.update(ab_loss.item(), bsz)
        # ab_prob_meter.update(ab_prob.item(), bsz)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'f_fenzi {f_fenzi_loss_meter.val:.3f} ({f_fenzi_loss_meter.avg:.3f})\t'
                  'f_fenmu {f_fenmu_loss_meter.val:.3f} ({f_fenmu_loss_meter.avg:.3f})\t'
                  'fenzi_f {fenzi_f_loss_meter.val:.3f} ({fenzi_f_loss_meter.avg:.3f})\t'
                  'fenzi_fenmu {fenzi_fenmu_loss_meter.val:.3f} ({fenzi_fenmu_loss_meter.avg:.3f})\t'
                  'fenmu_f {fenmu_f_loss_meter.val:.3f} ({fenmu_f_loss_meter.avg:.3f})\t'
                  'fenmu_fenzi {fenmu_fenzi_loss_meter.val:.3f} ({fenmu_fenzi_loss_meter.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   loss=losses, f_fenzi_loss_meter=f_fenzi_loss_meter, f_fenmu_loss_meter=f_fenmu_loss_meter,
                   fenzi_f_loss_meter=fenzi_f_loss_meter, fenzi_fenmu_loss_meter=fenzi_fenmu_loss_meter,
                   fenmu_f_loss_meter=fenmu_f_loss_meter, fenmu_fenzi_loss_meter=fenmu_fenzi_loss_meter))
            # import pdb;pdb.set_trace()
            # print(out_f.shape)
            sys.stdout.flush()
        opt.global_step += 1
        # import pdb;pdb.set_trace()
        experiment.log({
        'ALL loss': loss.item(),
        'f_fenzi_loss': f_fenzi_loss.item(),
        'f_fenmu_loss': f_fenmu_loss.item(),
        'fenzi_f_loss': fenzi_f_loss.item(),
        'fenzi_fenmu_loss': fenzi_fenmu_loss.item(),
        'fenmu_f_loss': fenmu_f_loss.item(),
        'fenmu_fenzi_loss': fenmu_fenzi_loss.item(),
        'step': opt.global_step,
        'epoch': epoch,
        'learning_rate': optimizer.param_groups[0]['lr']
        })

    # return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg
    return f_fenzi_loss_meter.avg, f_fenmu_loss_meter.avg, fenzi_f_loss_meter.avg, fenzi_fenmu_loss_meter.avg, fenmu_f_loss_meter.avg, fenmu_fenzi_loss_meter.avg


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_l = set_model(args, n_data)


    # set the optimizer
    optimizer = set_optimizer(args, model)

    # set mixed precision
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    experiment = wandb.init(project='CMC_Light_Singel_Encoder', resume='allow', anonymous='must')
    experiment.config.update(
    dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
        )
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            if args.amp and checkpoint['opt'].amp:
                print('==> resuming amp state_dict')
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    # logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    best_loss = 10000
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        f_fenzi_loss, f_fenmu_loss, fenzi_f_loss, fenzi_fenmu_loss, fenmu_f_loss, fenmu_fenzi_loss = train(epoch, train_loader, model, contrast, criterion_l, experiment, optimizer, args)
        time2 = time.time()
        # ========== 添加 CSV 写入逻辑 ==========
        log_file = os.path.join(args.model_folder, 'training_loss_log.csv')
        write_header = (epoch == 0 and not os.path.exists(log_file))
        with open(log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    'epoch', 'global_step',
                    'f_fenzi_loss', 'f_fenmu_loss',
                    'fenzi_f_loss', 'fenzi_fenmu_loss',
                    'fenmu_f_loss', 'fenmu_fenzi_loss',
                    'learning_rate'
                ])
            writer.writerow([
                epoch,
                args.global_step,
                f_fenzi_loss, f_fenmu_loss,
                fenzi_f_loss, fenzi_fenmu_loss,
                fenmu_f_loss, fenmu_fenzi_loss,
                optimizer.param_groups[0]['lr']
            ])
    # =======================================
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        # try:
        #     experiment.log({
        #         'learning rate': optimizer.param_groups[0]['lr'],
        #         'f_fenzi_loss': f_fenzi_loss,
        #         'f_fenmu_loss': f_fenmu_loss,
        #         'fenzi_f_loss': fenzi_f_loss,
        #         'fenzi_fenmu_loss': fenzi_fenmu_loss,
        #         'fenmu_f_loss': fenmu_f_loss,
        #         'fenmu_fenzi_loss': fenmu_fenzi_loss,
        #         'epoch': epoch,
        #     })
        # except:
        #     print('wandb failed')
        # tensorboard logger
        # logger.log_value('l_loss', l_loss, epoch)
        # logger.log_value('l_prob', l_prob, epoch)
        # logger.log_value('ab_loss', ab_loss, epoch)
        # logger.log_value('ab_prob', ab_prob, epoch)

        # save model
        # if epoch % args.save_freq == 0:
        if f_fenzi_loss < best_loss:
            best_loss = f_fenzi_loss
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'best_checkpoint.pth')
            torch.save(state, save_file)
            # help release GPU memory
            del state

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
