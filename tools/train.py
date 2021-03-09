# Note -- this training script is tweaked from the original version at:
#
#           https://github.com/pytorch/vision/tree/v0.3.0/references/segmentation
#
# It's also meant to be used against this fork of torchvision, which includes 
# some patches for exporting to ONNX and adds fcn_resnet18 and fcn_resnet34:
#
#           https://github.com/dusty-nv/vision/tree/v0.3.0
#
import argparse
import datetime
import time
import os
import shutil
import numpy as np
from PIL import Image

import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision


from datasets.mapillary import MapillarySegmentation
import transforms as T
from tools import utils

model_names = sorted(name for name in torchvision.models.segmentation.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(torchvision.models.segmentation.__dict__[name]))


#
# parse command-line arguments
#
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', default='mapillary', help='dataset type: voc, voc_aug, coco, '
                                                               'cityscapes, deepscene, mhp, nyu, sun, '
                                                               'apolloscapes (default: voc)')
    parser.add_argument('--save_dir', default='mapillary_dir', help='path where to save output models and logs')
    parser.add_argument('--color', action='store_true', help='color mode of visualization')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--ocr', action='store_true', help='model with ocr')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fcn_resnet18',
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: fcn_resnet18)')
    parser.add_argument('--aux_loss', action='store_true', help='train with auxilliary loss')
    parser.add_argument('--focal_loss', action='store_true', help='train with focal loss')
    parser.add_argument('--gamma_focal_loss', default=2, type=int, help='train with focal loss with gamma')
    parser.add_argument('--resolution', default=320, type=int, metavar='N',
                        help='NxN resolution used for scaling the training dataset (default: 320x320) '
                             'to specify a non-square resolution, use the --width and --height options')
    parser.add_argument('--width', default=1920, type=int, metavar='X',
                        help='desired width of the training dataset. '
                             'if this option is not set, --resolution will be used')
    parser.add_argument('--height', default=1080, type=int, metavar='Y',
                        help='desired height of the training dataset. '
                             'if this option is not set, --resolution will be used')
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test_only", help="Only test the model", action="store_true")
    parser.add_argument("--vis_only", help="Only vis the predictions of model", action="store_true")
    parser.add_argument("--pretrained_backbone", help="Use pre-trained models (only supported "
                                                                "for fcn_resnet101)", action="store_true")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


#
# load desired dataset
#
def get_dataset(name, path, image_set, transform):
    paths = {"mapillary": (path, MapillarySegmentation, 66)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


#
# create data transform
#
def get_transform(train, resolution):
    transforms = []

    # if square resolution, perform some aspect cropping
    # otherwise, resize to the resolution as specified
    if resolution[0] == resolution[1]:
        base_size = resolution[0] + 32
        crop_size = resolution[0]

        min_size = int((0.5 if train else 1.0) * base_size)
        max_size = int((2.0 if train else 1.0) * base_size)

        transforms.append(T.RandomResize(min_size, max_size))

        # during training mode, perform some data randomization
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.RandomCrop(crop_size))
    else:
        # transforms.append(T.Resize(resolution))

        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def evaluate(model, data_loader, device, num_classes):
    from config import MAPILLARY_CLASSNAMES as classnames
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes, classnames)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 200, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            confmat.update(target.cpu().flatten(), output.argmax(1).cpu().flatten())

        confmat.reduce_from_all_processes()

    return confmat


def vis(model, data_loader, device, save_dir, color):
    from config import MAPILLARY_PALETTE
    palette = MAPILLARY_PALETTE

    inference_times = []
    model.eval()
    with torch.no_grad():
        for i, (image, _) in enumerate(data_loader):
            image = image.to(device)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            output = model(image)
            output = output.argmax(1)

            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start_time
            if i != 0: inference_times.append(inference_time)
            if i % 100 == 0:
                print(inference_time)
                print(f'{i} / {len(data_loader)}')

            pred = np.asarray(output.cpu(), dtype=np.uint8)
            save_img = Image.fromarray(pred[0])
            if color: save_img.putpalette(palette)
            save_path = f'{save_dir}/{i:05d}.png'
            save_img.save(save_path)

    print(f'Images are saved at {os.path.abspath(save_dir)}\n')

    count = len(inference_times)
    mean = round(np.mean(inference_times), 5)
    std = round(np.std(inference_times), 5)

    print(f'count = {count}')
    print('mean inference time, std inference time')
    print(f'{mean}, {std}')


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        writer.add_scalar('training loss', loss.item(), epoch * len(data_loader) + i)
        writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch * len(data_loader) + i)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if "width" in args and "height" in args:
        resolution = (args.height, args.width)

    # load the train and val datasets
    dataset, num_classes = get_dataset(name=args.dataset,
                                       path="",
                                       image_set="train",
                                       transform=get_transform(train=True, resolution=resolution))

    dataset_test, _ = get_dataset(name=args.dataset,
                                  path="",
                                  image_set="val",
                                  transform=get_transform(train=False, resolution=resolution))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              sampler=train_sampler,
                                              num_workers=args.workers,
                                              collate_fn=utils.collate_fn,
                                              drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   sampler=test_sampler,
                                                   num_workers=args.workers,
                                                   collate_fn=utils.collate_fn)

    print("=> training with dataset: '{:s}' (train={:d}, val={:d})".format(args.dataset,
                                                                           len(dataset),
                                                                           len(dataset_test)))
    print("=> training with resolution: {:d}x{:d}, {:d} classes".format(resolution[1],
                                                                        resolution[0],
                                                                        num_classes))

    # create the segmentation model
    if args.ocr:
        if args.arch == 'resnet34_base_oc_layer3':
            from network.model import get_resnet34_base_oc_layer3 as get_model
        elif args.arch == 'resnet34_pyramid_oc_layer3':
            from network.model import get_resnet34_pyramid_oc_layer3 as get_model
        elif args.arch == 'resnet34_asp_oc_layer3':
            from network.model import get_resnet34_asp_oc_layer3 as get_model

        model = get_model(num_classes=num_classes, pretrained_backbone=args.pretrained_backbone)

        model_name = get_model.__name__[4:]
        print(f"=> training with model: {model_name}")

    elif args.arch == 'resnet34_layer3':
        from network.model import get_resnet34_layer3 as get_model
        model = get_model(num_classes=num_classes, pretrained_backbone=args.pretrained_backbone)

        model_name = get_model.__name__[4:]
        print(f"=> training with model: {model_name}")

    elif args.arch == 'resnet34':
        model = torchvision.models.segmentation.__dict__[args.arch](num_classes=num_classes,
                                                                    aux_loss=args.aux_loss,
                                                                    pretrained=args.pretrained_backbone)
        print("=> training with model: {:s}".format(args.arch))

    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # eval-only mode
    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    if args.vis_only:
        start_time = time.time()
        save_img_dir = os.path.join(args.save_dir, 'pred_images')
        os.makedirs(save_img_dir, exist_ok=True)

        dataset_test_vis, _ = get_dataset(name=args.dataset,
                                          path="",
                                          image_set="test",
                                          transform=get_transform(train=False, resolution=resolution))

        data_loader_test_vis = torch.utils.data.DataLoader(dataset_test_vis,
                                                           batch_size=1)

        vis(model, data_loader_test_vis, device=device, save_dir=save_img_dir, color=args.color)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Visualization time {}'.format(total_time_str))
        return

    # create the optimizer
    if args.ocr:
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.context.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.cls.parameters() if p.requires_grad]},
        ]
    elif args.arch == 'resnet34_layer3':
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.cls.parameters() if p.requires_grad]},
        ]
    elif args.arch == 'resnet34':
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]

        if args.aux_loss:
            params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    if args.focal_loss:
        from tools.utils import FocalLoss
        criterion = FocalLoss(gamma=args.gamma_focal_loss)
    else:
        from config import MAPILLARY_LOSS_WEIGHTS
        loss_weights = torch.FloatTensor(MAPILLARY_LOSS_WEIGHTS)
        loss_weights = loss_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)

    save_model_dir = os.path.join(args.save_dir, 'weights')
    os.makedirs(save_model_dir, exist_ok=True)

    writer = SummaryWriter(args.save_dir)

    best_IoU = 0.0
    epoch = 0
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_IoU = checkpoint['mean_IoU']
        epoch = checkpoint['epoch']
        print(f"best_IoU == {best_IoU}")
        print(f"epoch == {epoch}")

    start_time = time.time()
    for current_epoch in range(epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(current_epoch)

        # train the model over the next epoc
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, current_epoch, args.print_freq, writer)

        # test the model on the val dataset
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        iou = confmat.get_iou()
        for classname, iu in iou.items():
            writer.add_scalar(f'{classname} IoU', iu, current_epoch)

        # save model checkpoint
        checkpoint_path = os.path.join(save_model_dir, 'model_{}.pth'.format(current_epoch))

        utils.save_on_master(
            dict(model=model_without_ddp.state_dict(), optimizer=optimizer.state_dict(), epoch=current_epoch, args=args,
                 arch=args.arch, dataset=args.dataset, num_classes=num_classes, resolution=resolution,
                 accuracy=confmat.acc_global, mean_IoU=confmat.mean_IoU, lr_scheduler=lr_scheduler.state_dict()),
            checkpoint_path)

        print('saved checkpoint to:  {:s}  ({:.3f}% '
              'mean IoU, {:.3f}% accuracy)'.format(checkpoint_path, confmat.mean_IoU, confmat.acc_global))
        
        if confmat.mean_IoU > best_IoU:
            best_IoU = confmat.mean_IoU
            best_path = os.path.join(save_model_dir, 'model_best.pth')
            shutil.copyfile(checkpoint_path, best_path)
            print('saved best model to:  {:s}  ({:.3f}% mean IoU, {:.3f}% accuracy)'.format(best_path, best_IoU,
                                                                                            confmat.acc_global))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = parse_args()
    main(args)
