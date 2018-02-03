# Custom tensorboard logging
from TbLogger import Logger

import argparse
import os
import shutil
import time
import tqdm
from skimage.io import imsave

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# custom classes
from UNet import UNet11
from DilatedResnet import GapNet18,GapNetImg18
from Loss import BCEDiceLoss,TDiceLoss,DiceLoss
from presets import preset_dict
from SatellitesDataset import get_train_dataset_mul_ps_preds, SatellitesDatasetRefine
from SatellitesAugs import SatellitesTrainAugmentation,SatellitesTestAugmentation
from presets import preset_dict

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='PyTorch Satellites semseg training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='linknet34',
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-dl', '--dilation', default=1, type=int,
                    metavar='N', help='convolution dilation size for resnet')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                    help='model optimizer')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lognumber', '-log', default='test_model', type=str,
                    metavar='LN', help='text id for saving logs')
parser.add_argument('--preset', '-pres', default='mul_urban', type=str,
                    metavar='PS', help='preset for satellite channels')
parser.add_argument('--tensorboard', default=False, type=str2bool,
                    help='Use tensorboard to for loss visualization')
parser.add_argument('--augs', default=False, type=str2bool,
                    help='Use augs for training')
parser.add_argument('-im', '--imsize', default=320, type=int, metavar='N',
                    help='image size')
parser.add_argument('-s', '--seed', default=42, type=int, metavar='N',
                    help='seed for train test split (default: 42)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-pr', '--predict', dest='predict', action='store_true',
                    help='generate prediction masks')
parser.add_argument('-pr_train', '--predict_train', dest='predict_train', action='store_true',
                    help='generate prediction masks')
parser.add_argument('--tensorboard_images', default=False, type=str2bool,
                    help='Use tensorboard to see images')

best_val_loss = 100
train_minib_counter = 0
valid_minib_counter = 0

args = parser.parse_args()

print(args)

def to_np(x):
    x = x.cpu().numpy()
    if len(x.shape)>3:
        return x[:,0:3,:,:]
    else:
        return x

# remove the log file if it exists if we run the script in the training mode
if not (args.evaluate or args.predict or args.predict_train):
    print('Folder {} delete triggered'.format(args.lognumber))
    try:
        shutil.rmtree('tb_logs/{}/'.format(args.lognumber))
    except:
        pass

# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    logger = Logger('./tb_logs/{}'.format(args.lognumber))

def main():
    global args, best_prec1,best_val_loss
    global logger
    
    # get 8-bit images
    # gt masks
    # narrow and wide mask predictions
    bit8_imgs,bit8_masks,bit8_masks_narrow_pred,\
    bit8_masks_wide_pred,cty_no = get_train_dataset_mul_ps_preds('mul_ps_vegetation',
                                                                 preset_dict,
                                                                 narrow_mask_folder = 'ln34_mul_ps_vegetation_aug_dice_predict_train_pad',
                                                                 wide_mask_folder = 'ln34_wide_masks_mul_ps_vegetation_aug_dice_predict_train_pad'
                                                                )
    
    # not refactored for new pipeline
    # if args.predict:
    #     predict_imgs,predict_city_folders,predict_img_names,cty_no_test,predict_prefix = get_test_dataset(args.preset,
    #                                                                                       preset_dict)    
    # elif args.predict_train:
    #     predict_imgs,predict_city_folders,predict_img_names,cty_no_test,predict_prefix = get_train_dataset_for_predict(args.preset,
    #                                                                                       preset_dict)     

    train_imgs, val_imgs, train_masks, val_masks,\
    train_narrow_masks, val_narrow_masks,\
    train_wide_masks, val_wide_masks= train_test_split(bit8_imgs,
                                                        bit8_masks,
                                                        bit8_masks_narrow_pred,
                                                        bit8_masks_wide_pred,
                                                        test_size=0.2,
                                                        stratify=cty_no,
                                                        random_state=args.seed)

    if not (args.predict or args.predict_train):
        print('Train  images        : {}\n'
              'Train  masks         : {}\n'
              'Train  narrow masks  : {}\n'
              'Train  wide masks    : {}\n'
              'Val    images        : {}\n'
              'Val    masks         : {}\n'
              'Val    narrow masks  : {}\n'
              'Val    wide masks    : {}\n'.format(len(train_imgs),len(train_masks),len(train_narrow_masks),len(train_wide_masks),
                                          len(val_imgs),len(val_masks),len(val_narrow_masks),len(val_wide_masks)))
              
    # else:
    #    print('Predict images: {}\n'.format(len(predict_imgs)))        
    
    
    if args.arch.startswith('gapnet18'):
        print('Gapnet18 activated')
        model = GapNetImg18(num_classes=1,
                        num_channels=1,
                        dilation=args.dilation)
    else:
        raise ValueError('Model not supported')
    
    # train on 2 GPUs for speed
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
     
    if not (args.predict or args.predict_train):
        
        train_augs = SatellitesTrainAugmentation(shape=args.imsize,
                                                 aug_scheme = args.augs)

        val_augs = SatellitesTestAugmentation(shape=args.imsize)
        
        train_dataset = SatellitesDatasetRefine(preset = preset_dict[args.preset],
                                          image_paths = train_imgs,
                                          mask_paths = train_masks,
                                          narrow_mask_paths = train_narrow_masks,
                                          wide_mask_paths = train_wide_masks,                                       
                                          transforms = train_augs,
                                         )

        val_dataset = SatellitesDatasetRefine(preset = preset_dict[args.preset],
                                        image_paths = val_imgs,
                                        mask_paths = val_masks,
                                        narrow_mask_paths = val_narrow_masks,
                                        wide_mask_paths = val_wide_masks,                                                
                                        transforms = val_augs,
                                       )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)
        
    # else:
    #    predict_augs = SatellitesTestAugmentation(shape=args.imsize)    


    
    #    predict_dataset = SatellitesDataset(preset = preset_dict[args.preset],
    #                                    image_paths = predict_imgs,
    #                                    mask_paths = None,
    #                                    transforms = predict_augs,
    #                                   )          
        # predict loader loads the images sequentially
    #    predict_loader = torch.utils.data.DataLoader(
    #        predict_dataset,
    #        batch_size=args.batch_size,        
    #        shuffle=False,
    #        num_workers=args.workers,
    #        pin_memory=True)    

    # play with criteria?
    criterion = TDiceLoss().cuda()
    # criterion = DiceLoss().cuda()
    
    if args.optimizer.startswith('adam'):           
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                    lr = args.lr)
    elif args.optimizer.startswith('rmsprop'):
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                    lr = args.lr)
    else:
        raise ValueError('Optimizer not supported')        
        
    scheduler = ReduceLROnPlateau(optimizer = optimizer,
                                              mode = 'min',
                                              factor = 0.1,
                                              patience = 4,
                                              verbose = True,
                                              threshold = 1e-3,
                                              min_lr = 1e-5
                                             )    
    # if we pass evaluate or predict flat, training loop is omitted altogether
    # if args.evaluate:
    #    validate(val_loader, model, criterion)
    #    return
    
    # if args.predict or args.predict_train:
    #    predict(predict_loader,
    #            model,
    #            predict_imgs,
    #            predict_city_folders,
    #            predict_img_names,
    #            predict_prefix)
    #    return    

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss = validate(val_loader, model, criterion)
        
        scheduler.step(val_loss)

        # add code for early stopping here 
        # 
        #
  

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_epoch_loss': train_loss,
                'valid_epoch_loss': val_loss
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)                     
        
        # remember best prec@1 and save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
        },
        is_best,
        'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
        'weights/{}_best.pth.tar'.format(str(args.lognumber))
        )

def train(train_loader, model, criterion, optimizer, epoch):
    global train_minib_counter
    global logger
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_narrow, target_wide) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        target_narrow = target_narrow.float().cuda(async=True)
        target_wide = target_wide.float().cuda(async=True)
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_narrow_var = torch.autograd.Variable(target_narrow)
        target_wide_var = torch.autograd.Variable(target_wide)
        
        # compute output
        output = model(input_var,target_narrow_var,target_wide_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_loss': losses.val,
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter)                
        
        train_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    print(' * Avg Train Loss {loss.avg:.4f}'.format(loss=losses))         
            
    return losses.avg

def validate(val_loader, model, criterion):
    global valid_minib_counter
    global logger
    
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, target_narrow, target_wide) in enumerate(val_loader):
        
        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        target_narrow = target_narrow.float().cuda(async=True)
        target_wide = target_wide.float().cuda(async=True)
        
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        target_narrow_var = torch.autograd.Variable(target_narrow, volatile=True)
        target_wide_var = torch.autograd.Variable(target_wide, volatile=True)

        # compute output
        output = model(input_var,target_narrow_var,target_wide_var)
        
        #============ TensorBoard logging ============#              
        # Show original images
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'a_images': to_np(input.view(-1,len(preset_dict[args.preset]['channels']),args.imsize, args.imsize)[:10])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, train_minib_counter)
        # Show gt masks
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'b_gt': to_np(target.view(-1,args.imsize, args.imsize)[:10])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, train_minib_counter)                    
        # Show narrow masks
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'c_narrow_masks': to_np(target_narrow.view(-1,args.imsize, args.imsize)[:10])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, train_minib_counter)
        # Show wide masks
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'd_wide_masks': to_np(target_wide.view(-1,args.imsize, args.imsize)[:10])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, train_minib_counter)                     
        # Show the output masks
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'e_preds': to_np(output.data.view(-1,args.imsize, args.imsize)[:10])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, train_minib_counter)  
        
        
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'valid_loss': losses.val,
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, valid_minib_counter)            
        
        valid_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print(' * Avg Val Loss {loss.avg:.4f}'.format(loss=losses))

    return losses.avg

def predict(predict_loader,
            model,
            predict_imgs,
            predict_city_folders,
            predict_img_names,
            predict_prefix):
    
    global valid_minib_counter
    global logger
    
    print('Starting to do the predictions')
    c = 0
    # switch to evaluate mode
    model.eval()
    
    print(predict_img_names[0:16])
    print(len(predict_img_names))

    with tqdm.tqdm(total=len(predict_loader)) as pbar:
        for i, (input) in enumerate(predict_loader):

            input = input.float().cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)

            # compute output
            output = model(input_var)
            
            for pred_image in output:

                prediction_folder = os.path.join(predict_prefix,predict_city_folders[c],args.lognumber)
                
                # check that prediction folder exists
                if not os.path.exists(prediction_folder):
                    os.mkdir(prediction_folder)                    
                
                # print(c)
                # print(predict_img_names[c])
                im_path = os.path.join(prediction_folder,predict_img_names[c][:-3]+'jpg')
                
                # save image to disk
                imsave(im_path,pred_image.data.cpu().numpy()[0,:,:])
                
                c+=1

            # dry run
            # break
            
            pbar.update(1)            

    return 1

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 50 epochs"""
    lr = args.lr * (0.9 ** ( (epoch+1) // 50))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()