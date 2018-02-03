# Custom tensorboard logging
from TbLogger import Logger

import argparse
import os
import shutil
import time
import tqdm
from skimage.io import imsave
import numpy as np

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
from sklearn.model_selection import StratifiedKFold

# custom classes
from UNet import UNet11
from LinkNet import LinkNet34,LinkNet50,LinkNet50_full,LinkNeXt
from Loss import BCEDiceLoss,TDiceLoss,DiceLoss
from presets import preset_dict
from SatellitesDataset import get_test_dataset,get_train_dataset,SatellitesDataset,get_train_dataset_for_predict,get_train_dataset_wide_masks,get_train_dataset_layered_masks,get_train_dataset_all
from SatellitesAugs import SatellitesTrainAugmentation,SatellitesTestAugmentation,SatellitesTestAugmentationTTA
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
parser.add_argument('--city', '-cty', default='all', type=str,
                    metavar='CTY', help='a city to train on')
parser.add_argument('--hflip', default=False, type=str2bool,
                    help='horizontal TTA flip')
parser.add_argument('--vflip', default=False, type=str2bool,
                    help='vertical TTA flip')

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

# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    logger = Logger('./tb_logs/{}'.format(args.lognumber))

def main():
    global args, best_prec1,best_val_loss
    global logger

    # train either on normal masks or on wide masks or on 3-layer masks
    bit8_imgs,bit8_masks,cty_no = get_train_dataset_all(args.preset,
                                                    preset_dict,
                                                    args.city)
    
    # bit8_imgs,bit8_masks,cty_no = get_train_dataset(args.preset,
    #                                                 preset_dict,
    #                                                args.city)
    
    # bit8_imgs,bit8_masks,cty_no = get_train_dataset_wide_masks(args.preset,
    #                                                           preset_dict)    

    # bit8_imgs,bit8_masks,cty_no = get_train_dataset_layered_masks(args.preset,
    #                                                               preset_dict)      
    
    if args.predict:
        predict_imgs,predict_city_folders,predict_img_names,cty_no_test,predict_prefix = get_test_dataset(args.preset,
                                                                                           preset_dict,
                                                                                           args.city)    
    elif args.predict_train:
        predict_imgs,predict_city_folders,predict_img_names,cty_no_test,predict_prefix = get_train_dataset_for_predict(args.preset,
                                                                                           preset_dict,
                                                                                           args.city)     

    skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 42)
    f1, f2, f3 = skf.split(bit8_imgs, cty_no)
    folds = [f1, f2, f3]
    
    bit8_imgs = np.array(bit8_imgs)
    bit8_masks = np.array(bit8_masks)
    
    # train_imgs, val_imgs, train_masks, val_masks = train_test_split(bit8_imgs,
    #                                                                 bit8_masks,
    #                                                                 test_size=0.25,
    #                                                                 stratify=cty_no,
    #                                                                random_state=args.seed)    
    for i,fold in enumerate(folds):
        
        best_val_loss = 100
        
        # if i!=2:
        #    continue
            
        # remove the log file if it exists if we run the script in the training mode
        if not (args.evaluate or args.predict or args.predict_train):
            print('Folder {} delete triggered'.format(args.lognumber + '_fold{}'.format(i) ))
            try:
                pass
                # shutil.rmtree('tb_logs/{}/'.format(args.lognumber + '_fold{}'.format(i) ))
            except:
                pass        
        
        # update the tb logger
        if args.tensorboard or args.tensorboard_images:
            logger = Logger('./tb_logs/{}'.format(args.lognumber + '_fold{}'.format(i) ))        
        
        print('Processing fold : {}'.format(i))

        
        if args.arch.startswith('linknet34'):
            print('Full linknet34 activated')
            if args.preset in ['mul_ps_8channel','mul_8channel']:
                model = LinkNet34(num_channels=8,
                                  num_classes=1)
            else:
                model = LinkNet34(num_channels=3,
                                  num_classes=1)
        elif args.arch.startswith('linknext'):
            print('LinkNeXt101-32 activated')
            model = LinkNeXt(num_channels=3,
                              num_classes=1)            
        elif args.arch.startswith('linknet50_full'):
            print('Full linknet50 activated')
            if args.preset in ['mul_ps_8channel','mul_8channel']:
                model = LinkNet50_full(num_channels=8,
                                  num_classes=1)
            else:
                model = LinkNet50_full(num_channels=3,
                                  num_classes=1)               
        elif args.arch.startswith('linknet50'):
            print('Truncated linknet50 activated')
            if args.preset in ['mul_ps_8channel','mul_8channel']:
                model = LinkNet50(num_channels=8,
                                  num_classes=1)
            else:
                model = LinkNet50(num_channels=3,
                                  num_classes=1)            
        elif args.arch.startswith('unet11'):
            if args.preset in ['mul_ps_8channel','mul_8channel']:
                model = UNet11(num_classes=1,
                               num_channels=8)            
            else:
                model = UNet11(num_classes=1,
                               num_channels=3,
                               num_filters=32)
        else:
            raise ValueError('Model not supported')

        # train on 2 GPUs for speed
        # model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()        
        
        
        if not (args.predict or args.predict_train):
            print('Train images: {}\n'
                  'Train  masks: {}\n'
                  'Val   images: {}\n'
                  'Val    masks: {}\n'.format(len(bit8_imgs[fold[0]]),len(bit8_masks[fold[0]]),
                                              len(bit8_imgs[fold[1]]),len(bit8_masks[fold[1]])))
        else:
            print('Predict images: {}\n'.format(len(predict_imgs)))         

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume + '_fold{}'.format(i) + '_best.pth.tar'):
                print("=> loading checkpoint '{}'".format(args.resume + '_fold{}'.format(i) + '_best.pth.tar'))
                checkpoint = torch.load(args.resume + '_fold{}'.format(i) + '_best.pth.tar' )
                args.start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint['best_val_loss']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.evaluate, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume + '_fold{}'.format(i) + '_best.pth.tar'))

        cudnn.benchmark = True

        if not (args.predict or args.predict_train):

            train_augs = SatellitesTrainAugmentation(shape=args.imsize,
                                                     aug_scheme = args.augs)

            val_augs = SatellitesTestAugmentation(shape=args.imsize)
           
            train_dataset = SatellitesDataset(preset = preset_dict[args.preset],
                                              image_paths = bit8_imgs[fold[0]],
                                              mask_paths = bit8_masks[fold[0]],
                                              transforms = train_augs,
                                             )

            val_dataset = SatellitesDataset(preset = preset_dict[args.preset],
                                            image_paths = bit8_imgs[fold[1]],
                                            mask_paths = bit8_masks[fold[1]],
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

        else:
            
            print('TTA settings are vflip : {}'.format(args.vflip))
            print('                 hflip : {}'.format(args.hflip))
            
            predict_augs = SatellitesTestAugmentationTTA(padding=6,
                                                         hflip = args.hflip,
                                                         vflip = args.vflip
                                                         )    

            predict_dataset = SatellitesDataset(preset = preset_dict[args.preset],
                                            image_paths = predict_imgs,
                                            mask_paths = None,
                                            transforms = predict_augs,
                                           )          
            # predict loader loads the images sequentially
            predict_loader = torch.utils.data.DataLoader(
                predict_dataset,
                batch_size=args.batch_size,        
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True)    

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
        if args.evaluate:
            validate(val_loader, model, criterion)
            return

        if args.predict or args.predict_train:
            
            if args.hflip == True and args.vflip == True:
                tta_prefix = '_hflip1_vflip1'
            elif args.hflip == True:
                tta_prefix = '_hflip1_vflip0'
            elif args.vflip == True:
                tta_prefix = '_hflip0_vflip1'
            else:
                tta_prefix = '_hflip0_vflip0'
            
            predict(predict_loader,
                    model,
                    predict_imgs,
                    predict_city_folders,
                    predict_img_names,
                    predict_prefix,
                    str(i)+tta_prefix)
        else:
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
                'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber + '_fold{}'.format(i))),
                'weights/{}_best.pth.tar'.format(str(args.lognumber + '_fold{}'.format(i)))
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
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
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
    for i, (input, target) in enumerate(val_loader):
        
        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        
        
        #============ TensorBoard logging ============#              
        # Show original images
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'images': to_np(input.view(-1,len(preset_dict[args.preset]['channels']),args.imsize, args.imsize)[:5])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, train_minib_counter)
        # Show masks
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'masks': to_np(target.view(-1,args.imsize, args.imsize)[:5])
                }
                for tag, images in info.items():
                    logger.image_summary(tag, images, train_minib_counter)                  
        # Show the output masks
        if args.tensorboard_images:
            if i % args.print_freq == 0:
                info = {
                    'preds': to_np(output.data.view(-1,args.imsize, args.imsize)[:5])
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
            predict_prefix,
            fold_number):
    
    global valid_minib_counter
    global logger
    
    print('Starting to do the predictions')
    c = 0
    # switch to evaluate mode
    model.eval()
    
    print(len(predict_img_names))

    with tqdm.tqdm(total=len(predict_loader)) as pbar:
        for i, (input) in enumerate(predict_loader):

            input = input.float().cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)

            # compute output
            output = model(input_var)
            
            for pred_image in output:

                prediction_folder = os.path.join(predict_prefix,predict_city_folders[c],args.lognumber + '_fold{}'.format(fold_number))
                
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