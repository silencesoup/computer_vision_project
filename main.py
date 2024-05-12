import time
import shutil

import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder

from utils.functions import *
from utils.image_preprocess import *
from utils.lr_schedule import adjust_learning_rate
from utils.mixup import *
from model_params_flops import *
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch Cross Domain Training')
parser.add_argument('--data', default='./data', type=str, metavar='N',
                    help='root directory of dataset where directory train_data or val_data exists')
parser.add_argument('--result', default='./Results',
                    type=str, metavar='N', help='root directory of results')

# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet30',
                    help='model architecture: resnet30')
parser.add_argument('--num-classes', default=50, type=int, help='define the number of classes')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH', help='optionally resume from a checkpoint (default: none)')

# train
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr-type', default='step', type=str, metavar='LR',
                    help='different learning rate schedule(default:step)')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--milestones', default=[100, 140], type=int, nargs='+',
                    help='milestones for MultiStepLR')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), used for train and validation')
# optimizer
parser.add_argument('--optimizer', default='SGD', type=str, metavar='M', help='optimization method')

# Misc
parser.add_argument('--workers', default=8,type=int, metavar='N',
                    help='number of data loading workers(for linux:default 8;for Windows default 0)')
parser.add_argument('--print-freq', '-p', default=300, type=int, metavar='N',
                    help='print frequency (default: 300)')
parser.add_argument('--save-freq', '-sp', default=100, type=int, metavar='N',
                    help='save checkpoint frequency (default: 100)')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use.')#

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.start_epoch = 0
    # mkdir a new folder to store the checkpoint and best model
    args.result = os.path.join(args.result, args.arch + '_lr_{}'.format(args.lr))
    print(args)
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # Model building
    print('=> Building model...')
    modeltype = globals()[args.arch]
    model = modeltype(num_classes=args.num_classes)
    print(model)
    print(args)
    # compute the parameters and FLOPs of model
    model_params_flops(args.arch)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            if args.cuda:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_load_state_dict = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        print('GPU mode! ')
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    else:
        print('CPU mode! Cuda is not available!')

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'custom':
        """
            You can achieve your own optimizer here
        """
        pass
    else:
        raise KeyError('optimization method {} is not achieved')

    if args.resume:
        if os.path.isfile(args.resume):
            optimizer.load_state_dict(optimizer_load_state_dict)

    # Data loading and preprocessing
    print('=> Loading the DLUT_VLG_Cross_Domain data...')
    train_transforms, val_transforms = transforms_train_val()
    train_dir = os.path.join(args.data, 'train')
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    val_dir = os.path.join(args.data, 'val')
    val_dataset = ImageFolder(val_dir, transform=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    


    stats_ = stats(args.result, args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        print('learning rate:{}'.format(optimizer.param_groups[0]['lr']))
        # train for one epoch
        trainObj, top1, top5 = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        valObj, prec1, prec5 = validate(val_loader, model, criterion)

        # update stats
        stats_._update(trainObj, top1, top5, valObj, prec1, prec5)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = []
        filename.append(os.path.join(args.result, 'checkpoint.pth.tar'))
        filename.append(os.path.join(args.result, 'model_best.pth.tar'))
        stat = {'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()}
        save_checkpoint(stat, is_best, filename)
        if int(epoch+1) % args.save_freq == 0:
            print("=> save checkpoint_{}.pth.tar'".format(int(epoch + 1)))
            save_checkpoint(stat, False,
                            [os.path.join(args.result, 'checkpoint_{}.pth.tar'.format(int(epoch + 1)))])
        #plot curve
        plot_curve(stats_, args.result, True)
        data = stats_
        sio.savemat(os.path.join(args.result, 'stats.mat'), {'data': data})



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda :
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # mix up
        inputs, targets_a, targets_b, lam = mixup_data(input, target, 1., use_cuda=True)
        # inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)

        # compute the output
        output = model(inputs)

        # compute the loss
        loss= mixup_criterion(criterion,output,targets_a, targets_b, lam)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top1.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg








if __name__=='__main__':
    main()

