
def adjust_learning_rate(args, optimizer, epoch):
    """
     For AlexNet, the lr starts from 0.05, and is divided by 10 at 90 and 120 epochs
    """
    if args.lr_type == 'step':
        if epoch < args.milestones[0]:
            lr = args.lr
        elif epoch < args.milestones[1]:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    elif args.optimizer == 'custom':
        """
            You can achieve your own learning_rate schedule here
        """
        pass
    else:
        raise KeyError('learning_rate schedule method {} is not achieved')

