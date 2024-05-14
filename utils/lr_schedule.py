import math
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
            
    elif args.lr_type == 'custom':
        # 实现余弦退火
        lr_min = 0  # 设定最小学习率
        lr_max = args.lr  # 最大学习率为初始学习率
        T_max = args.epochs  # 总周期数设为总epoch数
        
        # 计算当前epoch的学习率
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / T_max * math.pi))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        raise KeyError('learning_rate schedule method {} is not achieved')

