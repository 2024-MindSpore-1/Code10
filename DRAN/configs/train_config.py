import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# fmt: off
def build_parser():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)

    ### env
    parser.add_argument('--logger_name', type=str, default='mindcv_test')
    parser.add_argument('--log_output', type=str, default='./')

    # data augmentation
    parser.add_argument('--image_size', default=(256,256))
    parser.add_argument('--sample', type=int, default=30, help='# frames sampled from each video')
    parser.add_argument('--crop_ratio', type=float, default=1.3, help='ratio to resize the crop bounding box')
    parser.add_argument('--data_augmentation', type=str, default='group1', choices=['group1', 'group2', 'test'])
    
    # loss
    parser.add_argument('--loss_list', type=list, default=['NormInNorm']) 
    parser.add_argument('--loss_weight', type=list, default=[1])
    # choices for items in loss_list can be 'MSE', 'RankL1', 'NormInNorm', 'KLloss', 'NormInNorm+KL'.
    # loss_weight should be list with the same length as loss_list, each item in this list is a weight for the corresponding loss term
    # the total loss will be a weighted sum of loss_weight[i]*loss_list[i]
    
    # optimizer
    parser.add_argument('--opt', type=str, default='Adam', choices=['Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--betas', type=list, default=[0.9, 0.99])

    # scheduler
    parser.add_argument('--scheduler', type=str, default='WarmupCos', choices=['MultiStep', 'WarmupCos'])
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--earlystop_epoch', type=int, default=-1, help='set to -1 to disable')
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--mile_stone', type=list, default=[3,6,9,12], help='mile stones for multistep scheduler')
    
    # train
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_epoches', type=int, default=200)
    

    #args = parser.parse_args()
    return parser