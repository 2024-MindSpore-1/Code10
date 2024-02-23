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
    parser = argparse.ArgumentParser(description='Testing Config', add_help=False)

    ### env
    parser.add_argument('--logger_name', type=str, default='mindcv_test')
    parser.add_argument('--log_output', type=str, default='./')

    ### data
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--label_dir', type=str, default='./')
    parser.add_argument('--test_phase', type=str, default='1', choices=['1', '2', '3'])
    parser.add_argument('--image_size', default=(256,256))
    parser.add_argument('--sample', type=int, default=[0.25, 0.5, 0.75], help='# frames sampled from each video')
    # if the input value for --sample is a int n, it will sample n frames.(e.g. sample=10 means to sample 10 frames for each video)
    # if the input value is a list of floats, e.g. sample=[0.25] and there are a total of 20 frames, it will sample the int(0.25*20)=5 th frame.
    parser.add_argument('--augmentation', type=str, default='test', choices=['group1_test', 'test'])
    
    
    # test
    parser.add_argument('--batch_size', type=int, default=8)
    

    #args = parser.parse_args()
    return parser