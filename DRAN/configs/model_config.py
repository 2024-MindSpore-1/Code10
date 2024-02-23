import argparse
import logging
import os


logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)

    ### model
    parser.add_argument('--model_name', type=str, default='SwinTransformerV2', choices=['ConvNext', 'SwinTransformerV2'])
    parser.add_argument('--ckpt', type=str, default='./convert/swintransformerv2/swinv2_best_from_torch.ckpt')

    #args = parser.parse_args()
    return parser