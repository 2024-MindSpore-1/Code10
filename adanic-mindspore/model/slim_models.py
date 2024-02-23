import copy
#import torchvision
import mindspore.nn as nn
#import torch.nn.functional as F
import mindspore.ops as ops
from compressai.models.priors import MeanScaleHyperprior
from models.slim_ops import AdaConv2d, AdaConvTranspose2d, SlimGDNPlus
from models.mlp_mixer import mlp_backbone, mlp_mixer_backbone

conv = lambda c_in, c_out, kernel_size=5, stride=2, in_shape_static=False, out_shape_static=False, M_mapping=None : AdaConv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, M_mapping=M_mapping, in_shape_static=in_shape_static, out_shape_static=out_shape_static)
deconv = lambda c_in, c_out, kernel_size=5, stride=2, in_shape_static=False, M_mapping=None : AdaConvTranspose2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, M_mapping=M_mapping, output_padding=stride-1, in_shape_static=in_shape_static)


def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)


class AdaMeanScaleHyperprior(MeanScaleHyperprior):
    def __init__(self, N_list, M_list, **kwargs):
        super().__init__(N_list[-1], M_list[-1], **kwargs)
        self.N_list = N_list
        self.M_list = sorted(list(set(M_list)))
        self.M_mapping = M_mapping = [self.M_list.index(x) for x in M_list]
        M_list = self.M_list
        self.g_a = nn.SequentialCell([
            conv(3, N_list),  # 3 -> N
            SlimGDNPlus(N_list),
            conv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list),
            conv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list),
            conv(N_list, M_list, out_shape_static=True, M_mapping=M_mapping),  # N -> M
        ])

        self.g_s = nn.SequentialCell([
            deconv(M_list, N_list, in_shape_static=True, M_mapping=M_mapping),  # M -> N
            SlimGDNPlus(N_list, inverse=True),
            deconv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list, inverse=True),
            deconv(N_list, N_list),  # N -> N
            SlimGDNPlus(N_list, inverse=True),
            deconv(N_list, 3),  # N -> 3
        ])
        
        self.h_a = nn.SequentialCell([
            conv(M_list, N_list, stride=1, kernel_size=3, in_shape_static=True, M_mapping=M_mapping), # M -> N (fake M[-1])
            nn.LeakyReLU(),
            conv(N_list, N_list), # N -> N
            nn.LeakyReLU(),
            conv(N_list, N_list, out_shape_static=True), # N -> N (fake N[-1])
        ])

        self.h_s = nn.SequentialCell([
            deconv(N_list, M_list, in_shape_static=True, M_mapping=M_mapping), # N -> M (fake N[-1])
            nn.LeakyReLU(),
            deconv(M_list, [k * 3 // 2 for k in M_list], M_mapping=M_mapping), # M -> M
            nn.LeakyReLU(),
            conv([k * 3 // 2 for k in M_list], [k * 2 for k in M_list], stride=1, kernel_size=3, out_shape_static=True, M_mapping=M_mapping), # M -> M (fake M[-1])
        ])
        
    def set_running_N(self, N):
        idx = self.g_a[0].out_channels_list.index(N)
        #for n, m in self.named_modules():
        for n, m in self.cells_and_names():
            if hasattr(m, "out_channels_list") and len(m.out_channels_list) > 1:
                # print(n)
                set_exist_attr(m, "channel_choice", idx)
    
    def set_running_width(self, N):
        self.set_running_N(N)
