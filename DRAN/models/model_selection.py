import os

import mindspore as ms
import mindcv as cv
import mindspore.nn as nn
from mindcv.models import create_model

from models import swintransformerv2 as sv2



def creat_model(args):
    if args.model_name=='SwinTransformerV2':
        model = sv2.SwinTransformerV2(in_channels=3, num_classes=1, window_size=16, embed_dim=192, depths=[2,2,18,2], num_heads=[6, 12, 24, 48])
        if args.ckpt is not None:
            ckpt = ms.load_checkpoint(args.ckpt)
            _, _ = ms.load_param_into_net(model, ckpt)

    elif args.model_name=='ConvNext':
        model = create_model(
            model_name='convnext_tiny',
            num_classes=1,
            in_channels=3,
            drop_rate=None,
            drop_path_rate=None,
            checkpoint_path=args.ckpt,
            ema=False,
        )

    return model




class ModelWithLoss(nn.Cell):
    def __init__(self, net, loss_fn):
        super(ModelWithLoss, self).__init__(auto_prefix=False)
        self.network = net
        self.loss_fn = loss_fn

    def construct(self, img, gt_score):
        pred = self.network(img)
        loss = self.loss_fn(pred, gt_score)
        return loss