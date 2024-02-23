from __future__ import absolute_import

import math
import mindspore as ms
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.train.callback._callback import Callback
from mindspore.ops import functional as F

class MultiStepLRCallback(Callback):
    '''
    Change the learning rate whenever epoch numbers reach a milestone
    milestones should be a list
    '''
    def __init__(self, milestones, gamma):
        super().__init__()
        self.milestones = milestones
        self.gamma = gamma
        self.current_place = 0
        
    def on_train_begin(self, run_context):
        self.current_place = 0


    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        arr_lr = cb_params.optimizer.learning_rate.asnumpy()
        lr = float(np.array2string(arr_lr))
        
        new_lr = lr
        if self.current_place+1<=len(self.milestones):
            if cb_params.cur_epoch_num==self.milestones[self.current_place]:
                self.current_place +=1
                new_lr = lr * self.gamma

        if not math.isclose(lr, new_lr, rel_tol=1e-10):
            F.assign(cb_params.optimizer.learning_rate, Tensor(new_lr, mstype.float32))
            print(f'At epoch {cb_params.cur_epoch_num}, learning_rate change to {new_lr}')


class WarmupCosineSchedule(Callback):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
        t_total = num_batchs*total_epochs
    """
    def __init__(self, warmup_steps, total_step, max_lr, cycles=.5, start_step=0,):
        super(WarmupCosineSchedule, self).__init__()
        self.warmup_steps = warmup_steps
        self.total_step = total_step
        self.max_lr = max_lr
        self.cycles = cycles
        self.start_step = start_step

    def on_train_begin(self, run_context):
        self.current_step = self.start_step


    def on_train_step_end(self, run_context):
        if self.current_step < self.warmup_steps:
            p =  float(self.current_step)/max(1.0, float(self.warmup_steps))
        
        else:
            p_cos = float(self.current_step - self.warmup_steps) / float(max(1., self.total_step - self.warmup_steps))
            p = max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * p_cos)))

        cb_params = run_context.original_args()
        F.assign(cb_params.optimizer.learning_rate, Tensor(p*self.max_lr, mstype.float32))