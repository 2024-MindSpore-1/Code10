

from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import ops
import mindspore as ms
import math
def stepLR(learning_rate,steps_per_epoch,total_epochs):
    total_steps = steps_per_epoch*total_epochs
    lr_list = []
    for i in range(total_steps):
        epoch = math.floor(i/ steps_per_epoch)
        # if (epoch+1) > (0.8*self.total_epochs) == 0:
        #     lr = lr*0.2
        if epoch < total_epochs * 3/10:
            lr = learning_rate
        elif epoch < total_epochs * 6/10:
            lr = learning_rate * 0.2
        elif epoch < total_epochs * 8/10:
            lr = learning_rate * 0.2 ** 2
        else:
            lr = learning_rate * 0.2 ** 3
        # self.curr_lr = lr
        lr_list.append(lr)
    return lr_list



# class StepLR(LearningRateSchedule):
#     def __init__(self, learning_rate, steps_per_epoch,total_epochs):
#         super(StepLR, self).__init__()
#         self.step_per_epoch = steps_per_epoch
#         self.base = learning_rate
#         self.total_epochs = total_epochs
#         self.curr_lr = learning_rate
#
#     def construct(self, global_step):
#         # lr = self.curr_lr
#         epoch = ops.floor(global_step / self.step_per_epoch)
#         # if (epoch+1) > (0.8*self.total_epochs) == 0:
#         #     lr = lr*0.2
#         if epoch < self.total_epochs * 3/10:
#             lr = self.base
#         elif epoch < self.total_epochs * 6/10:
#             lr = self.base * 0.2
#         elif epoch < self.total_epochs * 8/10:
#             lr = self.base * 0.2 ** 2
#         else:
#             lr = self.base * 0.2 ** 3
#         # self.curr_lr = lr
#         return lr