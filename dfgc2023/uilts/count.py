import numpy as np
import mindspore as ms
import mindcv as cv

import mindspore.nn as nn
from mindspore import Tensor, context
import mindspore.ops as ops
from mindspore import log as logger

from scipy.stats import pearsonr, spearmanr

class PearsonCorrelation(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, y):
        x = x-x.mean()
        y = y-y.mean()
        x_square_sum = x.square().sum()
        y_square_sum = y.square().sum()
        xy_sum = (x * y).sum()
        plcc = xy_sum / (ops.sqrt(x_square_sum * y_square_sum))
        return plcc


class SpearmanCorrelation(ms.nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, pred, gt):
        return spearmanr(pred.asnumpy(), gt.asnumpy())[0]

class EvaluationCorrelation(ms.train.Metric):
    def __init__(self):
        super(EvaluationCorrelation).__init__()
        self.clear()
        self.plcc = PearsonCorrelation()
        self.srcc = SpearmanCorrelation()

    def clear(self):
        self.pred = []
        self.gt = []

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError("For correlation evaluation, it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self.pred.append(y_pred)
        self.gt.append(y)


    def eval(self):
        pred_tensor = Tensor(self.pred, dtype=ms.float32).view(-1)
        gt_tensor = Tensor(self.gt, dtype=ms.float32).view(-1)
        plcc = self.plcc(pred_tensor, gt_tensor)
        #srcc = self.srcc(pred_tensor, gt_tensor)
        return {'PLCC':plcc.asnumpy()}


class EarlyStopCallback(ms.train.Callback):
    '''
    set early_stop_epoch=-1 to disable
    '''

    def __init__(self, early_stop_epoch):
        super().__init__()
        self.total_steps = 0
        self.early_stop_epoch = early_stop_epoch

    def on_train_begin(self, run_context):
        self.total_steps=0
    
    def on_train_step_end(self, run_context):
        self.total_steps += 1

        if self.total_steps>self.early_stop_epoch and self.early_stop_epoch>0:
            run_context.request_stop()
            print(f'early stop at the {self.total_steps-1} th step')


class EvaluationCallback(ms.train.Callback):
    def __init__(self, model, eval_dataset):
        super(EvaluationCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset


    def on_train_epoch_end(self, run_context):
        #pred = run_context.net.metrics["pred"]
        #gt = run_context.net.metrics["gt"]
        eval_result = self.model.eval(self.eval_dataset)
        print('this is evaluation result:', eval_result)
        #print(run_context.original_args()['net_outputs'])
        #self.pearson_coef = ms.nn.pearson_corrcoef(pred, gt)[0][1]
        #self.spearman_coef = ms.nn.spearman_corrcoef(pred, gt)[0][1]
        return 1.



if __name__=='__main__':
    pred = Tensor(np.random.random((10)).astype(np.float32))
    gt = Tensor(np.random.random((10)).astype(np.float32))
    
    plcc = PearsonCorrelation()
    srcc = SpearmanCorrelation()

    print(srcc(pred, gt))
    print(spearmanr(pred.asnumpy(), gt.asnumpy()))

    print(plcc(pred, gt))
    print(pearsonr(pred.asnumpy(), gt.asnumpy()))