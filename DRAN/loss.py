import numpy as np

import mindspore as ms
import mindcv as cv

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as ops

from uilts.count import PearsonCorrelation

sign = ops.Sign()
abs = ops.Abs()
relu = ops.ReLU()
sqrt = ops.Sqrt()

class LossList(nn.Cell):
    def __init__(self, loss_list, weight_list, batch_size=None):
        super(LossList, self).__init__()
        self.weight_list = weight_list
        self.loss_list = []
        for l in loss_list:
            if l=='RankL1':
                self.loss_list.append(L1RankLoss(l1_w=1.0, rank_w=1.0, use_plcc=0.5))
            elif l=='NormInNorm':
                self.loss_list.append(NormInNormLoss(batch_size))
            elif l=='KLloss':
                self.loss_list.append(nn.KLDivLoss())
            elif l=='NormInNorm+KL':
                self.loss_list.append(NormInNormKLLoss())
            elif l=='MSE':
                self.loss_list.append(nn.MSELoss())
            else:
                print('unknow loss!')

    def construct(self, preds, gts):
        total_loss = 0
        for i, loss in enumerate(self.loss_list):
            total_loss += loss(preds, gts)*self.weight_list[i]

        return total_loss


class L1RankLoss(nn.Cell):
    def __init__(self, l1_w=1, rank_w=1, hard_thred=1, use_plcc=0):
        super(L1RankLoss, self).__init__()
        self.l1_w = l1_w
        self.rank_w = rank_w
        self.hard_thred = hard_thred
        self.use_plcc = use_plcc
        self.l1_loss = nn.L1Loss()
        self.pearsonr = PearsonCorrelation()
        

    def construct(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = self.l1_loss(preds, gts) * self.l1_w

        # simple rank
        n = len(preds)
        preds_mat = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds_mat.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = sign(img_label - img_label_t)
        masks_hard = (abs(img_label - img_label_t) < self.hard_thred) & (abs(img_label - img_label_t) > 0)
        rank_loss = masks_hard * relu(- masks * (preds_mat - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w
        if self.use_plcc != 0:
            #plcc_loss = 1-abs(self.pearsonr(preds.squeeze(), gts))
            plcc_loss = 1-self.pearsonr(preds.squeeze(), gts)
            loss_total += plcc_loss * self.use_plcc
        return loss_total  


class NormInNormLoss(nn.Cell):
    def __init__(self, batch_size, alpha=[1, 1], p=2, q=2, exponent=True):
        super(NormInNormLoss, self).__init__()
        self.alpha = alpha
        self.p = p
        self.q = q
        self.exponent = exponent
        self.eps = 1e-8
        self.batch_size = batch_size
        self.scale = np.power(2, np.max([1,1./self.q])) * np.power(batch_size, np.max([0,1./self.p-1./self.q])) 
        
    def norm_loss_with_normalization(self, y_pred, y):
        """norm_loss_with_normalization: norm-in-norm"""
        #N = y_pred.shape[0]
        if self.batch_size > 1:
            y_pred = y_pred - y_pred.mean()  # very important!!
            normalization = ms.ops.norm(y_pred, ord=self.q)# Actually, z-score normalization is related to q = 2.
            y_pred = y_pred / (self.eps + normalization)  # very important!
            y = y - y.mean()
            y = y / (self.eps + ms.ops.norm(y, ord=self.q))
            #scale = np.power(2, np.max([1,1./self.q])) * np.power(self.batch_size, np.max([0,1./self.p-1./self.q])) # p, q>0

            loss0, loss1 = 0, 0
            if self.alpha[0] > 0:
                err = y_pred - y
                if self.p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                    err += self.eps
                loss0 = ms.ops.norm(err, ord=self.p) / self.scale  # Actually, p=q=2 is related to PLCC
                #loss0 = loss0/scale
                loss0 = ms.ops.pow(loss0, exponent=self.p) if self.exponent else loss0 #
            if self.alpha[1] > 0:
                rho =  ms.ops.cosine_similarity(y_pred.t(), y.t())  #
                err = rho * y_pred - y
                if self.p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                    err += self.eps
                loss1 = ms.ops.norm(y_pred, ord=self.p) / self.scale  # Actually, p=q=2 is related to LSR
                loss1 = ms.ops.pow(loss1, exponent=self.p) if self.exponent else loss1 #  #
            
            return (self.alpha[0] * loss0 + self.alpha[1] * loss1) / (self.alpha[0] + self.alpha[1])
        else:
            return ms.ops.l1_loss(y_pred, y_pred)  # 0 for batch with single sample.

    def monotonicity_regularization(self, y_pred, y):
        """monotonicity regularization"""
        if y_pred.shape[0] > 1:  #
            ranking_loss = ms.ops.relu((y_pred-y_pred.t()) * sign((y.t()-y)))
            scale =  1 + ms.ops.max(ranking_loss)[0]
            return ms.ops.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0]-1) / scale
        else:
            return ms.ops.l1_loss(y_pred, y_pred)  # 0 for batch with single sample.

    def construct(self, y_pred, y):
        y_pred = y_pred.view(-1,1)
        y = y.view(-1,1)

        loss = self.norm_loss_with_normalization(y_pred, y)
        #loss = 0.1 * self.monotonicity_regularization(y_pred, y)
        #loss=1.
        return loss


class NormInNormKLLoss(nn.Cell):
    def __init__(self):
        super(NormInNormKLLoss, self).__init__()
        self.NormInNorm = NormInNormLoss()
        self.kl = nn.KLDivLoss()
    
    def construct(self, y_pred, y):
        loss = self.NormInNorm(y_pred, y)
        aa = y_pred + y_pred.min().abs() + 1e-4
        pred_ = (abs(aa) / abs(aa).sum()).log()
        label_ = (y / y.sum())
        loss_kl = self.kl(pred_, label_)
        loss += loss_kl
        return loss


if __name__=='__main__':
    pred = Tensor([1.1, 2.1, 3.2])
    gt = Tensor([2.1, 1.1, 4.3])
    loss = NormInNormKLLoss()
    print(loss(pred, gt))
    #print('plcc = ', pearsonr(pred.asnumpy(), gt.asnumpy()))