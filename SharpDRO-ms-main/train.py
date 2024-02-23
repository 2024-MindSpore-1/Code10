import os
import time
from sam import SAM
from bypass_bn import disable_running_stats, enable_running_stats
import step_lr
import mindspore as ms
from mindspore import ops,nn

class Trainer:
    def __init__(self, model, logger, label_csv_logger, \
        val_csv_loggers, test_csv_loggers, num_distribution):
        self.model = model
        # concole logger
        self.logger = logger
        # csv logger
        self.label_csv_logger = label_csv_logger
        self.val_csv_logger = val_csv_loggers
        self.test_csv_logger = test_csv_loggers
        self.num_distribution = num_distribution
        self.adv_probs = ops.ones(self.num_distribution,dtype=ms.float32)/self.num_distribution
        self.grad_norm = ms.numpy.zeros(self.num_distribution)

        self.best_val_acc = 0.
        self.best_test_acc = 0.

        self.reset_stats()

    def compute_robust_loss(self, args, output, target, robust_idx, first_loss=None):
        if first_loss is not None:
            per_sample_loss = ops.cross_entropy(output, target, reduction="none") - first_loss
        else:
            per_sample_loss = ops.cross_entropy(output, target, reduction="none")

        distribution_map = (
                    robust_idx == ms.numpy.arange(self.num_distribution).unsqueeze(1).long()).float()
        distribution_count = distribution_map.sum(1)
        distribution_denom = distribution_count + (distribution_count == 0).float()  # avoid nans
        distribution_loss = (distribution_map @ per_sample_loss.view(-1)) / distribution_denom

        adjusted_loss = distribution_loss
        self.adv_probs = self.adv_probs * ops.exp(args.step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = distribution_loss @ self.adv_probs
        return robust_loss, per_sample_loss

    def compute_sharp_loss(self, args, output, target, robust_idx, first_loss):
        per_sample_sharpness = ops.cross_entropy(output, target, reduction="none") - first_loss

        distribution_map = (
                    robust_idx == ms.numpy.arange(self.num_distribution).unsqueeze(1).long()).float()
        distribution_count = distribution_map.sum(1)
        distribution_denom = distribution_count + (distribution_count == 0).float()  # avoid nans

        distribution_sharpness = (distribution_map @ per_sample_sharpness.view(-1)) / distribution_denom
        distribution_loss = (distribution_map @ first_loss.view(-1)) / distribution_denom

        adjusted_loss = distribution_loss
        self.adv_probs = self.adv_probs * ops.exp(args.step_size * adjusted_loss)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = distribution_sharpness @ self.adv_probs
        return robust_loss, first_loss

    def compute_distribution_avg(self, losses, robust_idx):
        # Refer to https://github.com/kohpangwei/group_DRO
        # compute observed counts and mean loss for each distribution
        distribution_map = (robust_idx == ms.numpy.arange(self.num_distribution).unsqueeze(1).long()).float()
        distribution_count = distribution_map.sum(1)
        distribution_denom = distribution_count + (distribution_count==0).float() # avoid nans
        distribution_loss = (distribution_map @ losses.view(-1))/distribution_denom
        return distribution_loss, distribution_count

    def reset_stats(self):
        self.avg_acc = 0.
        self.avg_loss = 0.
        self.processed_distribution_counts = ms.numpy.zeros(self.num_distribution)
        self.distribution_loss = ms.numpy.zeros(self.num_distribution)
        self.distribution_acc = ms.numpy.zeros(self.num_distribution)
        self.grad_norm = ms.numpy.zeros(1)

    def update_stats(self, per_sample_loss, output, y, robust_idx):
        distribution_loss, distribution_counts = self.compute_distribution_avg(per_sample_loss, robust_idx)

        correct = (ops.argmax(output,1)==y).float()
        distribution_acc, distribution_counts = self.compute_distribution_avg(correct, robust_idx)

        self.avg_acc = correct.mean()
        self.avg_loss = per_sample_loss.mean()
        self.processed_distribution_counts += distribution_counts
        self.distribution_loss += distribution_loss
        self.distribution_acc = distribution_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.get_parameters():
            data = param.value()
            norm_data = ms.numpy.norm(data)
            model_norm_sq += norm_data ** 2
        stats_dict['model_norm_sq'] = model_norm_sq
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq
        return stats_dict

    def get_stats(self, is_train, model=None, args=None):
        stats_dict = {}

        stats_dict['avg_acc'] = self.avg_acc

        if is_train:
            for idx in range(self.num_distribution):
                stats_dict[f'processed_distribution_counts_{idx}'] = self.processed_distribution_counts[idx]
                stats_dict[f'distribution_loss_{idx}'] = self.distribution_loss[idx]
                stats_dict[f'adv_prob_{idx}'] = self.adv_probs[idx]
                stats_dict[f'distribution_acc_{idx}'] = self.distribution_acc[idx]
        else:
            stats_dict['processed_distribution_counts_0'] = self.processed_distribution_counts[0]
            stats_dict['distribution_loss_0'] = self.distribution_loss[0]
            stats_dict['distribution_acc_0'] = self.distribution_acc[0]
            stats_dict['gradnorm_0'] = self.grad_norm


        # Model stats
        if is_train and model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_train=True):
        if logger is None:
            return

        logger.write(f'Avg acc: {self.avg_acc.asnumpy():.3f}  ')
        logger.write(f'Avg loss: {self.avg_loss.asnumpy():.3f}  \n')
        if is_train:
            for idx in range(self.num_distribution):
                logger.write(
                    f'\td_idx: {idx}  '
                    f'\t[n = {int(self.processed_distribution_counts[idx])}]:'
                    f'\tloss = {self.distribution_loss[idx].asnumpy():.3f}'
                    f'\tprobs = {self.adv_probs[idx].asnumpy():.3f}'
                    f'\tacc = {self.distribution_acc[idx].asnumpy():.3f}    \n')
        else:
            logger.write(
                f'\t[n = {int(self.processed_distribution_counts[0])}]:'
                f'\tloss = {self.distribution_loss[0].asnumpy():.3f}'
                f'\tacc = {self.distribution_acc[0].asnumpy():.3f}    ')

        logger.flush()

    def run_epoch(self, args, epoch, train_loader):
        self.model.set_train()
        first_loss_fn = nn.CrossEntropyLoss(reduction="none")
        second_loss_fn = SharpLoss(args)

        def first_forward_fn(data_x,data_y):
            output = self.model(data_x)
            loss = first_loss_fn(output,data_y)
            return loss,output

        def second_forward_fn(data_x,data_y,robust_idx,first_loss,num_distribution,adv_probs):
            output = self.model(data_x)
            robust_loss,per_sample_loss = second_loss_fn(output,data_y,robust_idx,first_loss,num_distribution,adv_probs)
            return robust_loss,per_sample_loss,output

        first_grad_fn = ms.value_and_grad(first_forward_fn,None,self.optimizer.parameters,has_aux=True)
        second_grad_fn = ms.value_and_grad(second_forward_fn,None,self.optimizer.parameters,has_aux=True)
        cast = ops.Cast()

        for it, (data_x,data_y, robust_idx) in enumerate(train_loader):
            data_y = cast(data_y,ms.int32)
            # first forward-backward step
            enable_running_stats(self.model)
            # output = self.model(data_x)
            # first_loss = first_loss_fn(output,data_y)
            (first_loss,output1),grads = first_grad_fn(data_x,data_y)

            #first step
            self.optimizer.first_step(grads)

            #second forward-backward step
            disable_running_stats(self.model)
            (robust_loss, per_sample_loss,output2),grads = second_grad_fn(data_x, data_y, robust_idx,first_loss,self.num_distribution,self.adv_probs)
            self.optimizer(grads)
            self.adv_probs = second_loss_fn.get_adv_probs()

            if it % args.log_train==0:
                self.logger.write(f'\n\nLogging iteration: {it:d}\n')
                self.update_stats(per_sample_loss, output2, data_y, robust_idx)
                self.label_csv_logger.log(epoch, it, self.get_stats(True, self.model, args))
                self.label_csv_logger.flush()
                self.log_stats(self.logger)
                self.reset_stats()

    def evaluation(self, args, epoch, eval_loaders, eval_csv_loggers):
        self.model.set_train(False)
        cast = ops.Cast()
        for d_idx, (eval_loader, eval_csv_logger) in enumerate(zip(eval_loaders, eval_csv_loggers)):

            total_distribution_count = 0.
            total_distribution_correct = 0.
            for it, (eval_x,eval_y, robust_idx) in enumerate(eval_loader):
                eval_y = cast(eval_y, ms.int32)
                outputs = self.model(eval_x)
                loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
                per_sample_loss = loss_fn(outputs, eval_y)

                # if args.record_norm and it % args.log_train == 0:
                #     self.model.train()
                #     ## record grad norm
                #     for i in range(self.num_distribution):
                #         self.optimizer.zero_grad()
                #         per_sample_loss.mean().backward(retain_graph=True)
                #         total_norm = 0
                #         for p in self.model.parameters():
                #             if p.grad is None: continue
                #             param_norm = p.grad.detach().data.norm(2)
                #             total_norm += param_norm.item() ** 2
                #         total_norm = total_norm ** 0.5
                #         self.grad_norm = total_norm
                #     self.optimizer.zero_grad()
                #     self.model.eval()

                total_distribution_correct += (ops.argmax(outputs, 1) == eval_y).float().sum()
                total_distribution_count += len(eval_y)

                if (it) % args.log_eval == 0:
                    self.update_stats(per_sample_loss, outputs, eval_y, robust_idx)
                    eval_csv_logger.log(epoch, it, self.get_stats(False))
                    eval_csv_logger.flush()
                    self.reset_stats()

            epoch_eval_acc = total_distribution_correct / total_distribution_count
            self.logger.write(f'\nDistr idx {d_idx:d}: Total Eval Acc: {epoch_eval_acc.asnumpy():.3f}\n')
        return per_sample_loss.mean(), epoch_eval_acc

    def train(self, args, labeled_loader, val_loader, test_loader):
        lr = args.lr
        step_per_epoch = labeled_loader.get_dataset_size()
        total_epoch = args.total_epoch
        steplr = step_lr.stepLR(learning_rate=lr,steps_per_epoch=step_per_epoch,total_epochs=total_epoch)

        self.optimizer = SAM(self.model.trainable_params(),steplr,base_optimizer=nn.SGD,rho=args.rho,adaptive=args.is_adaptive,momentum=args.momentum,weight_decay=args.weight_decay)

        s = time.time()
        for epoch in range(args.total_epoch):
            self.logger.write('\nEpoch [%d]:\n' % epoch)
            self.logger.write(f'Training:\n')

            self.run_epoch(args, epoch, labeled_loader)

            self.logger.write(f'\n\n\nValidation:\n')
            _, epoch_val_acc = self.evaluation(args, epoch, val_loader, self.val_csv_logger)

            self.logger.write(f'\n\nTest:\n')
            self.evaluation(args, epoch, test_loader, self.test_csv_logger)

            if epoch % args.save_step == 0:
                ms.save_checkpoint(self.model, os.path.join(args.log_dir, args.desc + '%d_model.pth' % epoch))
            if args.save_last:
                ms.save_checkpoint(self.model, os.path.join(args.log_dir, args.desc + 'last_model.pth'))
            if args.save_best:
                self.logger.write(f'Current validation accuracy: {epoch_val_acc.asnumpy()}\n')
                if epoch_val_acc[0] > self.best_val_acc:
                    self.best_val_acc = epoch_val_acc
                    ms.save_checkpoint(self.model, os.path.join(args.log_dir, 'best_model.pth'))
                    self.logger.write(f'Best model saved at epoch {epoch}\n')

            # Inspect learning rates
            if (epoch + 1) > (0.8 * args.total_epoch) == 0:
                self.logger.write('Current lr: %f\n' % self.optimizer.get_lr())
            #     for param_group in self.optimizer.param_groups:
            #         # lr decay
            #         param_group['lr'] *= 0.2
            #         curr_lr = param_group['lr']
            #         self.logger.write('Current lr: %f\n' % curr_lr)

            self.logger.write(
                f'\nRest Time: {(time.time() - s) * (args.total_epoch - epoch) / (3600 * (epoch + 1)):.2f} hrs')
            self.logger.write('\n')

def _grad_norm(params,gradients,adaptive):
    stack = ops.Stack()
    test=[]
    for i in range(len(params)):
        if gradients[i] is None:continue
        if adaptive:
            value = ops.abs(params[i])
        else:
            value = 1.0
        value = value*gradients[i]
        dims = [i for i in range(len(value.shape))]
        dims = tuple(dims)
        value = ops.norm(value,p=2,axis=dims)
        test.append(value)
    test = stack(test)
    test = ops.norm(test,p=2,axis=0)
    return test

class SharpLoss(nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.step_size = args.step_size
        self.adv_probs = None

    def get_adv_probs(self):
        return self.adv_probs

    def construct(self,output, target, robust_idx,first_loss,num_distribution,adv_probs):
        per_sample_sharpness = ops.cross_entropy(output,target,reduction="none") - first_loss
        distribution_map = (
                    robust_idx == ms.numpy.arange(num_distribution).unsqueeze(1).long()).float()
        distribution_count = distribution_map.sum(1)
        distribution_denom = distribution_count + (distribution_count == 0).float()  # avoid nans
        distribution_sharpness = (distribution_map @ per_sample_sharpness.view(-1)) / distribution_denom
        distribution_loss = (distribution_map @ first_loss.view(-1)) / distribution_denom
        adjusted_loss = distribution_loss
        adv_probs = adv_probs * ops.exp(self.step_size*adjusted_loss)
        adv_probs = adv_probs/(adv_probs.sum())
        robust_loss = distribution_sharpness @ adv_probs

        self.adv_probs = adv_probs
        return robust_loss,first_loss


