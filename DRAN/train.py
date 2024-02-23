import logging
import argparse

import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.dataset.transforms import Compose



from mindcv.utils import set_logger

from configs import train_config, data_config, model_config
from models import model_selection
from datasets import DFGCFrameDataset, build_transform
from loss import LossList
from uilts.count import EvaluationCallback, EvaluationCorrelation, EarlyStopCallback
from uilts.lr_scheduler import MultiStepLRCallback, WarmupCosineSchedule

import pdb

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mindcv.train')


def train(args):

    set_logger(name=args.logger_name, output_dir=args.log_output, color=False)
    logger.info("test log, starting training scrip...")

    # create dataset
    dataset_generator = DFGCFrameDataset(args.root_dir, args.label_file, seq=False, 
                                         sample=args.sample, extra_data=args.extra_data)
    dataset = ds.GeneratorDataset(dataset_generator, ['img', 'mos'], shuffle=True, python_multiprocessing=True)
    train_set, val_set = dataset.split([0.8, 0.2])

    train_transform = Compose(build_transform(input_size=args.image_size, aug=args.data_augmentation))
    val_transform = Compose(build_transform(input_size=args.image_size, aug=args.data_augmentation))
    train_set = train_set.map(operations=train_transform, input_columns=['img'])
    train_set = train_set.batch(batch_size=args.batch_size, drop_remainder=True)     # keep drop_remainder=True if using NormInNorm loss
    val_set = val_set.map(operations=val_transform, input_columns=['img'])
    val_set = val_set.batch(batch_size=args.batch_size, drop_remainder=True)   
    
    num_batches = train_set.get_dataset_size()
    logger.info('dataset loaded, train set size:%d, val set size:%d' % (train_set.get_dataset_size(), val_set.get_dataset_size()))

    pdb.set_trace()

    # creat model
    network = model_selection.creat_model(args)
    logger.info('model loaded...')
    

    # create loss
    loss = LossList(args.loss_list, args.loss_weight, args.batch_size)

    # create learning rate schedule
    if args.scheduler=='MultiStep':
        lr_scheduler = MultiStepLRCallback(milestones=args.mile_stone, gamma=args.gamma)
    elif args.scheduler=='WarmupCos':
        total_steps = num_batches*args.total_epoches
        warmup_step = num_batches*args.warmup_epoch
        start_step = num_batches*args.start_epoch
        lr_scheduler = WarmupCosineSchedule(warmup_step, total_steps, args.lr, start_step=start_step)
    #pdb.set_trace()

    # create optimizer
    if args.opt=='Adam':
        opt = nn.Adam(params=network.trainable_params(), learning_rate=args.lr, 
                        weight_decay=args.weight_decay, beta1=args.betas[0], beta2=args.betas[1])
    elif args.opt=='AdamW':
        opt = nn.AdamWeightDecay(params=network.trainable_params(), learning_rate=args.lr,
                                weight_decay=args.weight_decay, beta1=args.betas[0], beta2=args.betas[1])        

    
    # train model
    model = ms.Model(network, loss_fn=loss, optimizer=opt, 
                     metrics={'MSE':ms.train.MSE(), 'corr':EvaluationCorrelation()}, 
                     amp_level="O2")

    # checkpoint settings
    ckpt_save_dir = './ckpt/save_test1'
    ckpt_config = ms.CheckpointConfig(save_checkpoint_steps=int(num_batches*0.5))
    ckpt_cb = ms.ModelCheckpoint(prefix='model_sv2',
                            directory=ckpt_save_dir,
                            config=ckpt_config)

    # evaluation settings
    eval_cb = EvaluationCallback(model, val_set)

    # early stop settings, the training stops after certain steps 
    early_stop_cb = EarlyStopCallback(args.earlystop_epoch)

    # training
    model.train(args.total_epoches, train_set, 
                callbacks=[ms.LossMonitor(num_batches), ms.TimeMonitor(num_batches), 
                           lr_scheduler, eval_cb, ckpt_cb, early_stop_cb], 
                dataset_sink_mode=False)
    #model.eval(val_set, callbacks=[ms.LossMonitor(num_batches//5)], dataset_sink_mode=False)



if __name__=='__main__':
    ms.set_seed(42)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')

    train_args = train_config.build_parser()
    data_args = data_config.build_parser()
    model_args = model_config.build_parser()
    parser = argparse.ArgumentParser(parents=[train_args, data_args, model_args])
    args = parser.parse_args()

    train(args)