import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore.dataset.transforms import Compose

from mindcv.models import create_model

import pandas as pd
from tqdm import tqdm
import os
import argparse

from configs import test_config, data_config, model_config
from datasets import DFGCFrameTestDataset, build_transform
from uilts.view import view_pred_result
from models import model_selection



def predict(args):
    # create dataset
    label_file = os.path.join(args.label_dir, 'test_set'+args.test_phase+'.txt')

    dataset_generator = DFGCFrameTestDataset(args.root_dir, label_file, seq=False, sample=args.sample)
    dataset = ds.GeneratorDataset(dataset_generator, ['img', 'path'], shuffle=False, python_multiprocessing=True)

    transform_list = Compose(build_transform(input_size=args.image_size, aug=args.augmentation))
    dataset = dataset.map(operations=transform_list, input_columns=['img'])
    dataset = dataset.batch(batch_size=args.batch_size, drop_remainder=False)

    num_batches = dataset.get_dataset_size()
    print(f'test set size: {num_batches}, with batch size {args.batch_size}')

    # create model
    network = model_selection.creat_model(args)
    print('model loaded...')

    model = ms.Model(network)

    image_path_list = []
    pred_list = []
    for batchdata in tqdm(dataset.create_tuple_iterator()):
        pred = list(model.predict(batchdata[0]).asnumpy().squeeze())
        path = list(batchdata[1].asnumpy())
        
        pred_list += pred
        image_path_list += path
    
    view_pred_result(image_path_list, pred_list, label_file, args.root_dir, output_path='./pred/', phase=args.test_phase)
    


if __name__=='__main__':
    ms.set_seed(42)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')
    test_args = test_config.build_parser()
    model_args = model_config.build_parser()
    parser = argparse.ArgumentParser(parents=[test_args, model_args])
    args = parser.parse_args()
    predict(args)