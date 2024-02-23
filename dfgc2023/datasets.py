import numpy as np
import pandas as pd
import random
import re

from PIL import Image

import mindspore as ms
from mindspore import Tensor
import mindspore.dataset as ds
import mindspore.dataset.vision as py_trans
from mindspore.dataset.transforms import Compose

from uilts.aug import group1_aug_train

import os
import pdb



'''
data structure:
--root_dir
    --C1_frame
        --1-1-1-submit-00000
            --001.png
            --002.png
    --C2_frame
    --C3_frame

--label_dir
    --train_set.csv
'''

MEAN=(0.485*255., 0.456*255., 0.406*255.)
STD=(0.229*255., 0.224*255., 0.225*255.)


def build_transform(input_size=(224,224), aug='test'):
    transforms_list = [py_trans.Resize(size=input_size)]
    
    if aug=='group1':
        transforms_list.extend([
            py_trans.RandomErasing(scale=(0.01, 0.05), ratio=(0.5, 0.5), value=0),
            py_trans.HorizontalFlip(),
            #AddColor,
            #py_trans.Affine(),
            ])
    elif aug=='group1_test':    # test time augmentation
        transforms_list.extend([
            py_trans.HorizontalFlip(),
        ])

    elif aug=='group2' :
        transforms_list.extend([
            py_trans.HorizontalFlip(),
            py_trans.Affine(degrees=5, translate=[0.04, 0.04], scale=1.04, shear=0),
            py_trans.Rotate(degrees=90),])
    

    elif aug=='test':
        pass

    transforms_list.extend([py_trans.Normalize(mean=MEAN, std=STD),
                            py_trans.HWC2CHW(),
                            py_trans.ToType(ms.float32)])
    return transforms_list

def build_image_dataset(root_dir, video_list, mos_list=None, sample=5):
    
    image_list = []
    mos_list_image = []
    for i, video in enumerate(video_list):
        images = os.listdir(os.path.join(root_dir, video))
        images = [img for img in images if (img[-4:]=='.png' or img[-4:]=='.jpg')]
        images.sort()
            
        if isinstance(sample, int):
            if len(images)<sample: sample = len(images)
            image_sampled = random.sample(images, sample)
            for image in image_sampled: 
                image_list.append(os.path.join(root_dir, video, image))
                if mos_list is not None:
                    mos_list_image.append(mos_list[i])
        elif isinstance(sample, list):
            for p in sample:
                idx_sampled = int(len(images)*p)
                image_list.append(os.path.join(root_dir, video, images[idx_sampled]))
                if mos_list is not None:
                    mos_list_image.append(mos_list[i])
    

    return image_list, mos_list_image    



class DFGCFrameDataset():

    def __init__(self, root_dir, label_file, crop_file=None, crop_ratio=1.0, sample=5, extra_data=None, aug='test'):
        self.aug = aug
        self.root_dir = root_dir
        self.crop_df = pd.read_csv(crop_file) if crop_file is not None else None
        self.crop_ratio = crop_ratio

        label_dfgc = pd.read_csv(label_file)
        video_list_dfgc = list(label_dfgc['file'])
        video_list_dfgc = [v.replace('.mp4', '').replace('/', '_frame/') for v in video_list_dfgc]
        mos_list_dfgc = list(label_dfgc['mos'])

        self.video_list = []
        self.img_list = []
        self.mos_list = []

        # proceed DFGC data
        image_list_dfgc, mos_image_list_dfgc = build_image_dataset(root_dir, video_list_dfgc, mos_list_dfgc, sample=sample)
        self.video_list += video_list_dfgc
        self.img_list += image_list_dfgc
        self.mos_list += mos_image_list_dfgc

        # proceed extra data
        if extra_data is not None:
            extra_file = pd.read_csv(extra_data)
            video_list_extra = list(extra_file['file'])
            mos_list_extra = list(extra_file['mos'])
            
            image_list_extra, mos_image_list_extra = build_image_dataset(root_dir, video_list_extra, mos_list_extra, sample=sample)
            self.video_list += video_list_extra
            self.img_list += image_list_extra
            self.mos_list += mos_image_list_extra


    def __getitem__(self, index):

        img_path = self.img_list[index]
        img = Image.open(img_path)
        if self.aug == 'group1':
            img = group1_aug_train(img)
        
        if self.crop_df is not None:
            img_path = img_path.replace('_frame', '').replace(self.root_dir, '')
            crop_idx = list(self.crop_df['image_path']).index(img_path)
            x = list(self.crop_df['x'])[crop_idx]
            y = list(self.crop_df['y'])[crop_idx]
            w = list(self.crop_df['w'])[crop_idx]
            h = list(self.crop_df['h'])[crop_idx]

            if x>1:
                xc, yc = x+w/2, y+h/2
                bbox = (xc-w*self.crop_ratio/2, yc-w*self.crop_ratio/2, xc+w*self.crop_ratio/2, yc+h*self.crop_ratio/2)
            else:
                weight, hight = img.size
                w = w*weight*self.crop_ratio/2
                h = h*hight*self.crop_ratio/2
                xc, yc = x*weight+w/2, y*hight+h/2
                bbox = (xc-w/2, yc-h/2, xc+w/2, yc+h/2)
            
            img = img.crop(bbox)

        if len(self.mos_list)==len(self.img_list): label = self.mos_list[index]
        else: label=None
    
        return (img, label)


    def __len__(self):
        return len(self.img_list)


# for test data without label, also return img path for futher score fusion
class DFGCFrameTestDataset():

    def __init__(self, root_dir, label_file, crop_file=None, crop_ratio=1.0, sample=5):
        self.root_dir = root_dir
        self.crop_df = pd.read_csv(crop_file) if crop_file is not None else None
        self.crop_ratio = crop_ratio

        label_dfgc = pd.read_csv(label_file, names=['file'])
        video_list_dfgc = list(label_dfgc['file'])
        video_list_dfgc = [v.replace('/', '_frame/').replace('.mp4', '') for v in video_list_dfgc]

        self.video_list = []
        self.img_list = []

        # proceed DFGC data
        image_list_dfgc, _ = build_image_dataset(root_dir, video_list_dfgc, None, sample=sample)
        self.video_list += video_list_dfgc
        self.img_list += image_list_dfgc
            

    def __getitem__(self, index):

        img_path = self.img_list[index]
        img = Image.open(img_path)

        if self.crop_df is not None:
            img_path = img_path.replace('_frame', '').replace(self.root_dir, '')
            crop_idx = list(self.crop_df['image_path']).index(img_path)
            x = list(self.crop_df['x'])[crop_idx]
            y = list(self.crop_df['y'])[crop_idx]
            w = list(self.crop_df['w'])[crop_idx]
            h = list(self.crop_df['h'])[crop_idx]

            if x>1:
                xc, yc = x+w/2, y+h/2
                bbox = (xc-w*self.crop_ratio/2, yc-w*self.crop_ratio/2, xc+w*self.crop_ratio/2, yc+h*self.crop_ratio/2)
            else:
                weight, hight = img.size
                w = w*weight*self.crop_ratio/2
                h = h*hight*self.crop_ratio/2
                xc, yc = x*weight+w/2, y*hight+h/2
                bbox = (xc-w/2, yc-h/2, xc+w/2, yc+h/2)
            
            img = img.crop(bbox)
            print(img.size)            
       
        return (img, img_path)


    def __len__(self):
        return len(self.img_list)


# code test
if __name__=='__main__':

    root_dir = r"/hd3/sunxianyun/DFGC2023_code/"
    label_dir = r'/hd2/sunxianyun/DFGC2023_code/2-achilles10/data/label/train_set.csv'
    crop_dir = r'/hd2/sunxianyun/DFGC2023_re/crop/test_set1_crop_frame.csv'
    extra_data = r'/hd2/sunxianyun/DFGC2023_code/2-achilles10/data/label/train_set_plus_img1.csv'

    test_file = r"/hd3/sunxianyun/DFGC2023_code/1/DFGC-VRA-image/test_set1.txt"

    dataset_generator = DFGCFrameTestDataset(root_dir, test_file, crop_dir, crop_ratio=1.2, sample=[0.25, 0.5, 0.75])
    #dataset_generator = DFGCFrameDataset(root_dir, label_dir, crop_dir, sample=10)
    dataset = ds.GeneratorDataset(dataset_generator, ['img', 'path'], shuffle=True, python_multiprocessing=True)

    transform_list = Compose(build_transform(input_size=(224,224), aug='test'))
    dataset = dataset.map(operations=transform_list, input_columns=['img'])
    dataset = dataset.batch(batch_size=3, drop_remainder=True)

    for data in dataset.create_tuple_iterator():
        print(data[0].shape)
        print(data[1])
        break

    # 打印数据条数
    print("data size:", dataset.get_dataset_size())
