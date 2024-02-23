import numpy as np
import mindspore.dataset.transforms.c_transforms as transforms
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as pvision
from mindspore.dataset import RandomSampler,SequentialSampler,GeneratorDataset
from dataset.concat_dataset import ConcatDataset
from dataset.cifar import get_cifar10_dataset, get_cifar100_dataset, get_cifar10_c_dataset, get_cifar100_c_dataset
from dataset.imagenet import get_imagenet_30_datasets, get_imagenet_30_c_datasets
import math
from scipy.stats import poisson

########################
###  Transformation  ###
########################

def get_transform(size, mean, std, algorithm=None):

    train_transform = transforms.Compose([
        #pvision.Decode(),
        vision.Resize(size),
        vision.RandomHorizontalFlip(),
        #vision.RandomCrop(size=size,padding=int(size*0.125),padding_mode=vision.Border.REFLECT),
        pvision.ToTensor(),
        pvision.Normalize(mean=mean,std=std),
    ])

    test_transform = transforms.Compose([
        #pvision.Decode(),
        vision.Resize(size),
        pvision.ToTensor(),
        pvision.Normalize(mean=mean,std=std),
    ])
    
    return train_transform, test_transform


################
### SETTINGS ###
################

dataset_settings = {
    'cifar10':{
        'clean': get_cifar10_dataset,
        'robust': get_cifar10_c_dataset,
        'size': [32, 32],
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2471, 0.2435, 0.2616]
    },
    'cifar100':{
        'clean': get_cifar100_dataset,
        'robust': get_cifar100_c_dataset,
        'size': [32, 32],
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    },
    'imagenet30':{
        'clean': get_imagenet_30_datasets,
        'robust': get_imagenet_30_c_datasets,
        'size': [64, 64],
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
}


########################
### DATA PREPARATION ###
########################

def dataset_multi_split(args, labels):
    # according to the given labels and numbers, split dataset into \
    # 5 class-balanced subsets which follows the photon-limited imaging problem.
    num_per_class = args.num_per_class
    severity_level = np.arange(0, args.num_severity+1)
    probs = poisson.pmf(severity_level, 1)
    num_per_distribution = []
    idx_list = []
    for prob in probs:
        num_per_distribution.append(math.floor(num_per_class * prob))
        idx_list.append([])
    num_slices = [0]
    num_slices = num_slices + list(np.cumsum(num_per_distribution))
    labels = np.array(labels)
    val_idx = []
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, num_per_class, False)
        for j in range(args.num_severity+1):
            idx_list[j].extend(idx[num_slices[j]:num_slices[j+1]])

    val_idx = np.array(range(len(labels)))
    for i in range(args.num_severity+1):
        val_idx = [idx for idx in val_idx if idx not in idx_list[i]]
    return idx_list, val_idx


def get_cifar_prepared_dataset(args):
    # use target for splitting
    train_clean_dataset = dataset_settings[args.dataset]['clean'](
        args,
        data_dir=args.data_dir,
        is_train=True,
        transform=None)

    # Split label and unlabel idx
    idx_list, val_idx = dataset_multi_split(args, train_clean_dataset.targets)

    # constructing training dataset
    train_clean_dataset = dataset_settings[args.dataset]['clean'](
        args,
        indexs=idx_list[0],
        data_dir=args.data_dir,
        is_train=True,
        transform=None)

    train_dataset_list = [train_clean_dataset]

    for i in range(1, args.num_severity+1):
        train_dataset_list.append(dataset_settings[args.dataset]['robust'](
        args,
        indexs=idx_list[i],
        data_dir=args.data_dir,
        is_train=True,
        transform=None,
        severity=str(i)))

    train_dataset = ConcatDataset(train_dataset_list)


    # constructing validation dataset
    val_clean_dataset = ConcatDataset([dataset_settings[args.dataset]['clean'](
        args,
        indexs=val_idx,
        data_dir=args.data_dir,
        is_train=True,
        transform=None)])

    val_dataset_list = [val_clean_dataset]

    for i in range(1, args.num_severity+1):
        val_dataset_list.append(
            ConcatDataset(
                [dataset_settings[args.dataset]['robust'](
                    args,
                    indexs=val_idx,
                    data_dir=args.data_dir,
                    is_train=True,
                    transform=None,
                    severity=str(i))]
        )
        )
    
    val_dataset = tuple(val_dataset_list)
    
    # constructing test dataset
    test_clean_dataset = ConcatDataset([dataset_settings[args.dataset]['clean'](
        args,
        data_dir=args.data_dir,
        is_train=False,
        transform=None)])

    test_dataset_list = [test_clean_dataset]

    for i in range(1, args.num_severity+1):
        test_dataset_list.append(ConcatDataset([dataset_settings[args.dataset]['robust'](
        args,
        data_dir=args.data_dir,
        is_train=False,
        transform=None,
        severity=str(i))]))
    
    test_dataset = tuple(test_dataset_list)
    
    return train_dataset, val_dataset, test_dataset

def get_imagenet30_prepared_dataset(args):

    train_cleanfile = './imagenet30_filelist/imagenet30_distribution_0.txt'
    train_robustfile_list = []
    for i in range(args.num_severity+1):
        train_robustfile_list.append('./imagenet30_filelist/imagenet30_distribution_' + str(i) + '.txt')
    val_file = './imagenet30_filelist/imagenet30_val.txt'
    test_file = './imagenet30_filelist/imagenet30_test.txt'
    

    # constructing training dataset
    train_dataset_list = []
    train_dataset_list.append(dataset_settings[args.dataset]['clean'](
    args,
    train_cleanfile,
    is_train=True,
    transform=None))

    for i in range(1, args.num_severity+1):
        train_dataset_list.append(dataset_settings[args.dataset]['robust'](
        args,
        train_robustfile_list[i],
        is_train=True,
        transform=None,
        severity=str(i)))

    train_dataset = ConcatDataset(train_dataset_list)


    # constructing validation dataset
    val_clean_dataset = ConcatDataset([dataset_settings[args.dataset]['clean'](
        args,
        val_file,
        is_train=True,
        transform=None)])

    val_dataset_list = [val_clean_dataset]

    for i in range(1, args.num_severity+1):
        val_dataset_list.append(
            ConcatDataset(
                [dataset_settings[args.dataset]['robust'](
                    args,
                    val_file,
                    is_train=True,
                    transform=None,
                    severity=str(i))]
        )
        )
    
    val_dataset = tuple(val_dataset_list)
    
    # constructing test dataset
    test_clean_dataset = ConcatDataset([dataset_settings[args.dataset]['clean'](
        args,
        test_file,
        is_train=False,
        transform=None)])

    test_dataset_list = [test_clean_dataset]

    for i in range(1, args.num_severity+1):
        test_dataset_list.append(ConcatDataset([dataset_settings[args.dataset]['robust'](
        args,
        test_file,
        is_train=False,
        transform=None,
        severity=str(i))]))
    
    test_dataset = tuple(test_dataset_list)
    
    return train_dataset, val_dataset, test_dataset


def get_dataloader(args, train_dataset, val_dataset, test_dataset):

    train_transform, test_transform = get_transform(dataset_settings[args.dataset]['size'], \
        dataset_settings[args.dataset]['mean'], dataset_settings[args.dataset]['std'], args.algorithm)

    labeled_sampler = RandomSampler(replacement=True)
    val_sampler = [SequentialSampler()]
    for i in range(1, args.num_severity+1):
        val_sampler.append(SequentialSampler())
    test_sampler = [SequentialSampler()]
    for i in range(1, args.num_severity+1):
        test_sampler.append(SequentialSampler())

    train_loader = GeneratorDataset(source=train_dataset,
                                    column_names=["image", "label","robust_idx"],
                                    sampler=labeled_sampler,
                                    num_parallel_workers=args.num_workers)\
        .map(operations=train_transform, input_columns="image")\

    train_loader = train_loader.batch(batch_size=args.batch_size,drop_remainder=True)

    val_loader = GeneratorDataset(source=val_dataset[0],
                                  column_names=["image", "label","robust_idx"],
                                  sampler = val_sampler[0],
                                  num_parallel_workers = args.num_workers)\
        .map(operations=test_transform, input_columns="image")\
        .batch(batch_size=args.eval_batch_size,drop_remainder=False)
    val_loader_tuple = [val_loader]
    for i in range(1, args.num_severity+1):
        val_loader_tuple.append(GeneratorDataset(source=val_dataset[i],
                                      column_names=["image", "label","robust_idx"],
                                      sampler=val_sampler[i],
                                      num_parallel_workers=args.num_workers)
                                .map(operations=test_transform, input_columns="image")
                                .batch(batch_size=args.eval_batch_size, drop_remainder=False))
        
    test_loader = GeneratorDataset(source=test_dataset[0],
                                    column_names=["image", "label","robust_idx"],
                                    sampler=test_sampler[0],
                                    num_parallel_workers=args.num_workers)\
        .map(operations=test_transform, input_columns="image")\
        .batch(batch_size=args.eval_batch_size,drop_remainder=False)
    test_loader_tuple = [test_loader]
    for i in range(1, args.num_severity+1):
        test_loader_tuple.append(GeneratorDataset(source=test_dataset[i],
                                    column_names=["image", "label","robust_idx"],
                                    sampler=test_sampler[i],
                                    num_parallel_workers=args.num_workers)
                                 .map(operations=test_transform, input_columns="image")
                                 .batch(batch_size=args.eval_batch_size,drop_remainder=False))
    
    return train_loader, tuple(val_loader_tuple), tuple(test_loader_tuple)
