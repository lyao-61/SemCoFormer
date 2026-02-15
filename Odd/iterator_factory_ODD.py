import os
import numpy as np

import torch

from .video_iterator import FeatureIter

def get_ARP(file_num, ratio,
            data_root='/home/ouyangjun/workspace/VRP/Data/AG_prediction/',

                 **kwargs):
    """ feature iter for action genome prediction
    """
    file_list = get_train_test_list(file_num, ratio)
    train = FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                        spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                        semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
                        txt_list=os.path.join(data_root, 'AG_relation_prediction_重复筛选后.txt'),
                        train_test_list = file_list,
                        name='train',
                        )


    val   = FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                        spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                        semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
                        txt_list=os.path.join(data_root, 'AG_relation_prediction_重复筛选后.txt'),
                        train_test_list = file_list,
                        name='test',
                        )
    return (train, val)

def get_train_test_list(file_num, ratio):
    train_num = int(file_num * ratio)
    test_num = file_num - train_num
    train_list = np.zeros([train_num], dtype=int)
    test_list = np.ones([test_num], dtype=int)
    file_list = np.concatenate((train_list, test_list))
    np.random.shuffle(file_list)

    return file_list


def creat(file_num, ratio=0.9, batch_size=1, num_workers=1, **kwargs):

    train, val = get_ARP(file_num, ratio, **kwargs)

    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    return (train_loader, val_loader)