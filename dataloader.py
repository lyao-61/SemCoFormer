import torch
from .iterator import *

from Tools.config import config


def get_clip_dataset(file_num, ratio, data_root):
    file_list = get_train_test_list(file_num, ratio)
    file_list_test = []
    train = FeatureIter_clip(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                                            union_visual_feature_prefix=os.path.join(data_root, 'union_features_resnet'),
                                            spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                                            semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic_clip'),
                                            txt_list=os.path.join(data_root, config["train_file"]),
                                            train_test_list=file_list,
                                            name='train',
                                            )

    val = FeatureIter_clip(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                                          union_visual_feature_prefix=os.path.join(data_root, 'union_features_resnet'),
                                          spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                                          semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic_clip'),
                                          txt_list=os.path.join(data_root, config["train_file"]),
                                          train_test_list=file_list,
                                          name='test',
                                          )

    test = FeatureIter_clip(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                                          union_visual_feature_prefix=os.path.join(data_root, 'union_features_resnet'),
                                          spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                                          semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic_clip'),
                                          txt_list=os.path.join(data_root, config["test_file"]),
                                          train_test_list=file_list_test,
                                          name='test',
                                          )

    return (train, val, test)


def get_dataset(file_num, ratio, data_root):
    file_list = get_train_test_list(file_num, ratio)
    file_list_test = []
    train = FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                                            pair_visual_feature_prefix=os.path.join(data_root, 'pair_features_resnet'),
                                            spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                                            semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
                                            txt_list=os.path.join(data_root, config["train_file"]),
                                            train_test_list=file_list,
                                            name='train',
                                            )

    val = FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                                          pair_visual_feature_prefix=os.path.join(data_root, 'pair_features_resnet'),
                                          spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                                          semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
                                          txt_list=os.path.join(data_root, config["train_file"]),
                                          train_test_list=file_list,
                                          name='test',
                                          )

    test = FeatureIter(visual_feature_prefix=os.path.join(data_root, 'object_features_resnet'),
                                          pair_visual_feature_prefix=os.path.join(data_root, 'pair_features_resnet'),
                                          spatial_feature_prefix=os.path.join(data_root, 'object_features_spatial'),
                                          semantic_feature_prefix=os.path.join(data_root, 'object_features_semantic'),
                                          txt_list=os.path.join(data_root, config["test_file"]),
                                          train_test_list=file_list_test,
                                          name='test',
                                          )

    return (train, val, test)


def get_train_test_list(file_num, ratio):
    train_num = int(file_num * ratio)
    test_num = file_num - train_num
    train_list = np.zeros([train_num], dtype=int)
    test_list = np.ones([test_num], dtype=int)
    file_list = np.concatenate((train_list, test_list))
    np.random.shuffle(file_list)
    #file_list = train_list

    return file_list


def creat(num_workers=1):

    #train, val, test = get_dataset(config["train_num"], config["train_val_ratio"], config["data_root"])
    if config["mode"]=="clip_sttran":
        train, val, test = get_clip_dataset(config["train_num"], config["train_val_ratio"], config["data_root"])
    else:
        train, val, test = get_dataset(config["train_num"], config["train_val_ratio"], config["data_root"])

    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size=config["batch_size"], shuffle=True,
                                               num_workers=num_workers, pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val,
                                             batch_size=config["batch_size"], shuffle=True,
                                             num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test,
                                             batch_size=config["batch_size"], shuffle=False,
                                             num_workers=num_workers, pin_memory=True)

    return (train_loader, val_loader, test_loader)
