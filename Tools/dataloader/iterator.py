import os
import numpy as np
import csv

import torch
import torch.utils.data as data
from Tools.config import config

if config["dataset"] == "AGP":
    Relation_Class_ID = {'carrying': 0, 'have_it_on_the_back':1, 'leaning_on': 2, 'not_contacting': 3, 'standing_on': 4,
                         'twisting': 5, 'wiping': 6, 'covered_by': 7, 'eating': 8, 'holding': 9, 'lying_on': 10,
                         'sitting_on': 11, 'touching': 12, 'wearing': 13, 'other_relationship': 14}
    Relation_Class_ID = {'carrying': 0, 'leaning_on': 1, 'not_contacting': 2,
                         'standing_on': 3, 'wiping': 4, 'covered_by': 5, 'eating': 6, 'holding': 7, 'lying_on': 8,
                         'sitting_on': 9, 'touching': 10, 'wearing': 11, 'other_relationship': 12}
elif config["dataset"] == "VRP":
    Relation_Class_ID = {'watch': 0, 'bite': 1, 'kiss': 2, 'lick': 3, 'smell': 4, 'caress': 5, 'knock': 6, 'pat': 7,
                         'point_to': 8, 'squeeze': 9, 'hold': 10, 'press': 11, 'touch': 12, 'hit': 13, 'kick': 14,
                         'lift': 15, 'throw': 16, 'wave': 17, 'carry': 18, 'grab': 19, 'release': 20, 'pull': 21,
                         'push': 22, 'hug': 23, 'lean_on': 24, 'ride': 25, 'chase': 26, 'get_on': 27, 'get_off': 28,
                         'hold_hand_of': 29, 'shake_hand_with': 30, 'wave_hand_to': 31, 'speak_to': 32, 'shout_at': 33,
                         'feed': 34, 'open': 35, 'close': 36, 'use': 37, 'cut': 38, 'clean': 39, 'drive': 40,
                         'play(instrument)': 41, 'away': 42, 'towards': 43}
    Relation_Class_ID = {'watch': 0, 'bite': 1, 'kiss': 2, 'lick': 3, 'smell': 4, 'caress': 5, 'pat': 6,
                         'point_to': 7, 'hold': 8, 'press': 9, 'touch': 10, 'hit': 11, 'kick': 12,
                         'lift': 13, 'throw': 14, 'wave': 15, 'grab': 16, 'release': 17, 'pull': 18,
                         'push': 19, 'hug': 20, 'lean_on': 21, 'ride': 22, 'chase': 23, 'get_on': 24, 'get_off': 25,
                         'hold_hand_of': 26, 'shake_hand_with': 27, 'wave_hand_to': 28, 'speak_to': 29,
                         'feed': 30, 'use': 31, 'play(instrument)': 32, 'away': 33, 'towards': 34}


class FeatureIter_clip(data.Dataset):

    def __init__(self,
                 visual_feature_prefix,
                 union_visual_feature_prefix,
                 spatial_feature_prefix,
                 semantic_feature_prefix,
                 txt_list,
                 train_test_list,
                 name="<NO_NAME>"):
        super(FeatureIter_clip, self).__init__()

        self.MaxTime = config["max_frames"]
        self.FeatureChannel = config["visual_dim"]*3 + config["semantic_dim"]*2 + config["spatial_dim"]*1
        self.bbox_result, self.relation_result = self.read_label_csv(txt_list)
        self.video_name = []
        self.object_name = []
        self.subject_name = []
        self.visual_feature_path = []
        self.spatial_feature_path = []
        self.semantic_feature_path = []
        self.union_visual_feature_path = []
        self.frame_names = []
        self.relation_labels = []
        self.object_classes = []

        if name == 'train':
            file_index = 0
        elif name == 'test':
            file_index = 1

        with open(txt_list) as input_file:
            lines = input_file.readlines()
            if train_test_list == []:
                if name == 'train':
                    train_test_list = list(np.zeros(len(lines), dtype=np.int8))
                elif name == 'test':
                    train_test_list = list(np.ones(len(lines), dtype=np.int8))
            #for num,line in enumerate(lines):
            for num in range(len(lines)):
                line = lines[num]
                if train_test_list[num] == file_index:
                    self.video_name.append(line.strip().split(',')[0])
                    self.object_name.append(line.strip().split(',')[1])
                    self.subject_name.append(line.strip().split(',')[2])
                    self.visual_feature_path.append(os.path.join(visual_feature_prefix, line.strip().split(',')[0]))
                    self.union_visual_feature_path.append(os.path.join(union_visual_feature_prefix, line.strip().split(',')[0]))
                    self.spatial_feature_path.append(os.path.join(spatial_feature_prefix, line.strip().split(',')[0]))
                    self.semantic_feature_path.append(semantic_feature_prefix)


    def __getitem__(self, index):
        video_name = self.video_name[index]
        object_name = self.object_name[index]
        subject_name = self.subject_name[index]
        visual_feature_path = self.visual_feature_path[index]
        union_visual_feature_path = self.union_visual_feature_path[index]
        spatial_feature_path = self.spatial_feature_path[index]
        semantic_feature_path = self.semantic_feature_path[index]


        video_feature = np.zeros(shape=[self.MaxTime, self.FeatureChannel], dtype=np.float32)
        label_result = np.zeros(shape=[self.MaxTime, len(Relation_Class_ID)], dtype=np.float32)
        start_time = self.MaxTime - len(self.bbox_result[video_name])
        video_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        label_mask = np.ones(shape=self.MaxTime, dtype=np.int32)


        for img_name in self.bbox_result[video_name]:
            visual_feature_s = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_p.npy'))
            visual_feature_o = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_o.npy'))
            visual_feature_u = np.load(os.path.join(union_visual_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            #visual_feature_p = np.zeros((config["visual_dim"]))
            spatial_feature = np.load(os.path.join(spatial_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            #spatial_feature = np.zeros((config["spatial_dim"]))

            if start_time < self.MaxTime:
                self.frame_names.append(f"{video_name}/{img_name}")
                self.relation_labels.append(self.relation_result[video_name][img_name])
                self.object_classes.append(object_name)

            if config["dataset"] == "AGP":
                semantic_feature_o = np.load(os.path.join(semantic_feature_path, object_name + '.npy'))
                semantic_feature_s = np.zeros((config["semantic_dim"]))
            elif config["dataset"] == "VRP":
                semantic_feature_o = np.load(os.path.join(semantic_feature_path, object_name.split('/')[0] + '.npy'))
                semantic_feature_s = np.load(os.path.join(semantic_feature_path, subject_name.split('/')[0] + '.npy'))

            frame_feature = np.concatenate(
                (visual_feature_s, visual_feature_u, visual_feature_o, spatial_feature, semantic_feature_s, semantic_feature_o))
            video_feature[start_time] = frame_feature
            label_result[start_time][Relation_Class_ID[self.relation_result[video_name][img_name]]] = 1
            start_time += 1

        #return video_feature, video_mask, label_result, label_mask
        return (
            video_feature,
            video_mask,
            label_result,
            label_mask,
            self.video_name[index],  # video_id
            self.object_name[index],  # object
            self.subject_name[index],  # subject
            list(self.bbox_result[self.video_name[index]].keys()),  # frame_id
            list(self.relation_result[self.video_name[index]].values())  # relationship label
        )

    def __len__(self):
        return len(self.video_name)

    def read_label_csv(self, csv_path):
        bbox_result = {}
        relation_result = {}
        with open(csv_path, 'r') as op:
            csv_data = csv.reader(op, delimiter=',')
            for row in csv_data:
                bbox_result[row[0]] = {}
                relation_result[row[0]] = {}
                for frame in row[3:]:
                    frame_row = frame.split(' ')
                    bbox = frame_row[1:9]
                    bbox_result[row[0]][frame_row[0]] = bbox
                    relation = frame_row[-1]
                    relation_result[row[0]][frame_row[0]] = relation
        op.close()
        return bbox_result, relation_result


class FeatureIter(data.Dataset):

    def __init__(self,
                 visual_feature_prefix,
                 pair_visual_feature_prefix,
                 spatial_feature_prefix,
                 semantic_feature_prefix,
                 txt_list,
                 train_test_list,
                 name="<NO_NAME>"):
        super(FeatureIter, self).__init__()

        self.MaxTime = config["max_frames"]
        self.FeatureChannel = config["visual_dim"]*3 + 300*2 + config["spatial_dim"]*1
        self.bbox_result, self.relation_result = self.read_label_csv(txt_list)
        self.video_name = []
        self.object_name = []
        self.subject_name = []
        self.visual_feature_path = []
        self.spatial_feature_path = []
        self.semantic_feature_path = []
        self.pair_visual_feature_path = []

        if name == 'train':
            file_index = 0
        elif name == 'test':
            file_index = 1

        with open(txt_list) as input_file:
            lines = input_file.readlines()
            #if train_test_list == []:
            if len(train_test_list) == 0:
                if name == 'train':
                    train_test_list = list(np.zeros(len(lines), dtype=np.int8))
                elif name == 'test':
                    train_test_list = list(np.ones(len(lines), dtype=np.int8))
            #for num,line in enumerate(lines):
            for num in range(len(lines)):
                line = lines[num]
                if train_test_list[num] == file_index:
                    self.video_name.append(line.strip().split(',')[0])
                    self.object_name.append(line.strip().split(',')[1])
                    self.subject_name.append(line.strip().split(',')[2])
                    self.visual_feature_path.append(os.path.join(visual_feature_prefix, line.strip().split(',')[0]))
                    self.pair_visual_feature_path.append(os.path.join(pair_visual_feature_prefix, line.strip().split(',')[0]))
                    self.spatial_feature_path.append(os.path.join(spatial_feature_prefix, line.strip().split(',')[0]))
                    self.semantic_feature_path.append(semantic_feature_prefix)


    def __getitem__(self, index):
        video_name = self.video_name[index]
        object_name = self.object_name[index]
        subject_name = self.subject_name[index]
        visual_feature_path = self.visual_feature_path[index]
        pair_visual_feature_path = self.pair_visual_feature_path[index]
        spatial_feature_path = self.spatial_feature_path[index]
        semantic_feature_path = self.semantic_feature_path[index]


        video_feature = np.zeros(shape=[self.MaxTime, self.FeatureChannel], dtype=np.float32)
        label_result = np.zeros(shape=[self.MaxTime, len(Relation_Class_ID)], dtype=np.float32)
        start_time = self.MaxTime - len(self.bbox_result[video_name])
        video_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        label_mask = np.ones(shape=self.MaxTime, dtype=np.int32)


        for img_name in self.bbox_result[video_name]:
            visual_feature_s = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_p.npy'))
            visual_feature_o = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_o.npy'))
            visual_feature_p = np.load(os.path.join(pair_visual_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            #visual_feature_p = np.zeros((config["visual_dim"]))
            spatial_feature = np.load(os.path.join(spatial_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            #spatial_feature = np.zeros((config["spatial_dim"]))

            if config["dataset"] == "AGP":
                semantic_feature_o = np.load(os.path.join(semantic_feature_path, object_name + '.npy'))
                semantic_feature_s = np.zeros((config["semantic_dim"]))
            elif config["dataset"] == "VRP":
                semantic_feature_o = np.load(os.path.join(semantic_feature_path, object_name.split('/')[0] + '.npy'))
                semantic_feature_s = np.load(os.path.join(semantic_feature_path, subject_name.split('/')[0] + '.npy'))

            frame_feature = np.concatenate(
                (visual_feature_s, visual_feature_p, visual_feature_o, spatial_feature, semantic_feature_s, semantic_feature_o))
            video_feature[start_time] = frame_feature
            label_result[start_time][Relation_Class_ID[self.relation_result[video_name][img_name]]] = 1
            start_time += 1

        return video_feature, video_mask, label_result, label_mask

    def __len__(self):
        return len(self.video_name)

    def read_label_csv(self, csv_path):
        bbox_result = {}
        relation_result = {}
        with open(csv_path, 'r') as op:
            csv_data = csv.reader(op, delimiter=',')
            for row in csv_data:
                bbox_result[row[0]] = {}
                relation_result[row[0]] = {}
                for frame in row[3:]:
                    frame_row = frame.split(' ')
                    bbox = frame_row[1:9]
                    bbox_result[row[0]][frame_row[0]] = bbox
                    relation = frame_row[-1]
                    relation_result[row[0]][frame_row[0]] = relation
        op.close()
        return bbox_result, relation_result
