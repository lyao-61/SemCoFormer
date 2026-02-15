import os
import numpy as np
import csv

import torch.utils.data as data

Relation_Class_ID = {'watch': 0, 'bite': 1, 'kiss': 2, 'lick': 3, 'smell': 4, 'caress':5 , 'knock': 6, 'pat': 7,
    'point_to': 8, 'squeeze': 9, 'hold': 10, 'press': 11, 'touch': 12, 'hit': 13, 'kick': 14,
    'lift': 15, 'throw': 16, 'wave': 17, 'carry': 18, 'grab': 19, 'release': 20, 'pull': 21,
    'push': 22, 'hug': 23, 'lean_on': 24, 'ride': 25, 'chase': 26, 'get_on': 27, 'get_off': 28,
    'hold_hand_of': 29, 'shake_hand_with': 30, 'wave_hand_to': 31, 'speak_to': 32, 'shout_at': 33, 'feed': 34,
    'open': 35, 'close': 36, 'use': 37, 'cut': 38, 'clean': 39, 'drive': 40, 'play(instrument)': 41, 'away': 42, 'towards': 43}


class FeatureIter_VRP(data.Dataset):

    def __init__(self,
                 visual_feature_prefix,
                 pair_visual_feature_prefix,
                 spatial_feature_prefix,
                 semantic_feature_prefix,
                 txt_list,
                 train_test_list,
                 name="<NO_NAME>"):
        super(FeatureIter_VRP, self).__init__()

        self.MaxTime = 10
        self.FeatureChannel = 1536 + 1536 + 300 + 300 + 20
        self.bbox_result, self.relation_result = self.read_label_csv(txt_list)
        self.video_name = []
        self.object_name = []
        self.subject_name = []
        self.visual_feature_path = []
        self.spatial_feature_path = []
        self.semantic_feature_path = []

        if name == 'train':
            file_index = 0
        elif name == 'test':
            file_index = 1

        with open(txt_list) as input_file:
            lines = input_file.readlines()
            for num,line in enumerate(lines):
                if train_test_list[num] == file_index:
                    self.video_name.append(line.strip().split(',')[0])
                    self.object_name.append(line.strip().split(',')[1])
                    self.subject_name.append(line.strip().split(',')[2])
                    self.visual_feature_path.append(os.path.join(visual_feature_prefix, line.strip().split(',')[0]))
                    self.spatial_feature_path.append(os.path.join(spatial_feature_prefix, line.strip().split(',')[0]))
                    self.semantic_feature_path.append(semantic_feature_prefix)


    def __getitem__(self, index):
        video_name = self.video_name[index]
        object_name = self.object_name[index]
        subject_name = self.subject_name[index]
        visual_feature_path = self.visual_feature_path[index]
        spatial_feature_path = self.spatial_feature_path[index]
        semantic_feature_path = self.semantic_feature_path[index]


        video_feature = np.zeros(shape=[self.MaxTime, self.FeatureChannel], dtype=np.float32)
        label_result = np.zeros(shape=[self.MaxTime, len(Relation_Class_ID)], dtype=np.float32)
        start_time = self.MaxTime - len(self.bbox_result[video_name])
        video_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        label_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        video_mask[-1:] = 0
        label_mask[-1:] = 0


        for img_name in self.bbox_result[video_name]:
            visual_feature_p = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_p.npy'))
            visual_feature_o = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_o.npy'))
            spatial_feature = np.load(os.path.join(spatial_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            semantic_feature_o = np.load(os.path.join(semantic_feature_path, object_name.split('/')[0]+ '.npy'))
            semantic_feature_p = np.load(os.path.join(semantic_feature_path, subject_name.split('/')[0]+ '.npy'))
            frame_feature = np.concatenate((visual_feature_p, visual_feature_o, semantic_feature_p, semantic_feature_o, spatial_feature))

            video_feature[start_time] = frame_feature
            label_result[start_time][Relation_Class_ID[self.relation_result[video_name][img_name][0]]] = 1
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
                    relation = frame_row[9:12]
                    relation_result[row[0]][frame_row[0]] = relation
        op.close()
        return bbox_result, relation_result


class FeatureIter_VRP_GCN(data.Dataset):

    def __init__(self,
                 visual_feature_prefix,
                 spatial_feature_prefix,
                 semantic_feature_prefix,
                 txt_list,
                 train_test_list,
                 name="<NO_NAME>"):
        super(FeatureIter_VRP_GCN, self).__init__()

        self.MaxTime = 10
        self.FeatureChannel = 1536 + 1536 + 300 + 300 + 20
        self.bbox_result, self.relation_result = self.read_label_csv(txt_list)
        self.video_name = []
        self.object_name = []
        self.subject_name = []
        self.visual_feature_path = []
        self.spatial_feature_path = []
        self.semantic_feature_path = []

        if name == 'train':
            file_index = 0
        elif name == 'test':
            file_index = 1

        with open(txt_list) as input_file:
            lines = input_file.readlines()
            for num,line in enumerate(lines):
                if train_test_list[num] == file_index:
                    self.video_name.append(line.strip().split(',')[0])
                    self.object_name.append(line.strip().split(',')[1])
                    self.subject_name.append(line.strip().split(',')[2])
                    self.visual_feature_path.append(os.path.join(visual_feature_prefix, line.strip().split(',')[0]))
                    self.spatial_feature_path.append(os.path.join(spatial_feature_prefix, line.strip().split(',')[0]))
                    self.semantic_feature_path.append(semantic_feature_prefix)


    def __getitem__(self, index):
        video_name = self.video_name[index]
        object_name = self.object_name[index]
        subject_name = self.subject_name[index]
        visual_feature_path = self.visual_feature_path[index]
        spatial_feature_path = self.spatial_feature_path[index]
        semantic_feature_path = self.semantic_feature_path[index]


        video_feature = np.zeros(shape=[self.MaxTime, self.FeatureChannel], dtype=np.float32)
        label_result = np.zeros(shape=[self.MaxTime, len(Relation_Class_ID)], dtype=np.float32)
        start_time = self.MaxTime - len(self.bbox_result[video_name])
        video_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        label_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        video_mask[-1:] = 0
        label_mask[-1:] = 0


        for img_name in self.bbox_result[video_name]:
            visual_feature_p = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_p.npy'))
            visual_feature_o = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_o.npy'))
            spatial_feature = np.load(os.path.join(spatial_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            semantic_feature_o = np.load(os.path.join(semantic_feature_path, object_name.split('/')[0]+ '.npy'))
            semantic_feature_p = np.load(os.path.join(semantic_feature_path, subject_name.split('/')[0]+ '.npy'))
            frame_feature = np.concatenate((visual_feature_p, visual_feature_o, semantic_feature_p, semantic_feature_o, spatial_feature))

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
                    relation = frame_row[9:12]
                    relation_result[row[0]][frame_row[0]] = relation
        op.close()
        return bbox_result, relation_result


class FeatureIter_VRP_GCN_Transformer(data.Dataset):

    def __init__(self,
                 visual_feature_prefix,
                 spatial_feature_prefix,
                 semantic_feature_prefix,
                 txt_list,
                 train_test_list,
                 name="<NO_NAME>"):
        super(FeatureIter_VRP_GCN_Transformer, self).__init__()

        self.MaxTime = 10
        self.FeatureChannel = 1536 + 1536 + 300 + 300 + 20
        self.bbox_result, self.relation_result = self.read_label_csv(txt_list)
        self.video_name = []
        self.object_name = []
        self.subject_name = []
        self.visual_feature_path = []
        self.spatial_feature_path = []
        self.semantic_feature_path = []

        if name == 'train':
            file_index = 0
        elif name == 'test':
            file_index = 1

        with open(txt_list) as input_file:
            lines = input_file.readlines()
            for num,line in enumerate(lines):
                if train_test_list[num] == file_index:
                    self.video_name.append(line.strip().split(',')[0])
                    self.object_name.append(line.strip().split(',')[1])
                    self.subject_name.append(line.strip().split(',')[2])
                    self.visual_feature_path.append(os.path.join(visual_feature_prefix, line.strip().split(',')[0]))
                    self.spatial_feature_path.append(os.path.join(spatial_feature_prefix, line.strip().split(',')[0]))
                    self.semantic_feature_path.append(semantic_feature_prefix)


    def __getitem__(self, index):
        video_name = self.video_name[index]
        object_name = self.object_name[index]
        subject_name = self.subject_name[index]
        visual_feature_path = self.visual_feature_path[index]
        spatial_feature_path = self.spatial_feature_path[index]
        semantic_feature_path = self.semantic_feature_path[index]


        video_feature = np.zeros(shape=[self.MaxTime, self.FeatureChannel], dtype=np.float32)
        label_result = np.zeros(shape=[self.MaxTime, len(Relation_Class_ID)], dtype=np.float32)
        start_time = self.MaxTime - len(self.bbox_result[video_name])
        video_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        label_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        video_mask[-1:] = 0
        label_mask[-1:] = 0


        for img_name in self.bbox_result[video_name]:
            visual_feature_p = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_p.npy'))
            visual_feature_o = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_o.npy'))
            spatial_feature = np.load(os.path.join(spatial_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            semantic_feature_o = np.load(os.path.join(semantic_feature_path, object_name.split('/')[0]+ '.npy'))
            semantic_feature_p = np.load(os.path.join(semantic_feature_path, subject_name.split('/')[0]+ '.npy'))
            frame_feature = np.concatenate((visual_feature_p, visual_feature_o, semantic_feature_p, semantic_feature_o, spatial_feature))

            video_feature[start_time] = frame_feature
            label_result[start_time][Relation_Class_ID[self.relation_result[video_name][img_name][0]]] = 1
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
