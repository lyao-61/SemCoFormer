import os
import numpy as np
import csv

import torch
import torch.utils.data as data
from gensim.models import KeyedVectors

Relation_Class_ID = {'carrying':0, 'have_it_on_the_back':1, 'leaning_on':2, 'not_contacting':3, 'standing_on':4, 'twisting':5, 'wiping':6,
                             'covered_by':7, 'eating':8, 'holding':9, 'lying_on':10, 'sitting_on':11, 'touching':12, 'wearing':13, 'other_relationship':14}
Object_ID = {'person':0, 'bag':1,'bed':2,'blanket':3,'book':4,'box':5,'broom':6,'chair':7,'closetcabinet':8,'clothes':9,'cupglassbottle':10,'dish':11,'door':12,
             'doorknob':13,'doorway':14,'floor':15,'food':16,'groceries':17,'laptop':18,'light':19,'medicine':20,'mirror':21,'papernotebook':22,'phonecamera':23,
             'picture':24,'pillow':25,'refrigerator':26,'sandwich':27,'shelf':28,'shoe':29,'sofacouch':30,'table':31,'television':32,'towel':33,'vacuum':34,'window':35}

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

        self.MaxTime = 10
        self.FeatureChannel = 1536
        #self.FeatureChannel = 1536 + 1536 + 20
        self.FeatureChannel1 = 128
        self.bbox_result, self.relation_result = self.read_label_csv(txt_list)
        #self.word2vec = KeyedVectors.load_word2vec_format('/home/ouyangjun/workspace/VRP/Tools/dataloader/GoogleNews-vectors-negative300.bin', binary=True)
        self.embedding1 = torch.nn.Embedding(16,128)
        self.video_name = []
        self.object_name = []
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
            for num,line in enumerate(lines):
                if train_test_list[num] == file_index:
                    self.video_name.append(line.strip().split(',')[0])
                    self.object_name.append(line.strip().split(',')[1])
                    self.visual_feature_path.append(os.path.join(visual_feature_prefix, line.strip().split(',')[0]))
                    self.pair_visual_feature_path.append(os.path.join(pair_visual_feature_prefix, line.strip().split(',')[0]))
                    self.spatial_feature_path.append(os.path.join(spatial_feature_prefix, line.strip().split(',')[0]))
                    self.semantic_feature_path.append(semantic_feature_prefix)


    def __getitem__(self, index):
        video_name = self.video_name[index]
        object_name = self.object_name[index]
        visual_feature_path = self.visual_feature_path[index]
        pair_visual_feature_path = self.pair_visual_feature_path[index]
        spatial_feature_path = self.spatial_feature_path[index]
        semantic_feature_path = self.semantic_feature_path[index]

        #object_idx = torch.LongTensor([Object_ID[object_name]])
        #object_idx = Variable(object_idx)
        #object_embed = self.word2vec[object_name]
        #object_embed = object_embed.data.numpy()


        video_feature = np.zeros(shape=[self.MaxTime, 3, self.FeatureChannel], dtype=np.float32)
        label_result = np.zeros(shape=[self.MaxTime, len(Relation_Class_ID)], dtype=np.float32)
        start_time = self.MaxTime - len(self.bbox_result[video_name])
        video_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        label_mask = np.ones(shape=self.MaxTime, dtype=np.int32)
        video_mask[-2:] = 0
        label_mask[-2:] = 0


        for img_name in self.bbox_result[video_name]:
            visual_feature_p = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_p.npy'))
            visual_feature_o = np.load(os.path.join(visual_feature_path, os.path.splitext(img_name)[0]+'_o.npy'))
            visual_feature_r = np.load(os.path.join(pair_visual_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            #spatial_feature = np.load(os.path.join(spatial_feature_path, os.path.splitext(img_name)[0]+'.npy'))
            #semantic_feature = np.load(os.path.join(semantic_feature_path, object_name+ '.npy'))
            #frame_feature = np.concatenate((visual_feature_p, visual_feature_o, semantic_feature, spatial_feature))
            frame_feature = np.stack((visual_feature_p, visual_feature_r, visual_feature_o), axis=0)

            video_feature[start_time] = frame_feature
            label_result[start_time][Relation_Class_ID[self.relation_result[video_name][img_name][2]]] = 1
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