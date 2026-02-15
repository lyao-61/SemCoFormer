config = {
    # training
    #"cuda": 'cuda:3',
    #"learning_rate": 5e-5,
    "display_batch_interval": 50,
    "max_epoches": 50,
    "batch_size": 1,
    #"early_stopping": 5,
    #"lr_decay_n_iters": 3000,

    "dropout_prob": 0.9,
    #"dropout_rate": 0.0,

    "dataset":"VRP", #VRP
    "data_root": "/home/ouyangjun/workspace/data/a/liuyao/Data/VRP_gct/", #
    # "train_file": "3/AGP_annotation_free_new_new_30_train2.txt", #
    # "test_file": "3/AGP_annotation_free_new_new_30_test2.txt", #
    # "test_file_zero_shot": "3/AGP_annotation_free_new_new_30_zero_shot_test2.txt",
    "train_file": "VRP_annotation_10_train.txt",
    "test_file":"VRP_annotation_10_test.txt",
    "train_num": 1748, #V 1748 #A 12797
    "test_num": 175, #V 175 #A 650
    "train_val_ratio": 0.9,
    "max_frames": 30,
    "max_target": 30,
    "num_class": 35, # V 35 A 13

    # clip feature
    # "input_video_dim":512,
    # "feat_dims":256,
    # "visual_dim":512,
    # "semantic_dim":512,
    # "spatial_dim":20,

    # gct feature
    "input_video_dim": 1536,
    "feat_dims": 256,
    "visual_dim": 1536,
    "semantic_dim": 600,  # V 600 A 300
    "spatial_dim": 20,

    # model
    "cuda": 'cuda:3',
    "learning_rate": 5e-5,
    "mode": "transformer", #gcn_transformer
    # gcn
    "dropout_rate": 0.0,    #gcn
    "d_model":512,
    "d_ff": 2048,  #positionwisefeadforward
    "num_heads":128,
    "num_nodes": 3,
    "pred_step": 1,
    "MAdropout":0.1,  #Muliheadattention
    "PFdropout":0.1,  #positionwisefeadforward
    #V2T T2V
    "VLdmodel":512,
    "VLnhead":256,
    "VLdrop":0.8,
    #Sparse_Self_Attention
    "SAdropout": 0.1,
    "use_layer_scale": True,
    "layer_scale_init_value": 1e-6,
    "sparse_size":4,
    "drop_path":0.1,
    #QuadrangleAttention
    "QAdropout":0.1,
    "window_size":29,
    "rpe":'v2',
    "coords":20,
    #sttran
    "temporal_layer":1,
    "spatial_layer":1,

    "visualize_result_path":"/home/ouyangjun/workspace/data/a/lyao/SemCo/Models/VRP_10/t_v2t_t2v_qa_sa_4.txt",
    "model_save_best_loss": "/home/ouyangjun/workspace/data/a/lyao/SemCo/Models/VRP_10/t_v2t_t2v_qa_sa_4/model_bset_loss.pth",
    "model_save_best_acc": "/home/ouyangjun/workspace/data/a/lyao/SemCo/Models/VRP_10/t_v2t_t2v_qa_sa_4/model_bset_acc.pth",
    "model_save_path": "/home/ouyangjun/workspace/data/a/lyao/SemCo/Models/VRP_10/t_v2t_t2v_qa_sa_4/",
    "model_load_name": None,
    "acc_file": "/home/ouyangjun/workspace/data/a/lyao/SemCo/Results/VRP_10/t_v2t_t2v_qa_sa_4.txt",
    "acc_test_file": "/home/ouyangjun/workspace/data/a/lyao/SemCo/Results/VRP_10/test/t_v2t_t2v_qa_sa_4.txt",
}
