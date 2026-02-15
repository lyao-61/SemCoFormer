config = {
    "learning_rate": 5e-5,
    "lr_decay_n_iters": 3000,
    "max_epoches": 1000,
    "early_stopping": 5,
    "display_batch_interval": 1000,
    "dropout_prob": 0.9,
    "dropout_rate": 0.0,

    "batch_size": 12,
    "d_model": 256,
    "d_ff": 256,
    "num_heads": 8,

    "dataset": "AGP", # VRP
    "mode": "transformer",
    "video_num": 9656, # 928
    "ratio": 0.8,
    "max_frames": 10,
    "max_words": 10,
    "num_class": 15,
    "num_nodes": 3,
    'feat_dims': 1536,
    "input_video_dim": 3392,  # 3692

    "model_save_best": "/home/ouyangjun/workspace/VRP/Models/0.8_pre_trans/model_bset.pth",
    "model_save_last": "/home/ouyangjun/workspace/VRP/Models/0.8_pre_trans/model_last.pth",
    "model_load_name": None,
    "acc_file": "/home/ouyangjun/workspace/VRP/Results/0.8_transformer.txt"

}
