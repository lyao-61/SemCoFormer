from tqdm import tqdm

from Tools.ARconv import ARConv
from Tools.QuadrangleAttention import QuadrangleAttention
from Tools.SparseVIT import Sparse_Self_Attention
from Tools.layer import *
from Tools.config import config
from collections import defaultdict

from torchaudio.models.wav2vec2.components import FeedForward

from Tools.layer import *
from Tools.config import config
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from Tools.attention_masked import *

device = torch.device(config['cuda'] if torch.cuda.is_available() else 'cpu')
# define the transformer backbone here
STEncoderLayer = nn.TransformerEncoderLayer
STEncoder = nn.TransformerEncoder
class Clip_STTran(nn.Module):
    def __init__(self, d_model, visual_dim, target_dim, feat_dim, num_v, dropout=0.5):
        super().__init__()

        self.fusion_fc = nn.Linear(config["visual_dim"] * 3 + config["feat_dims"] + config["semantic_dim"], d_model)  #VRP 512 * 3 + 20 + 300*2
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])

        self.i_linear = nn.ModuleList([nn.Linear(d_model, d_model), nn.Linear(target_dim, d_model)])
        self.o_linear = nn.ModuleList([nn.Linear(d_model, config['num_class']), nn.Linear(config['max_target'], 1)])
        self.pos1 = PositionalEncoding(d_model, dropout=0.1, max_len=config['max_frames'])
        self.pos2 = PositionalEncoding(d_model, dropout=0.1, max_len=config['max_target'])
        # temporal encoder
        global_encoder = STEncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=config["num_heads"], batch_first=True)
        self.global_transformer = STEncoder(global_encoder, num_layers=config["temporal_layer"])
        # spatial encoder
        local_encoder = STEncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=config["num_heads"], batch_first=True)
        self.local_transformer = STEncoder(local_encoder, num_layers=config["spatial_layer"])
        #self.diff_func = get_derivatives(d_model=config['d_model'], diff_feat=config["diff_feat"], brownian_size=config["brownian_size"])
        #mask transformer


        self.loss = nn.CrossEntropyLoss()


    def build_sequences(video_ids, obj_classes, frame_ids):
        group_map = defaultdict(list)
        for idx, (vid, obj) in enumerate(zip(video_ids, obj_classes)):
            key = (vid, obj)
            group_map[key].append(idx)
        return [torch.tensor(v) for v in group_map.values()]

    def build_sequences_from_frame_list(frame_list):
        return [torch.tensor(list(range(len(frame_list))))]

    def spatial_message_passing(self, rel_features, device):
        frame_features = rel_features
        masks = torch.zeros(frame_features.shape[:2], dtype=torch.bool).to(device)
        rel_features = self.local_transformer(frame_features, src_key_padding_mask=masks)
        return rel_features

    def temporal_message_passing(self, rel_features, device):
        sequence_features = rel_features
        masks = torch.zeros((sequence_features.shape[0], sequence_features.shape[1]), dtype=torch.bool).to(device)
        pos_index = torch.arange(sequence_features.size(0), device=device)
        sequence_features = self.pos1(sequence_features, pos_index)
        seq_len = sequence_features.shape[1]
        in_mask = (1 - torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)).bool()
        rel_features = self.global_transformer(sequence_features, src_key_padding_mask=masks, mask=in_mask)
        return rel_features

    def sde_prediction(self, global_output, device):
        frame_list = torch.arange(config["max_frames"]).tolist()
        frame_idx = torch.tensor([frame_list.index(f) for f in frame_list])
        unique_times = torch.unique(frame_idx)
        frames_ranges = [0]
        for t in unique_times:
            frames_ranges.append((frame_idx == t).sum().item() + frames_ranges[-1])
        frames_ranges = torch.tensor(frames_ranges, device=device)

        anticipated_vals = torch.zeros(config["pred_step"], 0, config["d_model"], device=device)
        curr_id = 0
        for i in range(len(frames_ranges) - 1):
            end = frames_ranges[i + 1]
            if curr_id == end:
                continue
            batch_y0 = global_output[:, -1, :]  # [B, D]
            batch_times = torch.arange(config["pred_step"] + 1).float().to(device)
            ret = sdeint(self.diff_func, batch_y0, batch_times, method='reversible_heun', dt=1)[1:]  # [pred_step, B, D]
            anticipated_vals = torch.cat((anticipated_vals, ret), dim=1)
            curr_id = end
        return anticipated_vals

    def encode_with_mask_transformer(self, rel_features, T=30):
        B, TR, D = rel_features.shape
        T = T
        R = TR // T
        if not hasattr(self, 'mask_transformer') or self.mask_transformer.shape != (T, R, 1):
            self.mask_transformer = MaskTransformer(
                shape=(T, R, 1),
                d_model=config["d_model"],
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_dim"],
                mask_depth=config["mask_depth"],
                mask_heads=config["mask_head"],
                use_first_last=True
            ).to(rel_features.device)
            self.mask_transformer.shape = (T, R, 1)

        rel_features_5d = rel_features.view(B, T, R, 1, D)
        rel_features_out = self.mask_transformer(rel_features_5d)
        return rel_features_out

    def forward(self, src, src_mask, tgt, tgt_mask, device,
            video_id, obj, subj, frame_list, rel_list):
        # target, data and mask
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target, dim=2)
        src_mask[:, -config['pred_step']:] = 0
        tgt_mask[:, -config['pred_step']:] = 0

        src[:, -config['pred_step']:, :] = 0
        tgt[:, -config['pred_step']:, :] = 0
        # feat
        clip_feat_s = src[:, :-config['pred_step'], :config["visual_dim"]]  # subject features, shape [N, C]
        clip_feat_u = src[:, :-config['pred_step'], config["visual_dim"]:config["visual_dim"]*2]  # object features, shape [N, C]
        clip_feat_o = src[:, :-config['pred_step'], config["visual_dim"]*2:config["visual_dim"]*3]  # union region features, shape [N, C]
        clip_fused = torch.cat([clip_feat_s, clip_feat_u, clip_feat_o], dim=-1)  # [N, 3*C]
        spatial_feat = src[:,  :, config["visual_dim"]*3:config["visual_dim"]*3+config["spatial_dim"]]  # [N, 20]
        spatial_encoded = self.spatial(spatial_feat)  # [N, feat_dim]
        semantic_feat = src[:,  :, config["visual_dim"]*3+20:]  # [N, 300] or [N, 600] from GloVe / CLIP
        if config["dataset"] == "AGP":
            semantic_feat = semantic_feat[:,:, config["semantic_dim"]:]


        rel_features = torch.cat([clip_fused, spatial_encoded, semantic_feat], dim=-1)  # [N, D]
        rel_features = self.fusion_fc(rel_features)  # [N, hidden_dim]
        #mask module
        #rel_features = self.encode_with_mask_transformer(rel_features, T=config["max_frames"])
        #spatial
        #rel_features = self.spatial_message_passing(rel_features, device)
        #temporal
        #rel_features = self.temporal_message_passing(rel_features, device)

        x = rel_features


        predict = self.o_linear[0](x).view(-1, config['max_target'], config['num_class'])
        loss = self.loss(predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1))
        return predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1), loss


class GCN_Transformer(nn.Module):
    def __init__(self, d_model, visual_dim, target_dim, feat_dim, adj_matix, num_v,  dropout):
        super().__init__()

        encoders = nn.ModuleList([
            Sparse_Self_Attention(
                dim=d_model,
                num_heads=config["num_heads"],
                sparse_size=4,
                mlp_ratio=config["d_ff"] // d_model,
                qkv_bias=True,
                drop=config["MAdropout"],
                attn_drop=config["MAdropout"],
                drop_path=0.1,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_layer_scale=config["use_layer_scale"],
                layer_scale_init_value=config["layer_scale_init_value"]
            ),
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"]))

        ])
        self.encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"]))
        ])

        self.decoder = MyDecoder(decoders)

        self.i_linear = nn.ModuleList([nn.Linear(config["feat_dims"]*4+config["semantic_dim"], d_model), nn.Linear(target_dim, d_model)])
        self.o_linear = nn.ModuleList([nn.Linear(d_model, config['num_class']), nn.Linear(config['max_target'], 1)])
        self.pos1 = PositionEncoder(d_model, config['max_frames'])
        self.pos2 = PositionEncoder(d_model, config['max_target'])
        self.loss = nn.CrossEntropyLoss()

        self.adj_matrix = adj_matix + torch.eye(adj_matix.size(0)).to(adj_matix).detach().float()
        self.gcl = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2 = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])

        # STT position
        # self.pos3 = PositionalEncoding(d_model, dropout=config["dropout"], max_len=config['max_frames'])
        # self.pos4 = PositionalEncoding(d_model, dropout=config["dropout"], max_len=config['max_target'])

        # temporal encoder
        global_encoder = STEncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=config["num_heads"],
                                        batch_first=True)
        self.global_transformer = STEncoder(global_encoder, num_layers=config["temporal_layer"])
        # spatial encoder
        local_encoder = STEncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=config["num_heads"],
                                       batch_first=True)
        self.local_transformer = STEncoder(local_encoder, num_layers=config["spatial_layer"])
        self.spatial_proj = nn.Linear(config["feat_dims"]*4+config["semantic_dim"], d_model) #1324,d_model
        self.temporal_proj = nn.Linear(d_model, config["feat_dims"]*4+config["semantic_dim"])

        # V2T T2V
        # if gct_v2t_t2v config["feat_dims"]-->config["visual_dims"]
        # if v2t_t2v_gct config["visula_dims"]-->config["feat_dims"]

        self.semantic_proj1 = nn.Linear(config["semantic_dim"],config["visual_dim"])
        self.v2t_fusion = VisionLanguageFusionModule(
            VLdmodel=config["VLdmodel"], VLnhead=config["VLnhead"], visual_dim=config["visual_dim"], dropout=config["VLdrop"]
        )

        self.t2v_fusion = VisionLanguageFusionModule(
            VLdmodel=config["VLdmodel"], VLnhead=config["VLnhead"], visual_dim=config["visual_dim"], dropout=config["VLdrop"]
        )
        self.fuse_visual = nn.Sequential(
            nn.Linear(config["visual_dim"] * 2, config["visual_dim"]),
            nn.ReLU()
        )
        self.semantic_proj2 = nn.Linear(config["semantic_dim"], config["visual_dim"])

        #self.mamba_block = MambaBlock(d_model=config["feat_dims"] + 20 + 300)


        # Dropout
        self.input_dropout = nn.Dropout(p=config["dropout_rate"])
        self.output_dropout = nn.Dropout(p=config["dropout_rate"])

    def spatial_message_passing(self, rel_features, device):
        frame_features = rel_features
        masks = torch.zeros(frame_features.shape[:2], dtype=torch.bool).to(device)
        frame_features = self.spatial_proj(frame_features)  # [B, T, 1324] -> [B, T, 128]
        rel_features = self.local_transformer(frame_features, src_key_padding_mask=masks)
        return rel_features

    def temporal_message_passing(self, rel_features, device):
        sequence_features = rel_features
        masks = torch.zeros((sequence_features.shape[0], sequence_features.shape[1]), dtype=torch.bool).to(device)
        pos_index = torch.arange(sequence_features.size(0), device=device)
        sequence_features = self.pos3(sequence_features, pos_index)
        seq_len = sequence_features.shape[1]
        in_mask = (1 - torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)).bool()
        rel_features = self.global_transformer(sequence_features, src_key_padding_mask=masks, mask=in_mask)
        rel_features = self.temporal_proj(rel_features) # [B, T, 128] -> [B, T, 1324]
        return rel_features




    def forward(self, src, src_mask, tgt, tgt_mask,epoch, device):
        #target, data and mask
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target, dim=2)
        src_mask[:, -config['pred_step']:] = 0
        tgt_mask[:, -config['pred_step']:] = 0

        src[:, -config['pred_step']:, :] = 0
        tgt[:, -config['pred_step']:, :] = 0


        #feat
        visual_feat_s = src[:, :-config['pred_step'], :1536]
        visual_feat_p = src[:, :-config['pred_step'], 1536:3072]
        visual_feat_o = src[:, :-config['pred_step'], 3072:4608]

        spatial_feat = src[:, :, 4608:4628]
        semantic_feat = src[:, :, 4628:]
        if config["dataset"] == "AGP":
            semantic_feat = semantic_feat[:, :, 300:]

        spatial_feat = self.spatial(spatial_feat)

        # V2T
        projected_semantic_feat = self.semantic_proj1(semantic_feat)
        #projected_semantic_feat = self.semantic_gate(projected_semantic_feat)
        fused_semantic_feat = self.v2t_fusion(
            tgt=projected_semantic_feat,
            memory=visual_feat_p,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None
        )
        # T2V
        fused_visual_feat = self.t2v_fusion(
            tgt=visual_feat_p,
            memory=fused_semantic_feat,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None
        )
        visual_feat_p = torch.cat([visual_feat_p, fused_visual_feat], dim=-1)
        visual_feat_p = self.fuse_visual(visual_feat_p)


        visual_feat_src = torch.stack((visual_feat_s, visual_feat_p, visual_feat_o), axis=2)

        # graph
        visual_feat = self.gcl(self.adj_matrix, visual_feat_src)
        visual_feat = self.gcl2(self.adj_matrix, visual_feat)
        tmp_feat = torch.zeros(visual_feat.size(0), config["max_frames"], config["feat_dims"]).to(device)
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 0, :].squeeze()
        visual_feat_s = tmp_feat
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 1, :].squeeze()
        visual_feat_p = tmp_feat
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 2, :].squeeze()
        visual_feat_o = tmp_feat

        # Step 1: Spatial message passing
        #src_gcn = self.spatial_message_passing(src_gcn, device)  # shape: [B, T, D]

        # Step 2: Temporal message passing
        #src_gcn = self.temporal_message_passing(src_gcn, device)  # shape: [B, T, D]



        src_gcn = torch.cat([visual_feat_s, visual_feat_p, visual_feat_o, spatial_feat, semantic_feat], 2)

        #src_gcn = self.gcn_gate(src_gcn)
        #src_gcn = self.gcn_se(src_gcn)

        #postion wo_dropout
        src = self.pos1(self.i_linear[0](src_gcn))
        tgt = self.pos2(self.i_linear[1](tgt))
        #postion dropout
        # src = self.pos1(self.input_dropout(self.i_linear[0](src_gcn)))
        # tgt = self.pos2(self.input_dropout(self.i_linear[1](tgt)))

        #Encoder-Decoder
        x = self.encoder(src, src_mask)  # [nb, len1, hid]
        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        #x = self.decoder(tgt, src, src_mask, tgt_mask)
        #x = self.output_dropout(x)

        # # sde
        # global_output = x
        # x = self.sde_prediction(global_output, device)

        predict = self.o_linear[0](x).view(-1, config['max_target'], config['num_class'])
        #predict = predict.permute(0, 2, 1)
        #predict = self.o_linear[1](predict).view(-1, config['num_class'])
        #loss = self.loss(predict, target[:,-1])
        #return predict,target[:,-1],loss

        loss = self.loss(predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1))
        # merge rebuild loss
        #loss = loss+recon_loss*config["alpha"]
        #return predict[:,-6,:].reshape(-1, config['num_class']),target[:,-6],loss
        return predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1), loss



class Transformer(nn.Module):
    def __init__(self, d_model, visual_dim, target_dim):
        super().__init__()

        encoders = nn.ModuleList([
            # QuadrangleEncoderWrapper(
            #     QuadrangleAttention(
            #         dim=d_model,
            #         num_heads=config["num_heads"],
            #         qkv_bias=True,
            #         attn_drop=config["QAdropout"],
            #         proj_drop=config["QAdropout"],
            #         window_size=config["window_size"],
            #         rpe=config["rpe"],
            #         coords_lambda=config["coords"]
            #     ),
            #     norm_dim=config["d_model"],
            #     h=config["batch_size"],
            #     w=config["max_frames"]
            # ),
            Sparse_Self_Attention(
                dim=d_model,
                num_heads=config["num_heads"],
                sparse_size=config["sparse_size"],
                mlp_ratio=config["d_ff"] // d_model,
                qkv_bias=True,
                drop=config["SAdropout"],
                attn_drop=config["SAdropout"],
                drop_path=config["drop_path"],
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_layer_scale=config["use_layer_scale"],
                layer_scale_init_value=config["layer_scale_init_value"]
            ),
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),

            # QuadrangleEncoderWrapper(
            #     QuadrangleAttention(
            #         dim=d_model,
            #         num_heads=config["num_heads"],
            #         qkv_bias=True,
            #         attn_drop=config["QAdropout"],
            #         proj_drop=config["QAdropout"],
            #         window_size=config["window_size"],
            #         rpe=config["rpe"],
            #         coords_lambda=config["coords"]
            #     ),
            #     norm_dim=config["d_model"],
            #     h=config["batch_size"],
            #     w=config["max_frames"]
            # ),
            Sparse_Self_Attention(
                dim=d_model,
                num_heads=config["num_heads"],
                sparse_size=config["sparse_size"],
                mlp_ratio=config["d_ff"] // d_model,
                qkv_bias=True,
                drop=config["SAdropout"],
                attn_drop=config["SAdropout"],
                drop_path=config["drop_path"],
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                use_layer_scale=config["use_layer_scale"],
                layer_scale_init_value=config["layer_scale_init_value"]
            ),
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),


            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),
        ])
        self.encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([

            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),

            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         MultiHeadAttention(config['num_heads'], d_model, config["MAdropout"]),
                         PositionWiseFeedForward(d_model, config['d_ff'], config["PFdropout"])),
        ])

        self.decoder = MyDecoder(decoders)
        # config["feat_dims"]*2+config['semantic_dim']+config['spatial_dim']
        self.i_linear = nn.ModuleList([nn.Linear(config["feat_dims"]*4+config["semantic_dim"], d_model), nn.Linear(target_dim, d_model),
                                       nn.Linear(config["visual_dim"], config["feat_dims"])])
        self.o_linear = nn.ModuleList([nn.Linear(d_model, config["num_class"]), nn.Linear(config["max_target"], 1)])
        self.pos1 = PositionEncoder(d_model, config["max_frames"])
        self.pos2 = PositionEncoder(d_model, config["max_target"])
        self.loss = nn.CrossEntropyLoss()
        self.semantic = nn.Linear(config["semantic_dim"], config["feat_dims"])
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])

        self.semantic_proj1 = nn.Linear(config["semantic_dim"], config["visual_dim"])
        self.v2t_fusion = VisionLanguageFusionModule(
            VLdmodel=config["VLdmodel"], VLnhead=config["VLnhead"], visual_dim=config["visual_dim"],
            dropout=config["VLdrop"]
        )

        self.t2v_fusion = VisionLanguageFusionModule(
            VLdmodel=config["VLdmodel"], VLnhead=config["VLnhead"], visual_dim=config["visual_dim"],
            dropout=config["VLdrop"]
        )
        self.fuse_visual = nn.Sequential(
            nn.Linear(config["visual_dim"] * 2, config["visual_dim"]),
            nn.ReLU()
        )
        self.semantic_proj2 = nn.Linear(config["visual_dim"], config["semantic_dim"])


    def forward(self, src, src_mask, tgt, tgt_mask, epoch, device):

        # target, data and mask
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target, dim=2)
        src_mask[:, -config["pred_step"]:] = 0
        tgt_mask[:, -config["pred_step"]:] = 0

        src[:, -config["pred_step"]:, :] = 0
        tgt[:, -config["pred_step"]:, :] = 0

        # feat
        visual_feat_s = src[:, :-config["pred_step"], :1536]
        visual_feat_p = src[:, :-config["pred_step"], 1536:3072]
        visual_feat_o = src[:, :-config["pred_step"], 3072:4608]

        spatial_feat = src[:, :, 4608:4628]
        semantic_feat = src[:, :, 4628:]
        if config["dataset"] == "AGP":
            semantic_feat = semantic_feat[:, :, 300:]

        spatial_feat = self.spatial(spatial_feat)
        #semantic_feat = self.semantic(semantic_feat)



        # V2T
        projected_semantic_feat = self.semantic_proj1(semantic_feat)
        fused_semantic_feat = self.v2t_fusion(
            tgt=projected_semantic_feat,
            memory=visual_feat_p,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None
        )
        #semantic_feat = self.semantic_proj2(fused_semantic_feat)
        # T2V
        fused_visual_feat = self.t2v_fusion(
            tgt=visual_feat_p,
            memory=fused_semantic_feat,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None
        )
        visual_feat_p = torch.cat([visual_feat_p, fused_visual_feat], dim=-1)
        visual_feat_p = self.fuse_visual(visual_feat_p)


        tmp_feat = torch.zeros(visual_feat_s.size(0), config["max_frames"], config["feat_dims"]).to(device)
        tmp_feat[:, :-config["pred_step"], :] = self.i_linear[2](visual_feat_s)  #wo ARconv
        #tmp_feat[:, :-config["pred_step"], :] = vs[:, :, :]  #ARconv
        visual_feat_s = tmp_feat
        tmp_feat[:, :-config["pred_step"], :] = self.i_linear[2](visual_feat_p)
        visual_feat_p = tmp_feat
        tmp_feat[:, :-config["pred_step"], :] = self.i_linear[2](visual_feat_o)  #wo ARconv
        #tmp_feat[:, :-config["pred_step"], :] = vo[:, :, :]  #ARconv
        visual_feat_o = tmp_feat

        src_feat = torch.cat([visual_feat_s, visual_feat_p, visual_feat_o, spatial_feat, semantic_feat], 2)
        #src_feat = torch.cat([visual_feat_s, visual_feat_o, spatial_feat, semantic_feat], 2)

        src = self.pos1(self.i_linear[0](src_feat))
        tgt = self.pos2(self.i_linear[1](tgt))
        x = self.encoder(src, src_mask)  # [nb, len1, hid]

        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        predict = self.o_linear[0](x).view(-1, config['max_target'], config['num_class'])
        #predict = predict.permute(0, 2, 1)
        #predict = self.o_linear[1](predict).view(-1, config['num_class'])

        loss = self.loss(predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1))

        #return predict[:,-1:,:].reshape(-1, config['num_class']),target[:,-1],loss
        return predict[:, -config["pred_step"]:, :].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1), loss



class GGCN(nn.Module):
    def __init__(self, visual_dim, target_dim, feat_dim, adj_matix, num_v, dropout=0.5):
        super(GGCN, self).__init__()

        self.adj_matrix = adj_matix + torch.eye(adj_matix.size(0)).to(adj_matix).detach().float()
        self.gcl = GraphConvolution(visual_dim, 1024, num_v, dropout=dropout)
        self.gcl2 = GraphConvolution(1024, feat_dim, num_v, dropout=dropout)
        self.loss = nn.CrossEntropyLoss()
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])
        self.fc = nn.Linear(config["feat_dims"]*3, config["feat_dims"])
        self.fc1 = nn.Linear(config["feat_dims"], config["num_class"])


    def forward(self, src, src_mask, tgt, tgt_mask, device):

        # target, data and mask
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target)
        src_mask[:, -config['pred_step']:] = 0
        tgt_mask[:, -config['pred_step']:] = 0

        src[:, -config['pred_step']:, :] = 0
        tgt[:, -config['pred_step']:, :] = 0
        # feat
        visual_feat_s = src[:, :-config['pred_step'], :1536]
        visual_feat_p = src[:, :-config['pred_step'], 1536:3072]
        visual_feat_o = src[:, :-config['pred_step'], 3072:4608]

        spatial_feat = src[:, :, 4608:4628]
        semantic_feat = src[:, :, 4628:]
        if config["dataset"] == "AGP":
            semantic_feat = semantic_feat[:, :, 300:]

        spatial_feat = self.spatial(spatial_feat)

        visual_feat_src = torch.stack((visual_feat_s, visual_feat_p, visual_feat_o), axis=2)

        # graph
        visual_feat = self.gcl(self.adj_matrix, visual_feat_src)
        visual_feat = self.gcl2(self.adj_matrix, visual_feat)
        tmp_feat = torch.zeros(visual_feat.size(0), config["max_frames"], config["feat_dims"]).to(device)
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 0, :].squeeze()
        visual_feat_s = tmp_feat
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 1, :].squeeze()
        visual_feat_p = tmp_feat
        tmp_feat[:, :-config['pred_step'], :] = visual_feat[:, :, 2, :].squeeze()
        visual_feat_o = tmp_feat

        src_gcn = torch.cat([visual_feat_s, visual_feat_p, visual_feat_o], 2)

        predict = self.fc(src_gcn)
        predict = self.fc1(predict)
        target = target.unsqueeze(0)

        loss = self.loss(predict[:,-config["pred_step"]:,:].reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1))

        return predict[:,-1:,:].reshape(-1, config['num_class']),target[:,-1],loss



class RNNModel(nn.Module):

    def __init__(self):

        super(RNNModel, self).__init__()

        self.hiddenNum = config["feat_dims"]
        self.inputDim = config["feat_dims"]*4+config['semantic_dim']
        self.outputDim = config["num_class"]
        self.layerNum = 1
        self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hiddenNum,
                           num_layers=self.layerNum, dropout=0.0,
                           nonlinearity="tanh", batch_first=True, )
        self.fc = nn.Linear(self.hiddenNum, self.outputDim)
        self.spatial = nn.Linear(config["spatial_dim"], config["feat_dims"])
        self.visual = nn.Linear(config["visual_dim"], config["feat_dims"])

        self.loss = nn.CrossEntropyLoss()

    def forward(self, src, src_mask, tgt, tgt_mask, device):

        # target, data and mask
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target)
        src_mask[:, -config["pred_step"]:] = 0
        tgt_mask[:, -config["pred_step"]:] = 0

        src[:, -config["pred_step"]:, :] = 0
        tgt[:, -config["pred_step"]:, :] = 0
        # feat
        visual_feat_s = src[:, :-config["pred_step"], :1536]
        visual_feat_p = src[:, :-config["pred_step"], 1536:3072]
        visual_feat_o = src[:, :-config["pred_step"], 3072:4608]

        spatial_feat = src[:, :, 4608:4628]
        semantic_feat = src[:, :, 4628:]
        if config["dataset"] == "AGP":
            semantic_feat = semantic_feat[:, :, 300:]

        spatial_feat = self.spatial(spatial_feat)
        #semantic_feat = self.semantic(semantic_feat)

        tmp_feat = torch.zeros(visual_feat_s.size(0), config["max_frames"], config["feat_dims"]).to(device)
        tmp_feat[:, :-config["pred_step"], :] = self.visual(visual_feat_s)
        visual_feat_s = tmp_feat
        tmp_feat[:, :-config["pred_step"], :] = self.visual(visual_feat_p)
        visual_feat_p = tmp_feat
        tmp_feat[:, :-config["pred_step"], :] = self.visual(visual_feat_o)
        visual_feat_o = tmp_feat

        src_feat = torch.cat([visual_feat_s, visual_feat_p, visual_feat_o, spatial_feat, semantic_feat], 2)

        batchSize = src_feat.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize , self.hiddenNum)).to(device)

        rnnOutput, hn = self.cell(src_feat, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        predict = self.fc(hn)

        loss = self.loss(predict.reshape(-1, config['num_class']), target[:,-config["pred_step"]:].reshape(-1))

        return predict.reshape(-1, config['num_class']),target[:,-1],loss

