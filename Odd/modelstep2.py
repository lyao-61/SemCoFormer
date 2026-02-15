
class Transformer2Step(nn.Module):
    def __init__(self, d_model, visual_dim, word_dim):
        super().__init__()

        encoders = nn.ModuleList([
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            EncoderLayer([config['max_frames'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
        ])
        self.encoder = MyEncoder(encoders)

        decoders = nn.ModuleList([
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff'])),
            DecoderLayer([config['max_target'], d_model],
                         MultiHeadAttention(config['num_heads'], d_model),
                         MultiHeadAttention(config['num_heads'], d_model),
                         PositionWiseFeedForward(d_model, config['d_ff']))
        ])

        self.decoder = MyDecoder(decoders)

        self.i_linear = nn.ModuleList([nn.Linear(visual_dim, d_model), nn.Linear(word_dim, d_model)])
        self.o_linear = nn.ModuleList([nn.Linear(d_model, config['num_class']), nn.Linear(config['max_target'], 1)])
        self.pos1 = PositionEncoder(d_model, config['max_frames'])
        self.pos2 = PositionEncoder(d_model, config['max_target'])
        self.loss = nn.CrossEntropyLoss()
        self.constant = 1e-6

    def forward(self, src, src_mask, tgt, tgt_mask):
        target = torch.topk(tgt, 1, dim=2)[1]
        target = torch.squeeze(target)
        tgt[:,-2:,:] = 0
        src[:,-2:,:] = 0
        src = self.pos1(self.i_linear[0](src))
        tgt = self.pos2(self.i_linear[1](tgt))
        x = self.encoder(src, src_mask)  # [nb, len1, hid]

        x = self.decoder(tgt, x, src_mask, tgt_mask)  # [nb, len2, hid]
        predict = self.o_linear[0](x).view(-1, config['max_target'], config['num_class'])
        #predict = predict.permute(0, 2, 1)
        #predict = self.o_linear[1](predict).view(-1, config['num_class'])

        loss = self.loss(predict[:,-2:,:].reshape(-1, config['num_class']), target[:,-2:].reshape(-1))

        return predict[:,-1,:].reshape(-1, config['num_class']),target[:,-1],loss
