from .attention import *
from .embedding import *
from .utils import *
import dgl
import dgl.function as fn


class PairClassifier(nn.Module):
    def __init__(self, dim_model, dim_hidden, n_classes):
        super(PairClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(dim_hidden, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """
    log(softmax(Wx + b))
    """
    def __init__(self, dim_model, n_classes):
        super(Generator, self).__init__()
        self.proj = nn.Linear(dim_model, n_classes)

    def forward(self, x):
        return th.log_softmax(self.proj(x), dim=-1)


class SegmentTreeTransformer(nn.Module):
    def __init__(self, vocab_size, dim_embed, dim_model, dim_ff, h, n_classes, n_layers,
                 dropouti=0.1, dropouth=0.1, dropouta=0.1, dropoutc=0, rel_pos=False, ffn=True, pair=False):
        super(SegmentTreeTransformer, self).__init__()
        self.dim_embed = dim_embed
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.h = h
        assert self.dim_model % self.h == 0
        self.rel_pos = rel_pos
        self.ffn = ffn
        self.pair = pair

        if vocab_size > 0:
            self.embed = Embedding(vocab_size, dim_embed, scale=not rel_pos)
        else:
            self.embed = None

        if dim_embed != dim_model:
            self.embed_to_hidden = Embed2Hidden(dim_embed, dim_model)
        else:
            self.embed_to_hidden = None

        if not self.rel_pos:
            self.pos_enc = PositionalEncoding(dim_model)

        self.emb_dropout = nn.Dropout(dropouti)
        self.cls_dropout = nn.Dropout(dropoutc)

        layer_list = []
        for _ in range(n_layers):
            layer_list.append(
                SparseSelfAttention(self.dim_model, self.h, self.dim_ff,
                                    rel_pos=self.rel_pos, ffn=self.ffn, dropouth=dropouth, dropouta=dropouta))

        self.layers = nn.ModuleList(layer_list)

        if self.pair:
            self.classifier = PairClassifier(4 * dim_model, 1024, n_classes)
        else:
            self.generator = Generator(dim_model, n_classes)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, batch):
        g = batch.g
        leaf_ids = batch.leaf_ids
        internal_ids = batch.internal_ids

        # get embedding
        if self.embed:
            h = self.embed(g.nodes[leaf_ids].data['x'])

        # embed to hidden
        if self.embed_to_hidden:
            g.nodes[leaf_ids].data['h'] = self.embed_to_hidden(h)
        else:
            g.nodes[leaf_ids].data['h'] = h

        # add pos encoding
        if not self.rel_pos:
           g.nodes[leaf_ids].data['h'] += self.pos_enc(g.nodes[leaf_ids].data['pos'])

        # input dropout
        g.nodes[leaf_ids].data['h'] = self.norm(self.emb_dropout(g.nodes[leaf_ids].data['h']))

        # go through the layers
        for i, layer in enumerate(self.layers):
            layer(g)
            g.ndata['h'] = g.ndata['h_']

        # output
        if self.pair:
            x, y = (g.nodes[batch.readout_ids].data['h']).view(-1, 2, self.dim_model).unbind(dim=1)
            output = self.classifier(th.cat([x, y, x - y, x * y], dim=-1))
        else:
            output = self.generator(self.cls_dropout(g.nodes[batch.readout_ids].data['h']))

        clear_feature(g)
        return output

import torch.nn.init as INIT

def make_model(vocab_size, dim_embed, dim_model, dim_ff, h, n_classes, n_layers,
               dropouti=0.1, dropouth=0.1, dropouta=0.1, dropoutc=0, rel_pos=False, ffn=True, pair=False):
    model = SegmentTreeTransformer(vocab_size, dim_embed, dim_model, dim_ff, h, n_classes, n_layers,
                                   dropouti, dropouth, dropouta, dropoutc, rel_pos, ffn, pair)
    for p in model.parameters():
        if p.dim() > 1 and p.size(0) != vocab_size:
            INIT.xavier_uniform_(p)
        if p.dim() > 1 and p.size(0) == vocab_size:
            INIT.normal_(p, 0, 1. / np.sqrt(dim_embed))
    return model
