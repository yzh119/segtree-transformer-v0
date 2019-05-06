import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .op import *


class PositionwiseFeedForward(nn.Module):
    '''
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    '''
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(th.relu(self.w_1(x))))

csr_cache = {}

def get_csrs(g, device):
    n = g.number_of_nodes()
    global cache
    #if (n, device) in csr_cache:
    #    return csr_cache[(n, device)]

    #if len(csr_cache) > 5:
    #    csr_cache.pop(list(csr_cache.keys())[0])

    out_csr = g.adjacency_matrix_scipy(transpose=True, fmt='csr')
    out_csr = (th.tensor(out_csr.indptr, dtype=th.long, device=device),
               th.tensor(out_csr.indices, dtype=th.long, device=device),
               th.tensor(out_csr.data, dtype=th.long, device=device))
    in_csr = g.adjacency_matrix_scipy(fmt='csr')
    in_csr = (th.tensor(in_csr.indptr, dtype=th.long, device=device),
              th.tensor(in_csr.indices, dtype=th.long, device=device),
              th.tensor(in_csr.data, dtype=th.long, device=device))

    #csr_cache[(n, device)] = (out_csr, in_csr)
    return out_csr, in_csr

class SparseSelfAttention(nn.Module):
    MAX_ETYPE = 200
    def __init__(self, dim_model, h, dim_ff, rel_pos=False, ffn=True, dropouth=0.1, dropouta=0.1):
        super(SparseSelfAttention, self).__init__()
        self.dim_model = dim_model
        self.h = h
        self.dim_ff = dim_ff
        self.d_k = self.dim_model // self.h
        self.rel_pos = rel_pos
        self.drop_h = nn.Dropout(dropouth)
        self.drop_att = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(self.dim_model)
        self.norm_inter = nn.LayerNorm(self.dim_model)

        self.linears = nn.ModuleList(
            [nn.Linear(dim_model, dim_model, bias=False) for _ in range(4)]
        )

        if self.rel_pos:
            self.embed_ak = nn.Embedding(self.MAX_ETYPE, self.d_k)

        if ffn:
            self.ffn = nn.Sequential(
                PositionwiseFeedForward(self.dim_model, self.dim_ff, dropout=dropouth),
                nn.Dropout(dropouth)
            )
        else:
            self.ffn = None


    def forward(self, g):
        device = next(self.parameters()).device
        h = g.ndata['h'] # get pos embedding
        if self.rel_pos:
            g.edata['ak'] = self.embed_ak(g.edata['etype'])

        # get in and out csr
        out_csr, in_csr = get_csrs(g, device)
        
        # get queries
        g.ndata['q'] = self.linears[0](h).view(-1, self.h, self.d_k)
        # get keys and values
        g.ndata['k'] = self.linears[1](h).view(-1, self.h, self.d_k)
        g.ndata['v'] = self.linears[2](h).view(-1, self.h, self.d_k)

        edata = MaskedMMCSR.apply(
            out_csr[0], out_csr[2], out_csr[1], in_csr[0], in_csr[2], in_csr[1], g.ndata['k'], g.ndata['q'])

        e_rel = 0
        if self.rel_pos:
            e_rel = NodeMulEdge.apply(in_csr[0], in_csr[2], g.ndata['q'], g.edata['ak'])

        edata = self.drop_att(SparseSoftmax.apply(in_csr[0], in_csr[2], (e_rel + edata) / np.sqrt(self.d_k)))
        a = VectorSPMM.apply(
            in_csr[0], in_csr[2], in_csr[1],
            out_csr[0], out_csr[2], out_csr[1],
            edata, g.ndata['v']).view(-1, self.dim_model)
        o = self.drop_h(self.linears[3](a))
        h = self.norm_in(h + o)

        if self.ffn:
            h = self.norm_inter(h + self.ffn(h))

        g.ndata['h_'] = h
