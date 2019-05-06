from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from torchtext import datasets
from torchtext.datasets import LanguageModelingDataset
from .base import GraphBatcher, Batch
import numpy as np
import torch as th
import dgl


def get_lm_dataset(name='ptb'):
    if name == 'ptb':
        return datasets.PennTreebank
    elif name == 'wiki-2':
        return datasets.WikiText2
    elif name == 'wiki-103':
        return datasets.WikiText103
    else:
        raise KeyError('invalid dataset name')

class LMBatcher(GraphBatcher):
    def __init__(self, TEXT, fully=False):
        super(LMBatcher, self).__init__(triu=True, fully=fully)
        self.TEXT = TEXT
        self._cache = {}

    def __call__(self, batch):
        data = []
        labels = []

        v_shift, e_shift = 0, 0
        row, col = [], []
        leaf_ids, internal_ids, readout_ids = [], [], []
        pos_arr = []
        etypes = []

        for sent in batch:
            start = 0
            sent = self.TEXT.numericalize([sent]).view(-1)
            data.append(sent[:-1])
            labels.append(sent[1:])
            length = len(sent) - 1

            # get graph
            g = self._get_graph(length)
            # get pos
            pos_arr.append(th.arange(length))
            # gather leaf nodes
            leaf_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift)))
            readout_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift, start=start)))
            # gather internal nodes
            internal_ids.append(th.from_numpy(g.internal_ids(v_shift=v_shift)))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift)
            row.append(src)
            col.append(dst)
            etypes.append(etype)
            # update shift
            v_shift += g.number_of_nodes
            e_shift += g.number_of_edges

        n = v_shift
        leaf_ids = th.cat(leaf_ids)
        internal_ids = th.cat(internal_ids)
        readout_ids = th.cat(readout_ids)
        pos_arr = th.cat(pos_arr)
        row, col = map(np.concatenate, (row, col))
        etypes = np.concatenate(etypes)
        coo = coo_matrix((np.zeros_like(row), (row, col)), shape=(n, n))
        g = dgl.DGLGraph(coo, readonly=True)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)

        data = th.cat(data)
        labels = th.cat(labels)
        g.edata['etype'] = th.from_numpy(etypes)
        g.nodes[leaf_ids].data['x'] = data
        g.nodes[leaf_ids].data['pos'] = pos_arr

        return Batch(g=g, readout_ids=readout_ids, leaf_ids=leaf_ids, internal_ids=internal_ids, y=labels)


class LMDataset(Dataset):
    def __init__(self, lm_dataset, max_length=35, part=(0,1)):
        n = len(lm_dataset[0].text)
        part_size = (n + part[1] - 1) // part[1]
        self.data = lm_dataset[0].text[part_size * part[0]: part_size * (part[0] + 1)]
        self.max_length = max_length

    def __len__(self):
        return (len(self.data) + self.max_length - 1) // self.max_length

    def __getitem__(self, index):
        return self.data[index * self.max_length: (index + 1) * self.max_length]

