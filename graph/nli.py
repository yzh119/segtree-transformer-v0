from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from torchtext import datasets
from .base import GraphBatcher, Batch
import numpy as np
import torch as th
import dgl


def get_nli_dataset(name='snli'):
    if name == 'snli':
        return datasets.SNLI
    elif name == 'mnli':
        return datasets.MultiNLI
    else:
        raise KeyError('invalid dataset name')


class NLIBatcher(GraphBatcher):
    def __init__(self, TEXT, LABEL, fully=False, neigh=0):
        super(NLIBatcher, self).__init__(triu=True, fully=fully, neigh=neigh)
        self.TEXT = TEXT
        self.LABEL = LABEL
        self._cache = {}

    def __call__(self, batch):
        data = []
        labels = []

        v_shift, e_shift = 0, 0
        row, col = [], []
        root_ids, leaf_ids = [], []
        pos_arr, seg_arr = [], []
        etypes = []

        for premise, hypo, label in batch:
            premise = self.TEXT.numericalize([premise]).view(-1)
            hypo = self.TEXT.numericalize([hypo]).view(-1)
            label = self.LABEL.numericalize([label]).view(-1)

            data.append(th.cat([premise, hypo], -1))
            labels.append(label)

            length = len(premise) + len(hypo)
            # building premise graph
            # get graph
            g = self._get_graph(length)#, split=len(premise))
            # get pos
            pos_arr.append(th.arange(length))
            seg_arr.append(th.zeros(len(premise)).long())
            seg_arr.append(th.ones(len(hypo)).long())
            # gather leaf nodes
            root_ids.append(g.root_id(v_shift=v_shift))
            leaf_ids.append(th.from_numpy(g.leaf_ids(v_shift=v_shift)))
            # gather edges
            src, dst, etype = g.get_edges(v_shift=v_shift)
            row.append(src)
            col.append(dst)
            etypes.append(etype)
            # update shift
            v_shift += g.number_of_nodes
            e_shift += g.number_of_edges


        n = v_shift
        root_ids = th.tensor(root_ids)
        leaf_ids = th.cat(leaf_ids)
        pos_arr = th.cat(pos_arr)
        seg_arr = th.cat(seg_arr)
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
        g.nodes[leaf_ids].data['seg'] = seg_arr 

        return Batch(g=g, readout_ids=root_ids, leaf_ids=leaf_ids, y=labels)


class NLIDataset(Dataset):
    def __init__(self, nli_dataset):
        self.data = nli_dataset.examples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index].premise, self.data[index].hypothesis, self.data[index].label
