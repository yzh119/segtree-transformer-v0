from graphbuilder import SegmentTree, FullyConnected
import torch as th

class GraphBatcher:
    def __init__(self, triu=False, fully=False):
        self._graph_cache = {}
        self.triu = triu
        self.fully = fully

    def _get_graph(self, l, **kwargs):
        if l in self._graph_cache:
            return self._graph_cache[l]
        else:
            if self.fully:
                new_g = FullyConnected(l, triu=self.triu)
            else:
                new_g = SegmentTree(l, triu=self.triu)
            self._graph_cache[l] = new_g
            return new_g

    def __call__(self, batch):
        raise NotImplementedError

class Batch:
    def __init__(self, g=None, readout_ids=None, leaf_ids=None, internal_ids=None, y=None):
        self.g = g
        self.readout_ids = readout_ids
        self.leaf_ids = leaf_ids
        self.internal_ids = internal_ids
        self.y = y

