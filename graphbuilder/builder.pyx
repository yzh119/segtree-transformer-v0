#cython: boundscheck=False, wraparound=False, embedsignature=True, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport log2, ceil

cdef type DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

cdef inline bool overlap(size_t i, size_t j, bool left=True):
    if j == 0:
        return False
    if left:
        return (i >> (j - 1)) & 1 == 0
    else:
        return (i >> (j - 1)) & 1 == 1

"""
edges:
(src, dst, etype)

topdown/bottomup:
(eid)

etype:
3 * l:                 bottom up at level l
3 * l + 1:             topdown left at level l
3 * l + 2:             topdown right at level l
"""

cdef class SegmentTree:
    cdef size_t length, n_nodes, n_lvl, n_edges
    cdef bool triu
    cdef vector[DTYPE_t] topdown, bottomup
    cdef vector[DTYPE_t] edges[3]
    cdef vector[DTYPE_t] n_nodes_arr, shift

    def __cinit__(self, size_t length, bool triu=False):
        self.n_nodes = 0
        self.length = length
        self.triu = triu
        self.n_lvl = 1
        self.build_graph()

    def __reduce__(self):
        return SegmentTree, (self.length, self.triu, self.extend)

    def build_graph(self):
        # count nodes and compute shift for each level
        cdef size_t i = self.length
        self.shift.push_back(0) 
        while i >= 2:
            self.n_nodes_arr.push_back(i)
            self.n_nodes += i
            self.shift.push_back(self.n_nodes)
            self.n_lvl += 1
            i = (i + 1) >> 1

        # handle root
        self.n_nodes_arr.push_back(1)
        self.n_nodes += 1

        # add edges
        self.n_edges = 0
        cdef size_t j, v
        for i in range(self.length):
            v, shift = i, 0
            for j in range(self.n_lvl):
                # Self loop in top-down/bottom-up connections
                if (i == (v << j)):
                    self.edges[0].push_back(shift + v)
                    self.edges[1].push_back(shift + v)
                    self.edges[2].push_back(0)
                    if (j == 0):
                        self.topdown.push_back(self.n_edges)
                    else:
                        self.bottomup.push_back(self.n_edges)
                    self.n_edges += 1

                # Add top down connection
                # right connection
                if (not self.triu) and i + (1 << j) < self.length:  #v + 1 < self.n_nodes_arr[j]:
                    if overlap(i, j, left=False):
                        self.edges[0].push_back(self.shift[j - 1] + (((v + 1) << 1) + 1))
                    else:
                        self.edges[0].push_back(self.shift[j] + (v + 1))
                    self.edges[1].push_back(i)
                    self.edges[2].push_back(3 * j + 2)
                    self.topdown.push_back(self.n_edges)
                    self.n_edges += 1

                # left connection
                if v >= 1:
                    if overlap(i, j, left=True):
                        self.edges[0].push_back(self.shift[j - 1] + ((v - 1) << 1))
                    else:
                        self.edges[0].push_back(self.shift[j] + (v - 1))
                    self.edges[1].push_back(i)
                    self.edges[2].push_back(3 * j + 1)
                    self.topdown.push_back(self.n_edges)
                    self.n_edges += 1

                # Add bottom up connection
                if j > 0:
                    self.edges[0].push_back(i)
                    self.edges[1].push_back(shift + v)
                    self.edges[2].push_back(3 * j)
                    self.bottomup.push_back(self.n_edges)
                    self.n_edges += 1

                shift += self.n_nodes_arr[j]
                v >>= 1

    @property
    def number_of_nodes(self):
        return self.n_nodes

    @property
    def number_of_edges(self):
        return self.n_edges

    @property
    def number_of_levels(self):
        return self.n_lvl

    @property
    def is_triu(self):
        return self.triu

    def get_edges(self, v_shift=0):
        return np.asarray(self.edges[0]) + v_shift,\
            np.asarray(self.edges[1]) + v_shift,\
            np.asarray(self.edges[2])

    def td_eids(self, e_shift=0):
        "Return town down edges, (src, dst, eid)"
        return np.asarray(self.topdown) + e_shift

    def bu_eids(self, e_shift=0):
        "Return bottom up edges, (src, dst, eid)"
        return np.asarray(self.bottomup) + e_shift

    def leaf_ids(self, v_shift=0, start=0):
        return np.arange(v_shift + start, v_shift + self.n_nodes_arr[0])

    def internal_ids(self, v_shift=0):
        return np.arange(v_shift + self.n_nodes_arr[0], v_shift + self.n_nodes)

    def root_id(self, v_shift=0):
        return v_shift + self.n_nodes - 1

    def number_of_nodes_at_lvl(self, i):
        return self.n_nodes_arr[i]

cdef class FullyConnected:
    cdef size_t length, n_nodes, n_edges
    cdef bool triu
    cdef vector[DTYPE_t] topdown, bottomup
    cdef vector[DTYPE_t] edges[3]

    def __cinit__(self, size_t length, bool triu=False):
        self.n_nodes = length + 1
        self.length = length
        self.triu = triu
        self.build_graph()

    def __reduce__(self):
        return FullyConnected, (self.length, self.triu, self.extend)

    def build_graph(self):
        cdef size_t i, j
        for i in range(self.length):
            # topdown edges
            # self loop
            self.edges[0].push_back(i)
            self.edges[1].push_back(i)
            self.edges[2].push_back(0)
            self.topdown.push_back(self.n_edges)
            self.n_edges += 1
            
            for j in range(1, 129):
                # left
                if i >= j:
                    self.edges[0].push_back(i - j)
                    self.edges[1].push_back(i)
                    self.edges[2].push_back(2 * j - 1)
                    self.topdown.push_back(self.n_edges)
                    self.n_edges += 1
                # right
                if not self.triu and i + j < self.length:
                    self.edges[0].push_back(i + j)
                    self.edges[1].push_back(i)
                    self.edges[2].push_back(2 * j)
                    self.topdown.push_back(self.n_edges)
                    self.n_edges += 1
           
            # bottomup edges
            self.edges[0].push_back(i)
            self.edges[1].push_back(self.length)
            self.edges[2].push_back(0)
            self.bottomup.push_back(self.n_edges)
            self.n_edges += 1

    @property
    def number_of_nodes(self):
        return self.n_nodes

    @property
    def number_of_edges(self):
        return self.n_edges

    @property
    def is_triu(self):
        return self.triu

    def get_edges(self, v_shift=0):
        return np.asarray(self.edges[0]) + v_shift,\
            np.asarray(self.edges[1]) + v_shift,\
            np.asarray(self.edges[2])

    def td_eids(self, e_shift=0):
        "Return town down edges, (src, dst, eid)"
        return np.asarray(self.topdown) + e_shift

    def bu_eids(self, e_shift=0):
        "Return bottom up edges, (src, dst, eid)"
        return np.asarray(self.bottomup) + e_shift

    def root_id(self, v_shift=0):
        return v_shift + self.n_nodes - 1

    def leaf_ids(self, v_shift=0, start=0):
        return np.arange(v_shift + start, v_shift + self.length)

    def internal_ids(self, v_shift=0):
        return np.arange(v_shift + self.length, v_shift + self.n_nodes)

