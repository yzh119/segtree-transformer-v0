import torch as th
from graphop import *
from torch.autograd import Function

def to_contiguous(args):
    def wrapper(func):
        return func(*[arg.contiguous() if th.is_tensor(arg) else arg for arg in args])
    return wrapper

class SparseSoftmax(Function):
    @staticmethod
    def forward(ctx, ptr, eid, x):
        y = sparse_softmax_forward(ptr, eid, x)
        ctx.save_for_backward(ptr, eid, y)
        return y

    @staticmethod
    def backward(ctx, dy):
        ptr, eid, y = ctx.saved_tensors
        return None, None, sparse_softmax_backward(ptr, eid, y, dy)

class NodeMulEdge(Function):
    @staticmethod
    def forward(ctx, indptr, eid, A, B):
        ctx.save_for_backward(indptr, eid, A, B)
        return node_mul_edge_forward(indptr, eid, A, B)

    @staticmethod
    def backward(ctx, grad):
        indptr, eid, A, B = ctx.saved_tensors
        dA, dB = node_mul_edge_backward(indptr, eid, A, B, grad)
        return None, None, dA, dB

class MaskedMMCSR(Function):
    @staticmethod
    def forward(ctx, ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B):
        ctx.save_for_backward(ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B)
        return maskedmm_csr_forward(ptr_r, eid_r, nid_r, A, B)

    @staticmethod
    def backward(ctx, grad):
        ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B = ctx.saved_tensors
        dA, dB = maskedmm_csr_backward(ptr_r, eid_r, nid_r, ptr_c, eid_c, nid_c, A, B, grad)
        return None, None, None, None, None, None, dA, dB

class VectorSPMM(Function):
    @staticmethod
    def forward(ctx, ptr, eid, nid, ptr_t, eid_t, nid_t, edata, x):
        y = vector_spmm_forward(ptr, eid, nid, edata, x)
        ctx.save_for_backward(ptr, eid, nid, ptr_t, eid_t, nid_t, edata, x)
        return y

    @staticmethod
    def backward(ctx, dy):
        ptr, eid, nid, ptr_t, eid_t, nid_t, edata, x = ctx.saved_tensors
        dedata, dx = vector_spmm_backward(ptr, eid, nid, ptr_t, eid_t, nid_t, edata, dy, x)
        return None, None, None, None, None, None, dedata, dx
