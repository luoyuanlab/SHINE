import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.attention as attention
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import numpy as np, scipy.sparse as sp


def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.
    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
    For example, say I have a "matrix" with dimensions `(batch_size, num_queries, num_words,
    embedding_dim)`.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:
        - `(batch_size, num_queries, num_words)` (distribution over words for each query)
        - `(batch_size, num_documents, num_queries, num_words)` (distribution over words in a
          query for each document)
    are valid input "vectors", producing tensors of shape:
    `(batch_size, num_queries, embedding_dim)` and
    `(batch_size, num_documents, num_queries, embedding_dim)` respectively.
    """
    
    
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)

def masked_sum(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    **
    Adapted from AllenNLP's masked mean: 
    https://github.com/allenai/allennlp/blob/90e98e56c46bc466d4ad7712bab93566afe5d1d0/allennlp/nn/util.py
    ** 
    To calculate mean along certain dimensions on masked values
    
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    
    replaced_vector = vector.masked_fill(~mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    return value_sum 

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft)) 
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft)) 
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self): 
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def reset_parameters_xavier(self): 
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class HGNN_sg_attn_simple(nn.Module):
    
    def __init__(self, edim, ddim, atype='additive'):
        super(HGNN_sg_attn, self).__init__()
        if atype == 'additive':
            self.attention = attention.AdditiveAttention(edim, ddim)
        elif atype == 'multiplicative':
            self.attention = attention.MultiplicativeAttention(edim, ddim)
        else:
            exit(f'unrecognized attention type {atype}')


    def forward(self, x, sgs):
        xsize = list(x.size())
        bsize = sgs.shape[0]
        b_attn = torch.matmul(sgs, x)
        
        
        
        
        y = self.attention(b_attn, x)
        
        return y

class HGNN_sg_attn_Allen(nn.Module):
    
    def __init__(self, vdim, mdim, atype='additive'):
        super(HGNN_sg_attn, self).__init__()
        if atype == 'additive':
            self.attention = attention.AdditiveAttention(vdim, mdim)
        elif atype == 'dotprod':
            self.attention = attention.DotProductAttention()
        else:
            exit(f'unrecognized attention type {atype}')


    def forward(self, x, sgs):
        xsize = list(x.size())
        bsize = sgs.shape[0]
        b_attn = torch.matmul(sgs, x)
        attn_wts = self.attention(b_attn, x.unsqueeze(0).expand(bsize, *xsize))
        x = torch.matmul(attn_wts, x)
        return x
    
class HGNN_sg_attn(nn.Module):
    
    def __init__(self, vdim, mdim, atype='additive'):
        super(HGNN_sg_attn, self).__init__()
        self.attn_vector = torch.nn.Parameter(torch.zeros((vdim,1), dtype=torch.float), requires_grad=True)   
        
        stdv = 1. / math.sqrt(vdim) 
        self.attn_vector.data.uniform_(-stdv, stdv)


    def forward(self, x, sgs):
        xsize = list(x.size())
        bsize = sgs.shape[0]
        attn_wts = torch.matmul(x, self.attn_vector) 
        attn_wts = attn_wts.squeeze().unsqueeze(0).expand(bsize, xsize[0]) 
        x = torch.matmul(sgs*attn_wts, x)
        return x


class HGNN_sg_attn_multiplicative(nn.Module):
    
    def __init__(self, vdim, mdim, atype='additive'):
        super(HGNN_sg_attn, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(mdim, vdim), requires_grad=True)
        nn.init.xavier_uniform_(self.W)


    def forward(self, x, sgs):
        x = sgs.matmul(x).matmul(self.W)
        return x
    
class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x


class HGAT_sparse(nn.Module):

    def __init__(self, in_ch_n, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
        super(HGAT_sparse, self).__init__()
        self.e_dropout = nn.Dropout(dropout)
        self.in_ch_n = in_ch_n
        self.out_ch = out_ch
        self.alpha = alpha
        self.concat = concat
        
        self.transfer = transfer

        if self.transfer:
            self.wt = Parameter(torch.Tensor(self.in_ch_n, self.out_ch))
        else:
            self.register_parameter('wt', None)

        
        
        

        if bias:
            self.bias = Parameter(torch.Tensor(1, self.out_ch))
        else:
            self.register_parameter('bias', None)

        
        
        
        
        
        self.coarsen = coarsen

        self.reset_parameters()

    def reset_parameters(self): 
        stdv = 1. / math.sqrt(self.out_ch)
        if self.wt is not None:
            self.wt.data.uniform_(-stdv, stdv)

        
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        
        
        
        

    def reset_parameters_xavier(self): 
        if self.wt is not None:
            nn.init.xavier_uniform_(self.wt)

        
        
        
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias) 

        
        
        

    def std_scale(self, x):
        xstd = x.std(1, unbiased=False, keepdim=True)
        xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
        x = (x - x.mean(1, keepdim=True)) / xstd
        return x

    def forward(self, x, xe, pair, a, val=None, e_degs=None, n_degs=None): 
        
        
        if self.transfer:
            x = x.mm(self.wt) 
            xe = xe.mm(self.wt) 
            
            if self.bias is not None:
                
                x = x + self.bias
                xe = xe + self.bias

        
        

        n_edge = xe.shape[0] 
        n_node = x.shape[0] 
        
        
        if val is None:
            pair_h = xe[ pair[0] ] * x[ pair[1] ] 
        else:
            pair_h = xe[ pair[0] ] * x[ pair[1] ] * val
            
        if e_degs is not None:
            pair_h /= e_degs[ pair[0] ].sqrt().unsqueeze(-1) 
        if n_degs is not None:
            pair_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1) 
        pair_e = torch.mm(pair_h, a).squeeze() 
        
        
        

        
        

        e = torch.zeros(n_edge, n_node, device=pair.device)
        e[pair[0], pair[1]] = torch.exp(pair_e)        
        e = torch.log(1e-10 + self.e_dropout(e))
        

        
        attention_edge = F.softmax(e, dim=1) 

        xe_out = torch.mm(attention_edge, x) 
        
        

        
        
        
        
        


        
        
        
        

        attention_node = F.softmax(e.transpose(0,1), dim=1)

        x = torch.mm(attention_node, xe) 


        if self.concat:
            x = F.elu(x)
            xe_out = F.elu(xe_out)
        else:
            x = F.relu(x)
            xe_out = F.relu(xe_out)

        
        if self.coarsen:
            return x, xe_out, torch.exp(e.T) 
        else:
            return x, xe_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')'




class HyperGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, reapproximate=True, cuda=True):
        
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()
        


    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)



    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else: A = structure

        if self.cuda: A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'



class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2



def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators

    arguments:
    V: number of vertices
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without

    A: adjacency matrix of the graph approximation
    returns: 
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """
    
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])

    for k in E.keys():
        hyperedge = list(E[k])
        
        p = np.dot(X[hyperedge], rv)   
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        
        c = 2*len(hyperedge) - 3    
        if m:
            
            
            edges.extend([[Se, Ie], [Ie, Se]])
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    
    
    return adjacency(edges, weights, V)



def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights



def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A



def symnormalise(M):
    """
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    
    
    return (DHI.dot(M)).dot(DHI) 



def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)
