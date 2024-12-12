### SPLICING PREDICTOR ###
# description:  PCPS predictor model
# author:       HPR

import math
import typing
from typing import Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATv2Conv, GATConv, GraphNorm, GCNConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, NoneType, OptPairTensor, OptTensor, Size, SparseTensor, torch_sparse
from torch_geometric.utils import add_self_loops, add_remaining_self_loops, is_torch_sparse_tensor, remove_self_loops, softmax
from torch_geometric.utils.sparse import set_sparse_value
from dig.sslgraph.method.contrastive.views_fn import EdgePerturbation, NodeAttrMask

# # FIXME
# batch = next(iter(train_loader))
# B = batch.x.shape[0]


# ----- energetic attention GNN -----
class EGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, c_max=1., heads=1,
                 bias=True, edge_dim=None, negative_slope=0.2, dropout=0.0):
        super(EGATConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c_max = c_max
        self.heads = heads
        self.bias = bias
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.node_dim = 0  # this is important! and probably a bug in pytorch geometric!

        # parameters from EGNN
        self.weight = Parameter(torch.eye(heads*out_channels) * math.sqrt(c_max))

        # parameters to compute attention coefficients
        self.att = Parameter(torch.empty(1, heads, out_channels))
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)
        
        # bias
        if bias:
            self.bias = Parameter(torch.empty(heads * out_channels))
        else:
            self.register_parameter('bias', None)
        
        # reset all parameters
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)
    
    def norm(edge_index, num_nodes, alpha):
        row, col = edge_index
        deg = scatter_add(alpha, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * alpha * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr=None, x_0=None, beta=0., residual_weight=0., size: Size = None, return_attention_weights=None):
        
        H, C = self.heads, self.out_channels
        x_input = x.view(-1,H,C)

        # linear transformation
        x = torch.matmul(x, self.weight).view(-1,H,C)

        # skip connections
        x = (1-residual_weight-beta)*x + residual_weight*x_input + beta*x_0.view(-1,H,C)

        # compute attention coefficients
        alpha = (x*self.att).sum(dim=-1)
        
        # NOTE: make sure that there are already self loops
        
        # update edges
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, size=size)

        # propagate
        x = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        # concatenation
        out = x.view(-1, self.heads * self.out_channels)
        
        # add bias
        if self.bias is not None:
            out = out + self.bias

        # return updated node embeddings and attention weights (or not)
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, torch.Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j, edge_attr=None, index=None, alpha_i=None, ptr=None, dim_size=None):
        # sum up edge-level attention coefficients for source and target nodes to "emulate" concatenation
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha += alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j
        
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')



class ClassifierV5(nn.Module):
    def __init__(self, num_layers, num_feats, num_classes, dim_hidden,
                 heads=2, c_min=0.2, c_max=1., beta=0.1, dropout=0.6, output_dropout=0.2, gamma=1.):
        super(ClassifierV5, self).__init__()
        
        self.num_layers = num_layers
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.dim_hidden = dim_hidden
        self.heads = heads

        self.c_min = c_min
        self.c_max = c_max
        self.beta = beta
        self.dropout = dropout
        self.output_dropout = output_dropout
        self.gamma = gamma

        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        
        # graph convolution layers
        for i in range(self.num_layers):
            c_max = self.c_max if i==0 else 1.
            self.layers_GCN.append(EGATConv(in_channels=self.dim_hidden, out_channels=self.dim_hidden, c_max=c_max,
                                            heads=self.heads, bias=True, edge_dim=None))
            self.layers_activation.append(nn.ReLU(self.dim_hidden))
        # in/output layers
        self.input_layer = torch.nn.Linear(self.num_feats, self.dim_hidden*self.heads)
        self.output_layer = torch.nn.Linear(self.dim_hidden*self.heads, self.num_classes)


    def forward(self, x, edge_index, edge_weight):

        # input layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        x = F.relu(x)
        
        # graph convolution
        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta
            x = self.layers_GCN[i](x, edge_index, edge_attr=edge_weight, x_0=original_x, 
                                   beta=self.beta, residual_weight=residual_weight)
            # self.layers_GCN[i].weight = self.layers_GCN[i].lin.weight
            x = self.layers_activation[i](x)

        # output layer
        x = F.dropout(x, p=self.output_dropout, training=self.training)
        x = self.output_layer(x)
        y = F.log_softmax(x, dim=1)

        # loss constraint
        weight_standard = torch.eye(self.dim_hidden)
        weight_first_layer = weight_standard * math.sqrt(self.c_max)
        loss_orthogonal = 0.
        loss_orthogonal += torch.norm(self.layers_GCN[0].weight - weight_first_layer)
        for i in range(1, self.num_layers):
            loss_orthogonal += torch.norm(self.layers_GCN[i].weight - weight_standard)
        lconstr = self.gamma * loss_orthogonal

        # output
        return y,lconstr
    
    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier.ckpt'))
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier_best.ckpt'))




# ----- energetic GNN -----
# implementation from Zhou et al, NeurIPS 2021 (https://proceedings.neurips.cc/paper_files/paper/2021/file/b6417f112bd27848533e54885b66c288-Paper.pdf)
# https://github.com/Kaixiong-Zhou/EGNN/tree/main

class EGNNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, c_max=1.,
                 improved=False, cached=False, bias=True):
        super(EGNNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.weight = Parameter(torch.eye(in_channels) * math.sqrt(c_max))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        # fill_value = 1 if not improved else 2
        fill_value = 0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, norm

    def forward(self, x, edge_index, x_0 = None, beta=0.,
                residual_weight=0., edge_weight=None):
        """"""
        x_input = x
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        x = self.propagate(edge_index, x=x, norm=norm)
        x = (1-residual_weight-beta) * x + residual_weight * x_input \
            + beta * x_0
        x = torch.matmul(x, self.weight)
        return x
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc, bias):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(nc,))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, bias)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias


class ClassifierV4(nn.Module):
    def __init__(self, num_layers, num_feats, num_classes, dim_hidden,
                 transductive=False, c_min=0.2, c_max=1., beta=0.1,
                 bias_SReLU=-10, dropout=0.6, output_dropout=0.2,
                 gamma=1.):
        super(ClassifierV4, self).__init__()
        
        self.num_layers = num_layers
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.dim_hidden = dim_hidden

        # self.encoder = encoder
        # self.aug = aug

        self.cached = transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.c_min = c_min
        self.c_max = c_max
        self.beta = beta
        self.gamma = gamma

        self.bias_SReLU = bias_SReLU
        self.dropout = dropout
        self.output_dropout = output_dropout

        for i in range(self.num_layers):
            c_max = self.c_max if i==0 else 1.
            self.layers_GCN.append(EGNNConv(self.dim_hidden, self.dim_hidden,
                                            c_max=c_max, cached=self.cached, bias=False))
            self.layers_activation.append(nn.ReLU(self.dim_hidden))

        self.input_layer = torch.nn.Linear(self.num_feats, self.dim_hidden)
        self.output_layer = torch.nn.Linear(self.dim_hidden, self.num_classes)


    def forward(self, x, edge_index, edge_weight):

        # if self.aug:
        #     x = self.encoder(x,edge_index)

        # else:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        x = F.relu(x)
        
        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta
            x = self.layers_GCN[i](x, edge_index, original_x,
                                   edge_weight=edge_weight,
                                   beta=self.beta, residual_weight=residual_weight)
            x = self.layers_activation[i](x)


        x = F.dropout(x, p=self.output_dropout, training=self.training)
        x = self.output_layer(x)
        y = F.log_softmax(x, dim=1)

        weight_standard = torch.eye(self.dim_hidden)
        weight_first_layer = weight_standard * math.sqrt(self.c_max)
        loss_orthogonal = 0.
        loss_orthogonal += torch.norm(self.layers_GCN[0].weight - weight_first_layer)
        for i in range(1, self.num_layers):
            loss_orthogonal += torch.norm(self.layers_GCN[i].weight - weight_standard)
        lconstr = self.gamma * loss_orthogonal

        return y,lconstr
    
    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier.ckpt'))
        # torch.save(ckpt, f = f'results/{OUTNAME}/classifier_epoch{EPOCH}.ckpt')
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier_best.ckpt'))



# ---------------
# ----- OLD -----
# ---------------

class EncoderV4(nn.Module):
    def __init__(self, num_layers, num_feats, num_classes, dim_hidden,
                 transductive=True, c_min=0.2, c_max=1., beta=0.1,
                 bias_SReLU=-10, dropout=0.6, output_dropout=0.2):

        super(EncoderV4, self).__init__()
        self.num_layers = num_layers
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.dim_hidden = dim_hidden

        self.cached = transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.c_min = c_min
        self.c_max = c_max
        self.beta = beta

        self.bias_SReLU = bias_SReLU
        self.dropout = dropout
        self.output_dropout = output_dropout

        self.reg_params = []
        for i in range(self.num_layers):
            c_max = self.c_max if i==0 else 1.
            self.layers_GCN.append(EGNNConv(self.dim_hidden, self.dim_hidden,
                                            c_max=c_max, cached=self.cached, bias=False))
            self.layers_activation.append(SReLU(self.dim_hidden, self.bias_SReLU))
            self.reg_params.append(self.layers_GCN[-1].weight)
        
        self.input_layer = torch.nn.Linear(self.num_feats, self.dim_hidden)
        self.srelu_params = list(self.layers_activation[:-1].parameters())

    def forward(self, x, edge_index):
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        x = F.relu(x)

        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta
            x = self.layers_GCN[i](x, edge_index, original_x,
                                   beta=self.beta, residual_weight=residual_weight)
            x = self.layers_activation[i](x)
        return x
    
    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/encoder.ckpt'))
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/encoder_best.ckpt'))





# ----- encoder -----
class Encoder(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim, heads, norm=True):
        super(Encoder, self).__init__()
        
        # layers
        self.convs = nn.ModuleList()
        # self.acts = nn.ModuleList()
        if norm:
            self.norms = nn.ModuleList()

        n_layers = len(hidden_dim)
        self.n_layers = n_layers
        self.norm = norm

        # a = nn.LeakyReLU()
        for i in range(n_layers):
            start_dim = hidden_dim[i-1]*heads if i else node_dim
            conv = GATv2Conv(start_dim,hidden_dim[i]*heads,heads=heads,concat=False,add_self_loops=False)
            self.convs.append(conv)
            # self.acts.append(a)
            if norm:
                norm = GraphNorm(hidden_dim[i]*heads)
                self.norms.append(norm)
        

    def forward(self, x, edge_index):
        
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            # x = self.acts[i](x)
            if self.norm:
                x = self.norms[i](x)
        
        return x
    

    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/encoder.ckpt'))
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/encoder_best.ckpt'))



class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()

        self.lin1 = nn.Linear(in_dim,out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.lin2 = nn.Linear(out_dim,out_dim)
        self.lin3 = nn.Linear(out_dim,out_dim)

    def forward(self, x):

        # z = F.relu(self.bn(self.lin1(x)))
        z = self.lin1(x)
        # z = self.lin2(z)
        # z = self.lin3(z)
        # stopgrad!
        return z.detach()

# ----- data augmentation
def augmentation_fn(batch):
    # edge perturbation
    batch = EdgePerturbation(add=False, drop=True, ratio=0.5)(batch)
    # feature masking
    batch = NodeAttrMask(mask_ratio=0.1)(batch)

    return batch



# ----- classifier -----
class Classifier(torch.nn.Module):
    def __init__(self, encoder, aug, encoder_dim, hidden_dim, heads, freeze_encoder=False, norm=True, act=False):
        super(Classifier, self).__init__()

        n_layers = len(hidden_dim)
        self.n_layers = n_layers
        self.norm = norm
        self.act = act

        self.encoder = encoder
        self.aug = aug
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder and self.aug:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # layers
        self.convs = nn.ModuleList()
        if act:
            self.acts = nn.ModuleList()
            a = nn.LeakyReLU()
        if norm:
            self.norms = nn.ModuleList()

        for i in range(n_layers):
            start_dim = hidden_dim[i-1]*heads if i else encoder_dim
            conv = GATv2Conv(start_dim,hidden_dim[i]*heads,heads=heads,concat=False,add_self_loops=True)
            # conv = GATConv(start_dim,hidden_dim[i]*heads,heads=heads,concat=False,add_self_loops=False)
            self.convs.append(conv)
            if act:
                self.acts.append(a)
            if norm:
                norm = GraphNorm(hidden_dim[i]*heads)
                self.norms.append(norm)

        self.lin1 = nn.Linear(hidden_dim[-1]*heads,4)
        self.drop = nn.Dropout(0.5)


    def forward(self, x, edge_index):

        if self.aug:
            emb = self.encoder(x, edge_index)
        else:
            emb = x

        for i in range(self.n_layers):
            emb = self.convs[i](emb, edge_index)
            if self.act:
                emb = self.acts[i](emb)
            if self.norm:
                emb = self.norms[i](emb)

        emb = self.drop(self.lin1(emb))
        y = F.log_softmax(emb, dim=1)
        return y
    

    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier.ckpt'))
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier_best.ckpt'))



# ----- classifier with skip connections -----
class ClassifierV2(torch.nn.Module):
    def __init__(self, encoder, aug, self_loops, encoder_dim, hidden_dim, heads, freeze_encoder=False, norm=True):
        super(ClassifierV2, self).__init__()

        n_layers = len(hidden_dim)
        self.n_layers = n_layers
        self.norm = norm
        self.self_loops = self_loops

        self.encoder = encoder
        self.aug = aug
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder and self.aug:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # layers
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if norm:
            self.norms = nn.ModuleList()

        for i in range(n_layers):
            start_dim = hidden_dim[i-1]*heads if i else encoder_dim
            conv = GATv2Conv(start_dim,hidden_dim[i],heads=heads,concat=True,add_self_loops=self_loops)
            lin = torch.nn.Linear(start_dim,hidden_dim[i]*heads)
            self.convs.append(conv)
            self.lins.append(lin)
            if norm:
                norm = GraphNorm(hidden_dim[i]*heads)
                self.norms.append(norm)

        self.conv_final = GATv2Conv(hidden_dim[-1]*heads,5,heads=heads,concat=False,add_self_loops=self_loops)
        self.lin_final = torch.nn.Linear(hidden_dim[-1]*heads, 5)

    def forward(self, x, edge_index):

        if self.aug:
            emb = self.encoder(x, edge_index)
        else:
            emb = x

        for i in range(self.n_layers):
            cpart = self.convs[i](emb, edge_index)
            lpart = self.lins[i](emb)
            if self.norm:
                cpart = self.norms[i](cpart)
            emb = F.elu(cpart+lpart)

        cpart = self.conv_final(emb, edge_index)
        lpart = self.lin_final(emb)
        y = F.log_softmax(cpart+lpart, dim=1)
        # y = F.softmax(cpart+lpart, dim=1)
        return y
    

    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier.ckpt'))
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier_best.ckpt'))



# ----- gradient gating -----
# implementation from Rusch et al, ICLR 2023 (https://arxiv.org/pdf/2210.00513.pdf)
# https://github.com/tk-rusch/gradientgating/blob/main/node_regression/models.py

class G2(nn.Module):
    def __init__(self, conv, norm, nhid, nheads=4, p=2.):
        super(G2, self).__init__()
        self.conv = conv
        self.nheads = nheads
        self.p = p
        self.norm = norm

        if norm:
            # self.normop = LayerNorm(nhid*nheads, mode="node")
            self.normop = GraphNorm(nhid*nheads)

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        
        X = self.conv(X, edge_index)
        if self.norm:
            X = self.normop(X)
        
        X = F.elu(X).view(n_nodes, -1, int(self.nheads)).mean(dim=-1)
        gg = torch.tanh(scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0, dim_size=X.size(0), reduce='mean'))

        return gg



class ClassifierV3(nn.Module):
    def __init__(self, nfeat, nhid, nlayers,
                 self_loops=True, nheads=4, nclass=5, p=2.,
                 drop_in=0, drop=0, use_gg_conv=True, norm=True, encoder=None, aug=False):
        super(ClassifierV3, self).__init__()
        
        self.enc = nn.Linear(nfeat, nhid)
        self.lin = nn.Linear(nhid,nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nheads = nheads
        self.nlayers = nlayers
        self.norm = norm
        
        # GATConv or GATv2Conv?
        self.conv = GATv2Conv(nhid,nhid,heads=nheads,add_self_loops=self_loops,concat=True)
        if use_gg_conv == True:
            self.conv_gg = GATv2Conv(nhid,nhid,heads=nheads,add_self_loops=self_loops,concat=True)

        if use_gg_conv == True:
            self.G2 = G2(self.conv_gg,norm,nhid,nheads,p)
        else:
            self.G2 = G2(self.conv,norm,nhid,nheads,p)
        
        if norm:
            # self.normop = LayerNorm(nhid*nheads, mode="node")
            self.normop = GraphNorm(nhid*nheads)

        self.encoder = encoder
        self.aug = aug

    def forward(self, x, edge_index):
        n_nodes = x.size(0)

        if self.aug:
            X = self.encoder(x, edge_index)
        
        else:
            X = F.dropout(x, self.drop_in, training=self.training)
            X = torch.relu(self.enc(X))

            for i in range(self.nlayers):   
                X_ = self.conv(X, edge_index)
                if self.norm:
                    X_ = self.normop(X_)
                
                X_ = F.elu(X_).view(n_nodes, -1, int(self.nheads)).mean(dim=-1)
                tau = self.G2(X, edge_index)
                X = (1 - tau) * X + tau * X_
            
            X = F.dropout(X, self.drop, training=self.training)

        # return F.log_softmax(F.relu(self.dec(X)), dim=1)
        return F.log_softmax(self.dec(X), dim=1)
    
    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier.ckpt'))
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/classifier_best.ckpt'))



class EncoderV3(nn.Module):
    def __init__(self, nfeat, nhid, nlayers,
                 self_loops=True, nheads=4, nclass=5, p=2.,
                 drop_in=0, drop=0, use_gg_conv=True, norm=True):
        super(EncoderV3, self).__init__()
        
        self.enc = nn.Linear(nfeat, nhid)
        self.lin = nn.Linear(nhid,nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nheads = nheads
        self.nlayers = nlayers
        self.norm = norm
        
        # GATConv or GATv2Conv?
        self.conv = GATv2Conv(nhid,nhid,heads=nheads,add_self_loops=self_loops,concat=True)
        if use_gg_conv == True:
            self.conv_gg = GATv2Conv(nhid,nhid,heads=nheads,add_self_loops=self_loops,concat=True)

        if use_gg_conv == True:
            self.G2 = G2(self.conv_gg,norm,nhid,nheads,p)
        else:
            self.G2 = G2(self.conv,norm,nhid,nheads,p)
        
        if norm:
            # self.normop = LayerNorm(nhid*nheads, mode="node")
            self.normop = GraphNorm(nhid*nheads)

    def forward(self, x, edge_index):
        n_nodes = x.size(0)
        X = F.dropout(x, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        for i in range(self.nlayers):
            
            X_ = self.conv(X, edge_index)
            if self.norm:
                X_ = self.normop(X_)
            
            X_ = F.elu(X_).view(n_nodes, -1, int(self.nheads)).mean(dim=-1)
            tau = self.G2(X, edge_index)
            X = (1 - tau) * X + tau * X_
            
        X = F.dropout(X, self.drop, training=self.training)
        return X
    
    def save_checkpoint(self, OUTNAME, EPOCH, optimizer, is_best):
        ckpt = {}
        ckpt["epoch"] = EPOCH
        ckpt["model_state"] = self.state_dict()
        ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, f = str('results/'+OUTNAME+'/encoder.ckpt'))
        if is_best:
            torch.save(ckpt, f = str('results/'+OUTNAME+'/encoder_best.ckpt'))
