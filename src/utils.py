### SPLICING PREDICTOR ###
# description:  utility functions
# author:       HPR

import os
import random
import math
import polars as pl
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Batch
import torch_geometric.transforms as T
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()]) 


# ----- fix seed -----
def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


# ----- custom data loader function: v4 -----
class PseudoLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, k, mask, directed=False, transform=transform, **kwargs,):
        
        self.graph = graph
        self.k = k  # number of total hops
        self.mask = mask
        self.directed = directed
        self.transform = transform
        self.pseudonodes = self.graph.node_id[self.mask & self.graph.pseudo_mask]
        
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # iterate number of batches
        super().__init__(self.pseudonodes.view(-1).tolist(), collate_fn=self.collate_fn, **kwargs)


    def collate_fn(self, batch):        
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        # get educts of sampled nodes
        nodes = k_hop_subgraph(batch, num_hops=self.k, edge_index=self.graph.edge_index, flow='source_to_target', directed=self.directed)[0]
        smpl = torch.cat([batch,nodes]).unique().flatten()

        # combine all
        batchedgraph = self.graph.subgraph(smpl)
        # remove isolated nodes
        batchedgraph = self.transform(batchedgraph)
        
        return batchedgraph


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


    

# tmp: manual version for testing
def pseudo_loader(graph, batch_size, k=1):
    mask = graph.train_mask
    directed = False
    # nodes = graph.node_id[mask]
    pseudonodes = graph.node_id[mask & graph.pseudo_mask]

    batch = pseudonodes[56:57]
    nodes = k_hop_subgraph(batch, num_hops=k, edge_index=graph.edge_index, flow='source_to_target', directed=directed)[0]

    
    


# ----- custom data loader function: v3 -----
class EductLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, k, mask, directed=False, transform=transform, **kwargs,):
        
        self.graph = graph
        self.k = k  # number of total hops
        self.mask = mask
        self.directed = directed
        self.transform = transform
        self.nodes = self.graph.node_id[self.mask]
        
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # iterate number of batches
        super().__init__(self.nodes.view(-1).tolist(), collate_fn=self.collate_fn, **kwargs)
    
    def collate_fn(self, batch):
        
        # print(len(batch))
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        # get educts of sampled nodes
        educts = k_hop_subgraph(batch, num_hops=self.k, edge_index=self.graph.edge_index, flow='target_to_source', directed=self.directed)[0]

        # sample from PCP and pseudonode pool
        n = math.floor(self.graph.psp_mask[educts].sum()*0.05)
        n = n if n > 0 else 1
        if self.graph.pcp_mask[educts].sum() >= n:
            pcp = educts[self.graph.pcp_mask[educts]][torch.randperm(n)]
        else:
            pcp = educts[self.graph.pcp_mask[educts]]
        
        if self.graph.pseudo_mask[educts].sum() >= n:
            pseudo = educts[self.graph.pseudo_mask[educts]][torch.randperm(n)]
        else:
            pseudo = educts[self.graph.pseudo_mask[educts]]
        
        smpl = torch.cat([batch, educts[self.graph.psp_mask[educts]], pcp, pseudo]).unique().flatten()

        # combine all
        batchedgraph = self.graph.subgraph(smpl)
        # remove isolated nodes
        batchedgraph = self.transform(batchedgraph)
        
        return batchedgraph

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# tmp: manual version for testing
def educt_loader(graph, batch_size, k=1):
    mask = graph.train_mask
    nodes = graph.node_id[mask]
    directed = False

    # TODO: change!
    idx = torch.randperm(n = nodes.shape[0])
    idxx = nodes[idx[:batch_size]]
    
    educts = k_hop_subgraph(idxx, num_hops=k, edge_index=graph.edge_index, flow='target_to_source', directed=directed)[0]

    n = math.floor(graph.psp_mask[educts].sum()*0.25)
    pcp = smpl[graph.pcp_mask[educts]][torch.randperm(n)]
    pseudo = smpl[graph.pseudo_mask[educts]][torch.randperm(n)]

    smpl = torch.cat([idxx, educts[graph.psp_mask[educts]], pcp, pseudo]).unique().flatten()

    # combine all
    batchedgraph =  graph.subgraph(smpl.flatten())
    # remove isolated nodes
    batchedgraph = transform(batchedgraph)

    batchedgraph.x.shape[0]

    return batchedgraph

    

# ----- custom data loader function: reaction loader -----
class ReactionLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, k, num_pcp_nodes, num_psp_nodes, num_pseudo_nodes,
                 mask, transform=transform, **kwargs,):
        
        self.graph = graph
        self.k = k  # number of total hops = number of layers in GAT and classifier
        self.num_pcp_nodes = num_pcp_nodes
        self.num_psp_nodes = num_psp_nodes
        self.num_pseudo_nodes = num_pseudo_nodes
        self.transform = transform
        self.mask = mask

        self.pseudonodes = self.graph.node_id[self.graph.aanode_mask & self.mask]
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        # NOTE: just a test
        iterator = self.pseudonodes
        # num_iter = math.ceil(len(iterator)/50)
        # id = torch.randperm(len(iterator))[:num_iter]
        # iterator = iterator[id]

        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(self, idx):
        out = self.collate_fn(idx)
        return out
    
    def collate_fn(self, idx):
        
        # # subgraph with all PCPs and PSPs that share the same pseudonode
        # pseudo_anchor = k_hop_subgraph(idx, num_hops=1, edge_index=self.graph.edge_index, flow="source_to_target", directed=True)[0]
        # subgraph with all PCPs and PSPs that share the same pseudonode
        aa_anchor = k_hop_subgraph(idx, num_hops=self.k, edge_index= self.graph.edge_index,
                                   flow="source_to_target", directed=True)[0]

        # # subgraph with PCP nodes and substrate nodes
        # subID = self.graph.node_info.filter(pl.col('index') == idx)['substrateID'].unique()
        # substrate_node = torch.tensor(self.graph.node_info.filter(pl.col('substrateID') == subID).filter(pl.col('subnode_mask') == True)['index'], dtype=torch.long)
        # # NOTE: changed number of hops from k-1 to k!
        # substrate_anchor = k_hop_subgraph(torch.cat([pseudo_anchor,substrate_node]), num_hops=self.k, edge_index=self.graph.edge_index, flow="target_to_source")[0]
        
        aanode_subset = aa_anchor[self.graph.aanode_mask[aa_anchor]]
        pcp_subset = aa_anchor[self.graph.pcp_mask[aa_anchor]]
        psp_subset = aa_anchor[self.graph.psp_mask[aa_anchor]]
        pseudo_subset = aa_anchor[self.graph.pseudo_mask[aa_anchor]]

        # sample from the PCP graph to make it more shallow
        if self.num_pcp_nodes != -1:
            kk = torch.randperm(pcp_subset.shape[0])[:self.num_pcp_nodes]
            # pcp_subset_obs = pcp_subset[self.graph.y[pcp_subset] == 3]
            # final_pcp_subset = torch.cat([pcp_subset[kk],pcp_subset_obs])
            final_pcp_subset = pcp_subset[kk]
        else:
            final_pcp_subset = pcp_subset
        
        if self.num_psp_nodes != -1:
            jj = torch.randperm(psp_subset.shape[0])[:self.num_psp_nodes]
            # psp_subset_obs = psp_subset[self.graph.y[psp_subset] == 4]
            # final_psp_subset = torch.cat([psp_subset[jj],psp_subset_obs])
            final_psp_subset = psp_subset[jj]
        else:
            final_psp_subset = psp_subset
        
        if self.num_pseudo_nodes != -1:
            ii = torch.randperm(pseudo_subset.shape[0])[:self.num_pseudo_nodes]
            # pseudo_subset_obs = pseudo_subset[torch.isin(self.graph.y[pseudo_subset], torch.tensor((1,2)))]
            # final_pseudo_subset = torch.cat([pseudo_subset[ii],pseudo_subset_obs])
            final_pseudo_subset = pseudo_subset[ii]
        else:
            final_pseudo_subset = pseudo_subset

        # # combine all
        # batchedgraph = self.graph.subgraph(torch.cat([substrate_node,final_pcp_subset,final_psp_subset,final_pseudo_subset]).flatten())
        batchedgraph = self.graph.subgraph(torch.cat([aanode_subset,final_pcp_subset,final_psp_subset,final_pseudo_subset]).flatten())
        # remove isolated nodes
        batchedgraph = transform(batchedgraph)

        return batchedgraph
        
    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'




# tmp: manual version for testing
def reaction_loader(graph,k,num_pcp_nodes,num_psp_nodes,num_pseudo_nodes,transform,mask,upsample,num_upsamples):
    mask = graph.train_mask
    pseodonodes = graph.node_id[graph.aanode_mask & mask]
    iterator = pseodonodes
    idx = iterator[1:2]

    # subgraph with all PCPs and PSPs that share the same pseudonode
    aa_anchor = k_hop_subgraph(idx, num_hops=k, edge_index= graph.edge_index, flow="source_to_target", directed=True)[0]
    
    graph.aanode_mask[aa_anchor].sum()
    graph.psp_mask[aa_anchor].sum()
    graph.pseudo_mask[aa_anchor].sum()
    graph.pcp_mask[aa_anchor].sum()
    graph.subnode_mask[aa_anchor].sum()

    aanode_subset = aa_anchor[ graph.aanode_mask[aa_anchor]]
    pcp_subset = aa_anchor[ graph.pcp_mask[aa_anchor]]
    psp_subset = aa_anchor[ graph.psp_mask[aa_anchor]]
    pseudo_subset = aa_anchor[ graph.pseudo_mask[aa_anchor]]

    # sample from the PCP graph to make it more shallow
    if  num_pcp_nodes != -1:
        kk = torch.randperm(pcp_subset.shape[0])[: num_pcp_nodes]
        # pcp_subset_obs = pcp_subset[ graph.y[pcp_subset] == 3]
        # final_pcp_subset = torch.cat([pcp_subset[kk],pcp_subset_obs])
        final_pcp_subset = pcp_subset[kk]
    else:
        final_pcp_subset = pcp_subset
    
    if  num_psp_nodes != -1:
        jj = torch.randperm(psp_subset.shape[0])[: num_psp_nodes]
        # psp_subset_obs = psp_subset[ graph.y[psp_subset] == 4]
        # final_psp_subset = torch.cat([psp_subset[jj],psp_subset_obs])
        final_psp_subset = psp_subset[jj]
    else:
        final_psp_subset = psp_subset
    
    if  num_pseudo_nodes != -1:
        ii = torch.randperm(pseudo_subset.shape[0])[: num_pseudo_nodes]
        # pseudo_subset_obs = pseudo_subset[torch.isin( graph.y[pseudo_subset], torch.tensor((1,2)))]
        # final_pseudo_subset = torch.cat([pseudo_subset[ii],pseudo_subset_obs])
        final_pseudo_subset = pseudo_subset[ii]
    else:
        final_pseudo_subset = pseudo_subset

    # combine all
    batchedgraph =  graph.subgraph(torch.cat([aanode_subset,final_pcp_subset,final_psp_subset,final_pseudo_subset]).flatten())
    # remove isolated nodes
    batchedgraph = transform(batchedgraph)

    return batchedgraph


# ----- custom data loader function: v2 -----
class HierarchicalLoader(torch.utils.data.DataLoader):
    def __init__(self, graph, k, mask, transform=transform, **kwargs,):
        
        self.graph = graph
        self.k = k  # number of total hops
        self.transform = transform
        self.mask = mask

        self.pseudonodes = self.graph.node_id[self.graph.pseudo_mask & self.mask]
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        iterator = self.pseudonodes
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(self, idx):
        out = self.collate_fn(idx)
        return out
    
    def collate_fn(self, idx):
        
        # get k-hop neighbours of pseudonodes
        pseudo_anchor_down = k_hop_subgraph(idx, num_hops=int(self.k/2)+1, edge_index= self.graph.edge_index, flow="target_to_source", directed=True)[0]
        pseudo_anchor_up = k_hop_subgraph(idx, num_hops=int(self.k/2)+1, edge_index= self.graph.edge_index, flow="source_to_target", directed=True)[0]

        # get all adjacent pseudonodes for psps
        psp = self.graph.node_id[pseudo_anchor_down]
        psp_anchor_up = k_hop_subgraph(psp[self.graph.psp_mask[psp]], num_hops=1, edge_index= self.graph.edge_index, flow="source_to_target", directed=True)[0]
        pseudo_set = psp_anchor_up[self.graph.pseudo_mask[psp_anchor_up]]
        
        # combine all
        batchedgraph = self.graph.subgraph(torch.cat([pseudo_set,pseudo_anchor_down,pseudo_anchor_up]).flatten())
        # remove isolated nodes
        batchedgraph = transform(batchedgraph)

        return batchedgraph
        
        
    def __enter__(self):
        return self

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'



# tmp: manual version for testing
def hierarchical_loader(graph, k=2):
    mask = graph.train_mask
    pseodonodes = graph.node_id[graph.pseudo_mask & mask]
    iterator = pseodonodes
    idx = iterator[200:201]

    # get k-hop neighbours of pseudonodes
    pseudo_anchor_down = k_hop_subgraph(idx, num_hops=int(k/2)+1, edge_index= graph.edge_index, flow="target_to_source", directed=True)[0]
    pseudo_anchor_up = k_hop_subgraph(idx, num_hops=int(k/2)+1, edge_index= graph.edge_index, flow="source_to_target", directed=True)[0]

    # graph.pseudo_mask[pseudo_anchor_down].sum()
    # graph.psp_mask[pseudo_anchor_down].sum()
    # graph.pcp_mask[pseudo_anchor_down].sum()
    # graph.subnode_mask[pseudo_anchor_down].sum()

    # graph.pseudo_mask[pseudo_anchor_up].sum()
    # graph.psp_mask[pseudo_anchor_up].sum()
    # graph.pcp_mask[pseudo_anchor_up].sum()
    # graph.subnode_mask[pseudo_anchor_up].sum()

    # get all adjacent pseudonodes for psps
    psp = graph.node_id[pseudo_anchor_down]
    psp_anchor_up = k_hop_subgraph(psp[graph.psp_mask[psp]], num_hops=1, edge_index= graph.edge_index, flow="source_to_target", directed=True)[0]
    pseudo_set = psp_anchor_up[graph.pseudo_mask[psp_anchor_up]]
    
    # psp_anchor_up.shape[0]
    # graph.pseudo_mask[psp_anchor_up].sum()
    # graph.psp_mask[psp_anchor_up].sum()
    # graph.pcp_mask[psp_anchor_up].sum()

    # combine all
    batchedgraph =  graph.subgraph(torch.cat([pseudo_set,pseudo_anchor_down,pseudo_anchor_up]).flatten())
    # remove isolated nodes
    batchedgraph = transform(batchedgraph)

    return batchedgraph
    

