### SPLICING PREDICTOR ###
# description:  PCPS predictor data handling functions
# author:       HPR

import os
import torch_geometric
from torch_geometric.loader import ImbalancedSampler, NeighborLoader, RandomNodeLoader, ShaDowKHopSampler, GraphSAINTNodeSampler
from torch_geometric.utils import add_remaining_self_loops
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import polars as pl
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns


# ----- read in data files and create graph object -----
def parse_data(infileGraph, infileNodes, outfile, self_loops=False, redo=False, binary_width=15):

    # load csv files and create graph object
    if not os.path.exists(outfile) or redo:
        print(f'constructing graph object from files: {infileGraph} and {infileNodes}')
        
        # --- read in data
        # NOTE: increase infer_schema_length parameter if trouble reading in data
        adjacency_list = pl.read_csv(infileGraph, infer_schema_length=1000000000)
        node_embedding = pl.read_csv(infileNodes, infer_schema_length=1000000000)
        
        # labels
        node_labels = node_embedding['target']

        # node embeddings
        # NOTE: this is fiddly - change when input files change!
        # NOTE: this keeps the class
        embeddings = node_embedding.drop(['substrateID','substrateLength','product','ID','index', 'observed','target',
                                          'train_mask','val_mask','test_mask',
                                          'pcp_mask','psp_mask','pseudo_mask','pseudo1_mask','pseudo2_mask','subnode_mask'])
        embeddings = embeddings.with_columns(pl.all().str.replace("−","-").cast(pl.Float32, strict=False)).to_numpy()
        
        # NOTE: this removes the class
        # embeddings = node_embedding.drop(['substrateID','substrateLength','product','ID','index', 'observed','target',
        #                                   'train_mask','val_mask','test_mask', 'class',
        #                                   'pcp_mask','psp_mask','pseudo_mask','pseudo1_mask','pseudo2_mask','subnode_mask']).to_numpy()
        # # NOTE: this keeps the substrate length:
        # embeddings = node_embedding.drop(['substrateID','product','ID','index', 'observed','class', 'target',
        #                                   'train_mask','val_mask','test_mask','pcp_mask','psp_mask','pseudo_mask','subnode_mask']).to_numpy()
        # NOTE: this keeps the P1/P1' (and substrate length)
        # abs_pos = node_embedding['product'].str.split(by="_").cast(pl.List(pl.Float32)).to_numpy()
        # k1 = (node_embedding['pseudo_mask'] & node_embedding['product'].str.contains('p1_')).to_numpy()
        # k1_ = (node_embedding['pseudo_mask'] & node_embedding['product'].str.contains("p1'_")).to_numpy()
        # k2 = k1|k1_ == False
        # abs_pos_1 = np.vstack(abs_pos[k1])[:,1:]
        # abs_pos_1_ = np.vstack(abs_pos[k1_])[:,1:]
        # abs_pos_2 = np.vstack(abs_pos[k2])[:,1:]
        # abs_pos = np.vstack((np.hstack((abs_pos_1, np.zeros_like(abs_pos_1))),
        #                      np.hstack((np.zeros_like(abs_pos_1_), abs_pos_1_)),
        #                      abs_pos_2))
        # idx = np.concatenate((np.where(k1)[0], np.where(k1_)[0], np.where(k2)[0])).flatten()
        # abs_pos = abs_pos[np.argsort(idx),:]
        # abs_pos = np.nan_to_num(abs_pos,0)
        # substrateLength = node_embedding['substrateLength'].to_numpy()
        # embeddings = np.hstack([substrateLength.astype(float).reshape((-1,1)),abs_pos,embeddings.astype(float)])

        # # convert to binary
        # placeholder = list('0'*binary_width)
        # p1_bit = [list(np.binary_repr(x, width=binary_width)) if x >= 0 else placeholder for x in abs_pos[:,0].astype(int)]
        # p1p_bit = [list(np.binary_repr(x, width=binary_width)) if x >= 0 else placeholder for x in abs_pos[:,1].astype(int)]
        # L_bit = [list(np.binary_repr(x, width=binary_width)) if x >= 0 else placeholder for x in substrateLength.astype(int)]
        # bits = np.hstack([np.array(p1_bit), np.array(p1p_bit), np.array(L_bit)])
        # # combine with original embeddings
        # embeddings = np.hstack([bits.astype(float),embeddings.astype(float)])
        
        # edge weights
        if 'edge_weight' in node_embedding.columns:
            edge_weights = node_embedding.with_columns(pl.col('edge_weight').str.replace("−","-").cast(pl.Float32, strict=False)).to_numpy()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_weights = None


        # adjacency list
        if not self_loops:
            adjacency_list = adjacency_list.filter(pl.col('educt_index') != pl.col('product_index'))
        adj_list = torch.tensor(adjacency_list[['educt_index','product_index']].to_numpy().transpose(), dtype=torch.long)

        # --- concatenate to PyG graph object
        graph = torch_geometric.data.Data(x=torch.tensor(embeddings, dtype=torch.float),
                                          edge_index = adj_list, # adjacency list
                                          edge_attr = edge_weights,
                                          y = torch.tensor(node_labels, dtype=torch.long),
                                          node_id = torch.tensor(node_embedding['index'], dtype=torch.long),
                                          train_mask = torch.tensor(node_embedding['train_mask'], dtype=torch.bool),
                                          val_mask = torch.tensor(node_embedding['val_mask'], dtype=torch.bool),
                                          test_mask = torch.tensor(node_embedding['test_mask'], dtype=torch.bool),
                                          pcp_mask = torch.tensor(node_embedding['pcp_mask'], dtype=torch.bool),
                                          psp_mask = torch.tensor(node_embedding['psp_mask'], dtype=torch.bool),
                                          pseudo_mask = torch.tensor(node_embedding['pseudo_mask'], dtype=torch.bool),
                                          pseudo1_mask = torch.tensor(node_embedding['pseudo1_mask'], dtype=torch.bool),
                                          pseudo2_mask = torch.tensor(node_embedding['pseudo2_mask'], dtype=torch.bool),
                                          subnode_mask = torch.tensor(node_embedding['subnode_mask'], dtype=torch.bool),
                                          node_info = node_embedding[['substrateID','substrateLength','product','ID',
                                                                      'index', 'observed', 'target',
                                                                      'train_mask','val_mask','test_mask',
                                                                      'pcp_mask','psp_mask','pseudo_mask','pseudo1_mask','pseudo2_mask','subnode_mask']])
        
        # add self-loops
        if self_loops:
            graph.edge_index = add_remaining_self_loops(edge_index=graph.edge_index, edge_attr=graph.edge_attr, fill_value=0)[0]

        # save datasets
        torch.save(graph, f=outfile)

    # load graph object
    else:
        print(f'loading graph object from: {outfile}')
        graph = torch.load(outfile)
    

    imb = graph.y.bincount()
    
    # stats
    print("----------------------------------------")
    print(graph)
    print(f'class imbalance spliced: {100*imb[4]/graph.node_id[graph.psp_mask].numel():.2f}%')
    print(f'class imbalance non-spliced: {100*imb[3]/graph.node_id[graph.pcp_mask].numel():.2f}%')
    print(f'class imbalance pseudo P1 prime: {100*imb[2]/graph.node_id[graph.pseudo1_mask].numel():.2f}%')
    print(f'class imbalance pseudo P1: {100*imb[1]/graph.node_id[graph.pseudo2_mask].numel():.2f}%')
    print(f'number of nodes: {graph.num_nodes}')
    print(f'number of edges: {graph.num_edges}')
    print(f'node embedding dimensionality: {graph.num_node_features}')
    print(f'directed: {graph.is_directed()}')
    print(f'self-loops: {graph.has_self_loops()}')
    print(f'isolated nodes: {graph.has_isolated_nodes()}')
    print("----------------------------------------")

    return graph


# ----- data loaders -----
def data_loader(Loader, graph, batch_size, total_layers, num_neighbours):

    # total_layers += 1
    
    # --- neighbor loader
    ktrain = torch.where(graph.train_mask)[0]
    train_loader = NeighborLoader(data=graph.subgraph(ktrain), #data=graph, input_nodes=graph.train_mask, 
                                  num_neighbors=[num_neighbours]*total_layers, batch_size=batch_size,
                                  shuffle = True, subgraph_type='directional', replace=True)
    kval = torch.where(graph.val_mask)[0]
    val_loader = NeighborLoader(data=graph.subgraph(kval), #data=graph, input_nodes=graph.val_mask,
                                num_neighbors=[num_neighbours]*total_layers, batch_size=batch_size,
                                shuffle = True, subgraph_type='directional', replace=True)
    ktest = torch.where(graph.test_mask)[0]
    test_loader = NeighborLoader(data=graph.subgraph(ktest), #data=graph, input_nodes=graph.test_mask,
                                 num_neighbors=[num_neighbours]*total_layers, batch_size=batch_size,
                                 shuffle = True, subgraph_type='directional', replace=True)
    
    k = torch.where(graph.train_mask & graph.y != 0)[0]
    pos_graph = graph.subgraph(k)
    pos_loader = NeighborLoader(data=pos_graph, # input_nodes=graph.train_mask & graph.y != 0, # centre nodes detected, but graph containing non-detected products
                                num_neighbors=[num_neighbours]*total_layers, batch_size=batch_size,
                                shuffle = True, subgraph_type='directional', replace=True)
    
    # # --- custom loader
    # train_loader = Loader(graph=graph, k=total_layers, mask=graph.train_mask, batch_size=batch_size, shuffle=True)
    # val_loader = Loader(graph=graph, k=total_layers, mask=graph.val_mask, batch_size=batch_size, shuffle=True)
    # test_loader = Loader(graph=graph, k=total_layers, mask=graph.test_mask, batch_size=batch_size, shuffle=True)
    
    # imbalance
    batch = next(iter(train_loader))
    print(batch.y.bincount()/batch.y.numel())
    print(f'PCP: {batch.pcp_mask.sum()/batch.y.shape[0]:.4f}')
    print(f'PSP: {batch.psp_mask.sum()/batch.y.shape[0]:.4f}')

    return train_loader, val_loader, test_loader, pos_loader



# ----- plot training history -----
def running_mean(x,N=500):
    x = np.array(x)
    N = N if x.shape[0] > 200 else 1
    return np.convolve(x, np.ones(N)/N, mode='same')

def evaluate_performance(train_metrics, val_metrics, all_true, all_pred, OUTNAME):
    print("evaluate model performance")

    pcp_colour = "#EC9A56"
    psp_colour = "#7B80C7"
    pseudo1_colour = "#F03E72"
    # pseudo2_colour = "#A32A4F"
    val_colour = "#38BA41"    

    # --- format metrics
    # NOTE: this is fiddly, change when metrics change!
    if not train_metrics == None:
        train_df = pd.DataFrame.from_records(train_metrics.numpy(),
                                                columns=['train_loss','train_accuracy','train_pr','train_roc'])
        val_df = pd.DataFrame.from_records(val_metrics.numpy(),
                                            columns=['val_loss','val_accuracy','val_pr','val_roc'])
        train_df['iteration'] = train_df.index+1
        val_df['iteration'] = val_df.index+1

    # --- get ROC and PR curves
    all_pred = torch.exp(all_pred)
    
    pred_pseudo = all_pred[:,1].numpy()
    true_pseudo = np.zeros(pred_pseudo.shape[0])
    true_pseudo[all_true.numpy() == 1] = 1

    pred_pcp = all_pred[:,3].numpy()
    true_pcp = np.zeros(pred_pcp.shape[0])
    true_pcp[all_true.numpy() == 3] = 1

    pred_psp = all_pred[:,4].numpy()
    true_psp = np.zeros(pred_psp.shape[0])
    true_psp[all_true.numpy() == 4] = 1

    fpr_pseudo,tpr_pseudo,thresholds = roc_curve(true_pseudo, pred_pseudo)
    test_roc_pseudo = roc_auc_score(true_pseudo, pred_pseudo)
    prec_pseudo,rec_pseudo,thresholds = precision_recall_curve(true_pseudo, pred_pseudo)
    test_pr_pseudo = auc(rec_pseudo, prec_pseudo)

    fpr_pcp,tpr_pcp,thresholds = roc_curve(true_pcp, pred_pcp)
    test_roc_pcp = roc_auc_score(true_pcp,pred_pcp)
    prec_pcp,rec_pcp,thresholds = precision_recall_curve(true_pcp, pred_pcp)
    test_pr_pcp = auc(rec_pcp, prec_pcp)

    fpr_psp,tpr_psp,thresholds = roc_curve(true_psp, pred_psp)
    test_roc_psp = roc_auc_score(true_psp,pred_psp)
    prec_psp,rec_psp,thresholds = precision_recall_curve(true_psp, pred_psp)
    test_pr_psp = auc(rec_psp, prec_psp)

    # --- plot curves
    alpha = .1

    if not train_metrics == None:
        fig, axs = plt.subplots(3,4, figsize=(20,15))
    else:
        fig, axs = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle('predictor performance')
    
    if not train_metrics == None:
        # loss
        axs[0,0].plot(train_df['iteration'],train_df['train_loss'], color="black", alpha=alpha, label='train')
        axs[0,0].plot(train_df['iteration'],running_mean(train_df['train_loss']), color="black")
        axs[0,0].set_title('training loss')
        axs[0,0].set(xlabel='iteration',ylabel='loss')
        
        axs[0,1].plot(val_df['iteration'],val_df['val_loss'], color=val_colour, alpha=alpha, label='val')
        axs[0,1].plot(val_df['iteration'],running_mean(val_df['val_loss']), color=val_colour)
        axs[0,1].set_title('validation loss')
        axs[0,1].set(xlabel='iteration',ylabel='loss')
        
        # accuracy
        axs[0,2].plot(train_df['iteration'],train_df['train_accuracy'], color="black", alpha=alpha, label='train')
        axs[0,2].plot(train_df['iteration'],running_mean(train_df['train_accuracy']), color="black")
        axs[0,2].set_title('training accuracy')
        axs[0,2].set(xlabel='iteration',ylabel='accuracy')
        
        axs[0,3].plot(val_df['iteration'],val_df['val_accuracy'], color=val_colour, alpha=alpha, label='val')
        axs[0,3].plot(val_df['iteration'],running_mean(val_df['val_accuracy']), color=val_colour)
        axs[0,3].set_title('validation accuracy')
        axs[0,3].set(xlabel='iteration',ylabel='accuracy')
        
        # AUPR (over time)
        axs[1,0].plot(train_df['iteration'],train_df['train_pr'], color="black", alpha=alpha, label='train')
        axs[1,0].plot(train_df['iteration'],running_mean(train_df['train_pr']), color="black")
        axs[1,0].set_title('training AU-PR')
        axs[1,0].set(xlabel='iteration',ylabel='AU-PR')
        
        axs[1,1].plot(val_df['iteration'],val_df['val_pr'], color=val_colour, alpha=alpha, label='val')
        axs[1,1].plot(val_df['iteration'],running_mean(val_df['val_pr']), color=val_colour)
        axs[1,1].set_title('validation AU-PR')
        axs[1,1].set(xlabel='iteration',ylabel='AU-PR')
        
        # AUROC (over time)
        axs[1,2].plot(train_df['iteration'],train_df['train_roc'], color="black", alpha=alpha, label='train')
        axs[1,2].plot(train_df['iteration'],running_mean(train_df['train_roc']), color="black")
        axs[1,2].set_title('training AU-ROC')
        axs[1,2].set(xlabel='iteration',ylabel='AU-ROC')
        
        axs[1,3].plot(val_df['iteration'],val_df['val_roc'], color=val_colour, alpha=alpha, label='val')
        axs[1,3].plot(val_df['iteration'],running_mean(val_df['val_roc']), color=val_colour)
        axs[1,3].set_title('validation AU-ROC')
        axs[1,3].set(xlabel='iteration',ylabel='AU-ROC')
        
        # ROC curve
        axs[2,0].plot(tpr_pseudo,fpr_pseudo, color=pseudo1_colour, label='pseudo')
        axs[2,0].plot(tpr_pcp,fpr_pcp, color=pcp_colour, label='pcp')
        axs[2,0].plot(tpr_psp,fpr_psp, color=psp_colour, label='psp')
        axs[2,0].set_title(f'ROC curve, AUC: {100*test_pr_pseudo:.2f}%, {100*test_roc_pcp:.2f}%, {100*test_roc_psp:.2f}%')
        axs[2,0].set(xlabel='1-specificity (FPR)',ylabel='sensitivity (TPR)')
        axs[2,0].legend()
        # PR curve
        axs[2,1].plot(rec_pseudo,prec_pseudo, color=pseudo1_colour, label='pseudo')
        axs[2,1].plot(rec_pcp,prec_pcp, color=pcp_colour, label='pcp')
        axs[2,1].plot(rec_psp,prec_psp, color=psp_colour, label='psp')
        axs[2,1].set_title(f'PR curve, AUC: {100*test_pr_pseudo:.2f}%, {100*test_pr_pcp:.2f}%, {100*test_pr_psp:.2f}%')
        axs[2,1].set(xlabel='recall',ylabel='precision')
        axs[2,1].legend()

    else:
        # ROC curve
        axs[0].plot(tpr_pseudo,fpr_pseudo, color=pseudo1_colour, label='pseudo')
        axs[0].plot(tpr_pcp,fpr_pcp, color=pcp_colour, label='pcp')
        axs[0].plot(tpr_psp,fpr_psp, color=psp_colour, label='psp')
        axs[0].set_title(f'ROC curve, AUC: {100*test_roc_pseudo:.2f}%, {100*test_roc_pcp:.2f}%, {100*test_roc_psp:.2f}%')
        axs[0].set(xlabel='1-specificity (FPR)',ylabel='sensitivity (TPR)')
        axs[0].legend()
        # PR curve
        axs[1].plot(rec_pseudo,prec_pseudo, color=pseudo1_colour, label='pseudo')
        axs[1].plot(rec_pcp,prec_pcp, color=pcp_colour, label='pcp')
        axs[1].plot(rec_psp,prec_psp, color=psp_colour, label='psp')
        axs[1].set_title(f'PR curve, AUC: {100*test_pr_pseudo:.2f}%, {100*test_pr_pcp:.2f}%, {100*test_pr_psp:.2f}%')
        axs[1].set(xlabel='recall',ylabel='precision')
        axs[1].legend()

    fig.savefig(str('results/'+OUTNAME+'/performance.png'), dpi=200)
    
    if not train_metrics == None:
        val_df['dataset'] = 'validation'
        train_df['dataset'] = 'training'
        metrics = pd.concat([train_df, val_df],axis=0)
    else:
        metrics = None

    return metrics


# ----- plot loss of encoder -----
def plot_loss(metric, OUTNAME):
    fig, axs = plt.subplots(1, figsize=(5,5))
    fig.suptitle('encoder')
    
    # loss - training
    axs.plot(np.arange(metric.shape[0]), metric, color="black")
    axs.set_title('loss')
    axs.set(xlabel='iteration',ylabel='loss')
    
    fig.savefig(str('results/'+OUTNAME+'/encoder_loss.png'), dpi=200)


# ----- plot gradient/weights -----
def plot_gradient(classifier, OUTNAME, suffix='init'):

    with torch.no_grad():

        n = len(classifier.layers_GCN)

        fig, axs = plt.subplots(n+2,2, figsize=(10,int(5*(n+2))))
        fig.suptitle('gradients and weights')
        
        sns.heatmap(ax=axs[0,0],
                    data=classifier.input_layer.weight.grad.numpy().transpose(),
                    cmap=sns.cubehelix_palette(as_cmap=True))
        axs[0,0].set_title('gradient: input layer')

        sns.heatmap(ax=axs[0,1],
                    data=classifier.input_layer.weight.detach().numpy().transpose(),
                    cmap=sns.cubehelix_palette(as_cmap=True))
        axs[0,1].set_title('weights: input layer')

        for i in range(n):
            sns.heatmap(ax=axs[i+1,0], data=classifier.layers_GCN[i].weight.grad.numpy().transpose(), cmap=sns.cubehelix_palette(as_cmap=True))
            axs[i+1,0].set_title(f'gradient: GCN layer {i}')
            sns.heatmap(ax=axs[i+1,1], data=classifier.layers_GCN[i].weight.detach().numpy().transpose(), cmap=sns.cubehelix_palette(as_cmap=True))
            axs[i+1,1].set_title(f'weights: GCN layer {i}')
        
        sns.heatmap(ax=axs[n+1,0],
                    data=classifier.output_layer.weight.grad.numpy().transpose(),
                    cmap=sns.cubehelix_palette(as_cmap=True))
        axs[n+1,0].set_title('gradient: output layer')
        sns.heatmap(ax=axs[n+1,1],
                    data=classifier.output_layer.weight.detach().numpy().transpose(),
                    cmap=sns.cubehelix_palette(as_cmap=True))
        axs[n+1,1].set_title('weights: output layer')

        fig.savefig(str('results/'+OUTNAME+'/gradients_'+suffix+'.png'), dpi=200)
        plt.close()


# ----- save weights -----
def get_layers(model):
    layers = []
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Sequential):
            layers += get_layers(module)
        elif isinstance(module, torch.nn.ModuleList):
            for m in module:
                layers += get_layers(m)
        else:
            layers.append(module)
    return layers

