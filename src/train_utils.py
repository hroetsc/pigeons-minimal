### SPLICING PREDICTOR ###
# description:  PCPS predictor training functions
# author:       HPR

from tqdm import tqdm
import torch
import numpy as np
from torcheval.metrics import MulticlassAccuracy,MulticlassAUPRC,MulticlassAUROC
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import add_remaining_self_loops
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# ----- reset model parameters -----
def reset_parameters(model, OUTNAME):

    model.load_state_dict(torch.load(str('results/'+OUTNAME+'/classifier_init.pt')))

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    return model

# deg = degree(graph.edge_index[0], graph.num_nodes)
# deg[1:50]
# torch.max(deg)
# graph.node_info['product']

# ----- performance metrics -----
def performance_metrics(pred, true):
    
    # accuracy
    metric = MulticlassAccuracy(num_classes=5,average='macro')
    metric.update(pred, true.to(torch.int64))
    acc = metric.compute().item()

    # area under PR curve
    metric = MulticlassAUPRC(num_classes=5,average=None)
    metric.update(pred, true)
    pr = metric.compute()[-1].item()

    # area under ROC curve
    metric = MulticlassAUROC(num_classes=5,average='macro')
    metric.update(pred, true)
    roc = metric.compute().item()

    return acc, pr, roc


def gradient_size(classifier):
    with torch.no_grad():
        grad = classifier.output_layer.weight.grad.detach()
        gradnorm = torch.norm(grad, dim=1)

    return gradnorm.numpy()


# def dirichlet_energy(x, edge_index, num_nodes, self_loops=True):
#     edge_weight = torch.ones((edge_index.size(1), ))
#     if self_loops:
#         edge_index = add_remaining_self_loops(edge_index)[0]
#     num_nodes = x.shape[0]

#     # get node degrees
#     row, col = edge_index
#     deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#     if not self_loops:
#         deg += 1.
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
#     # calculate Dirichlet energy
#     term1 = torch.transpose(x[edge_index[0]],-1,0)*deg_inv_sqrt[row]
#     term2 = torch.transpose(x[edge_index[1]],-1,0)*deg_inv_sqrt[col]
#     edge_difference = edge_weight * (torch.norm(torch.transpose(term1,-1,0) - torch.transpose(term2,-1,0),dim=1) ** 2)
    
#     return 0.5*torch.sum(edge_difference)


# ----- plot intermediate metrics -----
def running_mean(x,N=100):
    x = np.array(x)
    N = N if x.shape[0] > 200 else 1
    return np.convolve(x, np.ones(N)/N, mode='same')


def plot_intermediate_metrics(tot_loss, tot_pr, tot_roc, tot_acc, tot_grad, OUTNAME, suffix):

    n_iters = len(tot_loss)
    alpha = .1

    fig, axs = plt.subplots(2,3, figsize=(15,10))
    fig.suptitle(suffix)

    # loss
    axs[0,0].plot(np.arange(n_iters), tot_loss, color="black", alpha=alpha)
    axs[0,0].plot(np.arange(n_iters), running_mean(tot_loss), color="black")
    axs[0,0].set_title('loss')
    axs[0,0].set(xlabel='iteration',ylabel='loss')
    # AU-PR
    axs[0,1].plot(np.arange(n_iters), tot_pr, color="black", alpha=alpha)
    axs[0,1].plot(np.arange(n_iters), running_mean(tot_pr), color="black")
    axs[0,1].set_title('AU-PR')
    axs[0,1].set(xlabel='iteration',ylabel='AU-PR')
    # AU-ROC
    axs[0,2].plot(np.arange(n_iters), tot_roc, color="black", alpha=alpha)
    axs[0,2].plot(np.arange(n_iters), running_mean(tot_roc), color="black")
    axs[0,2].set_title('AU-ROC')
    axs[0,2].set(xlabel='iteration',ylabel='AU-ROC')
    # accuracy
    axs[1,0].plot(np.arange(n_iters), tot_acc, color="black", alpha=alpha)
    axs[1,0].plot(np.arange(n_iters), running_mean(tot_acc), color="black")
    axs[1,0].set_title('accuracy')
    axs[1,0].set(xlabel='iteration',ylabel='accuracy')
    # gradient size
    axs[1,1].plot(np.arange(n_iters), [x[0] for x in tot_grad], color="black", alpha=alpha)
    axs[1,1].plot(np.arange(n_iters), [x[1] for x in tot_grad], color="#F03E72", alpha=alpha)
    axs[1,1].plot(np.arange(n_iters), [x[2] for x in tot_grad], color="#A32A4F", alpha=alpha)
    axs[1,1].plot(np.arange(n_iters), [x[3] for x in tot_grad], color="#EC9A56", alpha=alpha)
    axs[1,1].plot(np.arange(n_iters), [x[4] for x in tot_grad], color="#7B80C7", alpha=alpha)
    axs[1,1].plot(np.arange(n_iters), running_mean([x[0] for x in tot_grad]), color="black")
    axs[1,1].plot(np.arange(n_iters), running_mean([x[1] for x in tot_grad]), color="#F03E72")
    axs[1,1].plot(np.arange(n_iters), running_mean([x[2] for x in tot_grad]), color="#A32A4F")
    axs[1,1].plot(np.arange(n_iters), running_mean([x[3] for x in tot_grad]), color="#EC9A56")
    axs[1,1].plot(np.arange(n_iters), running_mean([x[4] for x in tot_grad]), color="#7B80C7")
    axs[1,1].set_title('gradients')
    axs[1,1].set(xlabel='iteration',ylabel='L2 norm gradient output layer')

    fig.savefig(str('results/'+OUTNAME+'/'+suffix+'.png'), dpi=200)
    plt.close()


# ----- get class weights -----
def get_class_weights(graph, LOSSWEIGHTFCT, reverse=False):
    
    labels = graph.y
    class_counts = labels.bincount()

    total = torch.Tensor([graph.x.shape[0],
                          graph.node_id[graph.pseudo1_mask].shape[0],
                          graph.node_id[graph.pseudo2_mask].shape[0],
                          graph.node_id[graph.pcp_mask].shape[0],
                          graph.node_id[graph.psp_mask].shape[0]])
    # total = graph.x.shape[0]

    if reverse:
        weights = 1-(class_counts/total)
        weights[4] = LOSSWEIGHTFCT*weights[4]
        weights = weights/torch.max(weights)
    else:
        weights = total/class_counts
        weights[4] = LOSSWEIGHTFCT*weights[4]
        
    print(f"class weight: {weights}")
    
    return weights



# ----- initial loss -----
def get_initial_loss(model, loader, loss_fn):
    batch = next(iter(loader))

    model.eval()
    out,lconstr = model(batch.x, batch.edge_index, batch.edge_attr)
    
    initial_loss = loss_fn(out, batch.y)+lconstr

    print(f"initial loss: {initial_loss:.4f}")



# ----- overfit on single batch -----
def sanity_check(model, loader, optimizer, loss_fn, OUTNAME, n_iters = 100):
    
    tot_loss = []
    tot_acc = []
    tot_pr = []
    tot_roc = []
    tot_grad = []

    batch = next(iter(loader))

    print(f'PCP: {batch.pcp_mask.sum()/batch.y.shape[0]:.4f}')
    print(f'PSP: {batch.psp_mask.sum()/batch.y.shape[0]:.4f}')

    for i in range(n_iters):
        # zero gradients for each batch
        optimizer.zero_grad()

        # apply model
        out,lconstr = model(batch.x, batch.edge_index, batch.edge_attr)

        # compute loss and its gradients
        loss = loss_fn(out, batch.y)+lconstr
        loss.backward()

        # adjust weights
        optimizer.step()

        acc, pr, roc = performance_metrics(out, batch.y)
        grad = gradient_size(model)
        tot_acc.append(acc)
        tot_pr.append(pr)
        tot_roc.append(roc)

        tot_grad.append(grad)
        tot_loss.append(loss.item())
    
    plot_intermediate_metrics(tot_loss, tot_pr, tot_roc, tot_acc, tot_grad,
                              OUTNAME, suffix='metrics_overfitting')

    return tot_loss[-1], tot_acc[-1], tot_pr[-1], tot_roc[-1]


# ----- check settings on few iterations -----
def check_settings(model, loader, optimizer, loss_fn, OUTNAME, n_iters=100, suffix='metrics_initial', scheduler=None):

    model.train()
    loop = tqdm(loader)

    tot_loss = []
    tot_acc = []
    tot_pr = []
    tot_roc = []
    tot_grad = []
    
    for i, batch in enumerate(loop):
        if i >= n_iters:
            break
        else:
            # zero gradients for each batch
            optimizer.zero_grad()
            # apply model
            out,lconstr = model(batch.x, batch.edge_index, batch.edge_attr)

            loss = loss_fn(out, batch.y) + lconstr
            loss.backward()
            # adjust weights
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            # reporting
            acc, pr, roc = performance_metrics(out, batch.y)
            grad = gradient_size(model)

            tot_acc.append(acc)
            tot_pr.append(pr)
            tot_roc.append(roc)
            tot_grad.append(grad)
            tot_loss.append(loss.item())

            loop.set_postfix(loss = loss.item(), grad = grad[4])
    
    plot_intermediate_metrics(tot_loss, tot_pr, tot_roc, tot_acc, tot_grad, OUTNAME, suffix=suffix)




# ----- training -----
def train(model, loader, optimizer, loss_fn, pcp_mask, augmentation_fn, N, aug, OUTNAME, epoch):
    
    tot_loss = []
    tot_acc = []
    tot_pr = []
    tot_roc = []
    tot_grad = []

    loop = tqdm(loader)
    for i, batch in enumerate(loop):
        
        # zero gradients for each batch
        optimizer.zero_grad()

        # apply model
        out,lconstr = model(batch.x, batch.edge_index, batch.edge_attr)
        y = batch.y
        # augmentation
        if aug:
            # get positive class
            k = torch.where(batch.y != 0)[0]
            if (len(k)>0):
                sub = batch.subgraph(k)
                # augment N times
                for j in range(N):
                    augbatch = augmentation_fn(sub)
                    # pass through model
                    phat,cntlconstr = model(x = augbatch.x, edge_index = augbatch.edge_index, augbatch=batch.edge_attr)

                    out = torch.cat((out,phat), dim=0)
                    lconstr = torch.cat((lconstr,cntlconstr), dim=0)
                    y = torch.cat((y,sub.y),dim=0)

        # compute loss and its gradients
        if pcp_mask:
            loss = loss_fn(out[batch.pcp_mask], y[batch.pcp_mask]) + lconstr  # !!!!!!!!
        else:
            loss = loss_fn(out, y) + lconstr
        loss.backward()

        # adjust weights
        optimizer.step()

        # reporting
        acc, pr, roc = performance_metrics(out, y)
        grad = gradient_size(model)

        tot_acc.append(acc)
        tot_pr.append(pr)
        tot_roc.append(roc)
        tot_loss.append(loss.item())
        tot_grad.append(grad)

        loop.set_postfix(loss = loss.item(), grad = grad[4])

    plot_intermediate_metrics(tot_loss, tot_pr, tot_roc, tot_acc, tot_grad,
                              OUTNAME, suffix=f'metrics_epoch{epoch}')

    return torch.tensor(tot_loss), torch.tensor(tot_acc), torch.tensor(tot_pr), torch.tensor(tot_roc)



# ----- validation -----
def validate(model, loader, loss_fn, augmentation_fn=None, N=None, aug=False):
    
    tot_loss = []
    tot_acc = []
    tot_pr = []
    tot_roc = []

    with torch.no_grad():

        loop = tqdm(loader)
        for i,batch in enumerate(loop):

            # apply model
            out,lconstr = model(batch.x, batch.edge_index, batch.edge_attr)
            y = batch.y
            
            if aug:
                # get positive class
                k = torch.where(batch.y != 0)[0]
                if (len(k)>0):
                    sub = batch.subgraph(k)
                    # augment N times
                    for j in range(N):
                        augbatch = augmentation_fn(sub)
                        # pass through model
                        phat,cntlconstr = model(x = augbatch.x, edge_index = augbatch.edge_index, augbatch=batch.edge_attr)

                        out = torch.cat((out,phat), dim=0)
                        lconstr = torch.cat((lconstr,cntlconstr), dim=0)
                        y = torch.cat((y,sub.y),dim=0)

            # compute loss and its gradients
            loss = loss_fn(out, y) + lconstr
            
            acc, pr, roc = performance_metrics(out, y)
            tot_acc.append(acc)
            tot_pr.append(pr)
            tot_roc.append(roc)
            tot_loss.append(loss.item())

            loop.set_postfix(loss = loss.item())

    return torch.tensor(tot_loss), torch.tensor(tot_acc), torch.tensor(tot_pr), torch.tensor(tot_roc)



# ----- testing -----
def test(model, loader, graph=None):

    all_pred = torch.tensor([], dtype=torch.float32)
    all_true = torch.tensor([], dtype=torch.float32)
    
    batch0 = next(iter(loader))
    all_info = batch0.node_info.clear()
    ginfo = graph.node_info.filter(pl.col('test_mask') == True)

    with torch.no_grad():

        # k = torch.where(graph.test_mask)[0]
        # test_graph = graph.subgraph(k)
        # all_pred, lconstr = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)
        # all_info = graph.node_info.filter(pl.col('test_mask') == True)
        # all_true = test_graph.y

        loop = tqdm(loader)
        for i, batch in enumerate(loop):
            
            out,lconstr = model(batch.x, batch.edge_index, batch.edge_attr)
            
            all_pred = torch.cat((all_pred, out.detach()), dim=0)
            all_true = torch.cat((all_true, batch.y), dim=0)
            
            nid = batch.n_id.numpy()
            all_info = all_info.extend(ginfo[nid])
        
    acc, pr, roc = performance_metrics(all_pred, all_true)

    return all_pred, all_true, all_info, acc, pr, roc



# ----- self-supervised pre-training -----
def augmentation(encoder, proj_head, augmentation_fn, loader, optimizer, loss_fn, mode="single"):
    
    tot_loss = []
    loop = tqdm(loader)
    for i, batch in enumerate(loop):

        # zero gradients for each batch
        optimizer.zero_grad()

        # apply model
        augbatch = augmentation_fn(batch)
        p1 = encoder(augbatch.x, augbatch.edge_index)
        z1 = proj_head(p1)

        if mode == "single":
            p = encoder(batch.x, batch.edge_index)
            z = proj_head(p)
        elif mode == "double":
            augbatch0 = augmentation_fn(batch)
            p = encoder(augbatch0.x, augbatch0.edge_index)
            z = proj_head(p)

        # compute loss and its gradients
        loss = -0.5*loss_fn(p,z1)-0.5*loss_fn(p1,z)
        loss.backward()

        # adjust weights
        optimizer.step()

        # reporting
        tot_loss.append(loss.item())
        loop.set_postfix(loss = loss.item())

    return torch.tensor(tot_loss)