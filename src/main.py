### SPLICING PREDICTOR ###
# description:  main script for training PCPS predictor
# author:       HPR

import sys
import os
import yaml
import torch
from torch.optim.lr_scheduler import ConstantLR,StepLR,SequentialLR,CosineAnnealingLR, LambdaLR, LinearLR
import numpy as np
import polars as pl
import torch.nn as nn

# from src.model import *
# from src.train_utils import *
# from src.data_utils import *
# from src.loss import *
# from src.utils import *
from model import *
from train_utils import *
from data_utils import *
from loss import *
from utils import *

set_seed(123)

# hyperparameters
with open('src/args.yaml') as stream:
    args = yaml.safe_load(stream)

OUTNAME = args["OUTNAME"]
PRED_ONLY = args["PRED_ONLY"]
USE_BEST_MODEL = args["USE_BEST_MODEL"]

if PRED_ONLY:
    f = str('results/'+OUTNAME+'/hyperparameters.yaml')
    if os.path.exists(f):
        with open(f) as stream:
            args = yaml.safe_load(stream)

EPOCHS = args["EPOCHS"]
EPOCHS_AUG = args["EPOCHS_AUG"]
FOCAL_LOSS = args["FOCAL_LOSS"]
TRAIN_ENCODER = args["TRAIN_ENCODER"]
TRAIN_CLASSIFIER = args["TRAIN_CLASSIFIER"]
DATA_AUGMENTATION = args["DATA_AUGMENTATION"]
RESUME_TRAINING = args["RESUME_TRAINING"]

os.makedirs(str('results/'+OUTNAME+'/'), exist_ok=True)

# device
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# hyperparameters
with open(str('results/'+OUTNAME+'/hyperparameters.yaml'), "w") as fp:
    yaml.dump(args, fp)

# ----- INPUT -----
print("DATA")
# get graph object
graph = parse_data(infileGraph=args["INFILE_GRAPH"],
                   infileNodes=args["INFILE_NODES"],
                   outfile=args["GRAPHFILE"],
                   self_loops=args["SELF_LOOPS"],
                   redo=False)
graph.to(device)

# graph.edge_attr = None
# sanity!!!
# torch.unique(graph.y[graph.pseudo1_mask])
# torch.unique(graph.y[graph.pseudo2_mask])

print(f'number of nodes in datasets:\ntrain: {graph.train_mask.sum().item()}, test: {graph.test_mask.sum().item()}, validation: {graph.val_mask.sum().item()}')
node_dimension = graph.x.shape[1]
print(f"embedding dimension: {node_dimension}")


# ----- preprocessing -----
total_layers = 1
train_loader, val_loader, test_loader, pos_loader = data_loader(Loader=EductLoader,
                                                                graph=graph,
                                                                total_layers=total_layers,
                                                                batch_size=args["BATCHSIZE"],
                                                                num_neighbours=args["NUM_NEIGHBOURS"])


print("MODELS")
# --- encoder ---
if DATA_AUGMENTATION:
    encoder = EncoderV4(num_layers=args["LAYERS_ENCODER"],
                        num_feats=node_dimension,
                        num_classes=graph.y.bincount().shape[0],
                        dim_hidden=args["DIM_ENCODER"])
    encoder.to(device)
    proj_head = ProjectionHead(in_dim=args["DIM_ENCODER"],
                               out_dim=args["DIM_ENCODER"])
    proj_head.to(device)
else:
    encoder = None
    proj_head = None


# ----- data augmentation -----
if TRAIN_ENCODER and not PRED_ONLY:

    print("AUGMENTATION")
    lr_encoder = args["ENC_LR"]*(args["BATCHSIZE"]/256)
    optimizer_enc = torch.optim.SGD([*encoder.parameters(), *proj_head.parameters()],
                                    lr=lr_encoder, momentum=0.9, weight_decay=args["WEIGHT_DECAY"])
    loss_fn_enc = LatentLoss()
    scheduler_enc = CosineAnnealingLR(optimizer=optimizer_enc, T_max=args["BATCHSIZE"])
    
    if RESUME_TRAINING:
        if USE_BEST_MODEL:
            encoder.load_state_dict(torch.load(str('results/'+OUTNAME+'/encoder_best.pt')))
        else:
            encoder.load_state_dict(torch.load(str('results/'+OUTNAME+'/encoder.pt')))

    aug_metrics = torch.tensor([], dtype=torch.float32)
    stopping_criteron = torch.tensor(np.tile(np.Inf, EPOCHS_AUG))
    for epoch in range(EPOCHS_AUG):
        print(f"EPOCH {epoch+1}/{EPOCHS_AUG}")

        # --- training
        encoder.train(True)
        proj_head.train()
        
        enc_loss = augmentation(encoder=encoder, proj_head=proj_head,
                                augmentation_fn=augmentation_fn,
                                loader=pos_loader, optimizer=optimizer_enc,
                                loss_fn=loss_fn_enc, mode=args["AUG_MODE"])
        print(f'data augmentation (train dataset):\nloss: {torch.mean(enc_loss):.4f}')
        aug_metrics = torch.cat((aug_metrics, enc_loss), dim=0)
        
        scheduler_enc.step()

        stopping_criteron[epoch] = torch.mean(enc_loss)
        print("saving model")
        if torch.all(torch.mean(enc_loss) <= stopping_criteron):
            print("best model so far!")
            is_best = True
            torch.save(encoder.state_dict(), f=str('results/'+OUTNAME+'/encoder_best.pt'))
        else:
            is_best=False
            torch.save(encoder.state_dict(), f=str('results/'+OUTNAME+'/encoder.pt'))

        encoder.save_checkpoint(OUTNAME=OUTNAME, EPOCH=epoch, optimizer=optimizer_enc, is_best=is_best)

    plot_loss(aug_metrics,OUTNAME)


if DATA_AUGMENTATION:
    encoder.load_state_dict(torch.load(str('results/'+OUTNAME+'/encoder_best.pt')))
    encoder.eval()

    # embedding of graph
    obs_subgraph = graph.subgraph(subset = graph.node_id[graph.node_info['observed']])
    embedding = encoder(obs_subgraph.x, obs_subgraph.edge_index)

    emb_dataset = graph.node_info.filter('observed')
    emb_dataset = emb_dataset.hstack(pl.DataFrame(embedding.detach().numpy()))
    emb_dataset.write_csv(str('results/'+OUTNAME+'/embedding.csv'))


# --- classifier ---
classifier = ClassifierV4(num_layers=args["LAYERS_CLASSIFIER"],
                          num_feats=node_dimension,
                          num_classes=graph.y.bincount().shape[0],
                          dim_hidden=args["DIM_CLASSIFIER"],
                          gamma=args["GAMMA_CONSTRAINT"],
                          dropout=args["DROPOUT"],
                          output_dropout=args["DROPOUTOUT"],
                          c_max=args["CMAX"])
# classifier = ClassifierV5(num_layers=args["LAYERS_CLASSIFIER"],
#                           num_feats=node_dimension,
#                           num_classes=graph.y.bincount().shape[0],
#                           dim_hidden=args["DIM_CLASSIFIER"],
#                           heads=1,
#                           gamma=args["GAMMA_CONSTRAINT"],
#                           dropout=args["DROPOUT"],
#                           output_dropout=args["DROPOUTOUT"],
#                           c_max=args["CMAX"])

classifier.to(device)
torch.save(classifier.state_dict(), f=str('results/'+OUTNAME+'/classifier_init.pt'))


# ----- actual training -----
if TRAIN_CLASSIFIER and not PRED_ONLY:
    print("MODEL TRAINING")
    
    # --- optimizer / lr scheduling
    optimizer_cl0 = torch.optim.Adam(classifier.parameters(), lr=args["LEARNING_RATE"], weight_decay=0)
    optimizer_cl = torch.optim.Adam(classifier.parameters(), lr=args["LEARNING_RATE"], weight_decay=args["WEIGHT_DECAY"])

    train_scheduler = StepLR(optimizer=optimizer_cl, step_size=1, gamma=0.9)
    

    # --- loss
    if args["USE_CLASS_WEIGHTS"]:
        class_weight = get_class_weights(graph=graph, LOSSWEIGHTFCT=args["LOSSWEIGHTFCT"], reverse=args["REVERSE_WEIGHTS"])
    else:
        class_weight = None

    if FOCAL_LOSS:
        loss_fn_cl = FocalLossNLL2(weight=class_weight, gamma=args["FOCAL_GAMMA"])
        # loss_fn_cl = CombinedLoss(weight=class_weight, gamma=args["FOCAL_GAMMA"], alpha1=args["ALPHA1"], alpha2=args["ALPHA2"])
    else:
        loss_fn_cl = nn.NLLLoss(weight=class_weight)

    if RESUME_TRAINING:
        if USE_BEST_MODEL:
            classifier.load_state_dict(torch.load(str('results/'+OUTNAME+'/classifier_best.pt')))
        else:
            classifier.load_state_dict(torch.load(str('results/'+OUTNAME+'/classifier.pt')))
    else:
        # --- get initial loss as sanity check
        get_initial_loss(model=classifier, loader=train_loader, loss_fn=loss_fn_cl)
        s_loss, s_acc, s_pr, s_roc = sanity_check(model=classifier, loader=train_loader,
                                                  optimizer=optimizer_cl0, loss_fn=loss_fn_cl,
                                                  OUTNAME=OUTNAME, n_iters=100)
        print(f'classifier sanity check:\nloss: {s_loss:.4f}, accuracy: {s_acc:.4f}, AU-PR: {s_pr:.4f}, AU-ROC: {s_roc:.4f}')
        
        plot_gradient(classifier=classifier, OUTNAME=OUTNAME, suffix='init')
        # sys.exit(0)

        # reset model parameters
        classifier = reset_parameters(classifier,OUTNAME)

        
        # --- check initial losses
        print("CHECKING SETTINGS")
        check_settings(model=classifier, loader=train_loader,
                       optimizer=optimizer_cl0,loss_fn=loss_fn_cl,
                       OUTNAME=OUTNAME,n_iters=200)
        # reset model parameters
        classifier = reset_parameters(classifier,OUTNAME)

        # --- warmup
        if args["WARMUP"]:
            print("WARMUP")
            optimizer_wu = torch.optim.Adam(classifier.parameters(), lr=args["LEARNING_RATE"], weight_decay=0)
            wu_scheduler = LinearLR(optimizer=optimizer_wu, start_factor=args["WARMUP_FACTOR"], total_iters=args["NOWARMUP"])

            check_settings(model=classifier, loader=train_loader,
                           optimizer=optimizer_wu,loss_fn=loss_fn_cl,
                           OUTNAME=OUTNAME,n_iters=args["NOWARMUP"],
                           suffix='metrics_warmup', scheduler=wu_scheduler)
            optimizer_cl.load_state_dict(optimizer_wu.state_dict())


        
    # --- iterate epochs
    train_metrics = torch.tensor([], dtype=torch.float32)
    val_metrics = torch.tensor([], dtype=torch.float32)
    stopping_criteron = torch.tensor(np.tile(np.Inf, EPOCHS))
    # stopping_criteron = torch.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        
        # --- training
        if DATA_AUGMENTATION:
            encoder.train()
        classifier.train()
        
        t_loss, t_acc, t_pr, t_roc = train(model=classifier,
                                           loader=train_loader,
                                           optimizer=optimizer_cl,
                                           loss_fn=loss_fn_cl,
                                           pcp_mask = True if epoch < 0 else False,
                                           augmentation_fn=augmentation_fn,
                                           N=int(args["NO_AUGS"]),
                                           aug=DATA_AUGMENTATION,
                                           OUTNAME=OUTNAME,
                                           epoch=epoch+1)
        print(f'train dataset:\nloss: {torch.mean(t_loss):.4f}, accuracy: {torch.mean(t_acc):.4f}, AU-PR: {torch.mean(t_pr):.4f}, AU-ROC: {torch.mean(t_roc):.4f}')
        train_metrics = torch.cat((train_metrics, torch.stack((t_loss, t_acc, t_pr, t_roc), dim=-1)), dim=0)

        # --- validation
        if DATA_AUGMENTATION:
            encoder.eval()
        classifier.eval()
        v_loss, v_acc, v_pr, v_roc = validate(model=classifier,
                                              loader=val_loader,
                                              loss_fn=loss_fn_cl,
                                              augmentation_fn=augmentation_fn,
                                              N=args["NO_AUGS"],
                                              aug=DATA_AUGMENTATION)
        print(f'validation dataset:\nloss: {torch.mean(v_loss):.4f}, accuracy: {torch.mean(v_acc):.4f}, AU-PR: {torch.mean(v_pr):.4f}, AU-ROC: {torch.mean(v_roc):.4f}')
        val_metrics = torch.cat((val_metrics, torch.stack((v_loss, v_acc, v_pr, v_roc), dim=-1)), dim=0)
        
        train_scheduler.step()

        # --- stats
        stopping_criteron[epoch] = torch.mean(v_loss)

        print("saving model")
        if torch.all(torch.mean(v_loss) <= stopping_criteron):
            print("best model so far!")
            is_best = True
            torch.save(classifier.state_dict(), f=str('results/'+OUTNAME+'/classifier_best.pt'))
            torch.save(classifier.state_dict(), f=str('results/'+OUTNAME+'/classifier.pt'))
        else:
            is_best=False
            torch.save(classifier.state_dict(), f=str('results/'+OUTNAME+'/classifier.pt'))

        classifier.save_checkpoint(OUTNAME=OUTNAME, EPOCH=epoch, optimizer=optimizer_cl, is_best=is_best)
        # torch.save(classifier.state_dict(), f=f'results/{OUTNAME}/classifier_epoch{epoch}.pt')
    
    plot_gradient(classifier=classifier, OUTNAME=OUTNAME, suffix=f'epoch_{EPOCHS}')
                

if TRAIN_CLASSIFIER or PRED_ONLY:

    if PRED_ONLY:
        train_metrics, val_metrics = None, None
    
    # ----- prediction -----
    print("PREDICTION")

    # load best model
    if USE_BEST_MODEL:
        classifier.load_state_dict(torch.load(str('results/'+OUTNAME+'/classifier_best.pt')))
    else:
        classifier.load_state_dict(torch.load(str('results/'+OUTNAME+'/classifier.pt')))
        
    classifier.eval()
    if DATA_AUGMENTATION:
        encoder.eval()

    # run prediction
    all_pred, all_true, all_info, test_acc, test_pr, test_roc = test(model=classifier, loader=test_loader, graph=graph)
    print(f'test dataset:\naccuracy: {test_acc:.4f}, AU-PR: {test_pr:.4f}, AU-ROC: {test_roc:.4f}')
    print(OUTNAME)
    
    # plot performance curves
    metrics = evaluate_performance(train_metrics, val_metrics, all_true, all_pred, OUTNAME)

    # save predictions
    # all_info = all_info[['substrateID','product','index']]
    test_dataset = all_info.with_columns(
            true = pl.Series(all_true.flatten().numpy()),
            pred_0 = pl.Series(all_pred[:,0].flatten().numpy()),
            pred_1 = pl.Series(all_pred[:,1].flatten().numpy()),
            pred_2 = pl.Series(all_pred[:,2].flatten().numpy()),
            pred_3 = pl.Series(all_pred[:,3].flatten().numpy()),
            pred_4 = pl.Series(all_pred[:,4].flatten().numpy())
            )#.unique()

    test_dataset.write_csv(str('results/'+OUTNAME+'/prediction.csv'))
    
    # ----- OUTPUT -----
    if not PRED_ONLY:
        # training history
        metrics.to_csv(str('results/'+OUTNAME+'/metrics.csv'), index=False)