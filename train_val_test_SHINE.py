import os
import random
import numpy as np
import copy
import torch
import sys
import time
import torch.optim as optim
import pprint as pp
import pandas as pd
import utils.hg_ops as hgo
import pickle as pkl
from models import SHINE
from config import get_config
import getopt

if len(sys.argv) < 2:
    sys.exit("Use: python train_val_test_SHINE.py result.csv -c config/config.tcga.SHINE.yaml")

opts, extraparams = getopt.getopt(sys.argv[2:], 's:c:',
                                  ['seed=', 'config='])

for o,p in opts:
    if o in ['-s', '--seed']:
        seed = int(p)
    if o in ['-c', '--config']:
        fncfg = p

fnres = sys.argv[1]


cfg = get_config(fncfg)

ddn = cfg['data_root']
fn_H = cfg['fn_H']
fn_G = cfg['fn_G']
fn_m = cfg['fn_m']
fn_train = cfg['fn_train']
fn_val = cfg['fn_val']
fn_test = cfg['fn_test']
devstr = cfg['devstr']
dataset = cfg['on_dataset']
rdn = cfg['result_root']
lr_list = cfg['lr']
weight_decay_list = cfg['weight_decay']
gamma_list = cfg['gamma']
n_hid_list = cfg['n_hid']
patience_list=cfg['patience']
atype=cfg['attention_type']
use_subj_edge = cfg['use_subj_edge']
tfidf_H = cfg['tfidf_H']
use_knn = cfg['use_knn']
jk = cfg['jk']
if 'metric' in cfg:
    metric = cfg['metric']
else:
    metric = 'f1'

if 'seed' in cfg:
    seeds = cfg['seed']
else:
    seeds = [0]


train_idx = pd.read_csv(fn_train, header=None)[0] 
val_idx = pd.read_csv(fn_val, header=None)[0]
test_idx = pd.read_csv(fn_test, header=None)[0]

f = open(fn_m, 'rb')
if dataset == 'MC3':
    [m, cf, y] = pkl.load(f)
    y, yuniques = pd.factorize(y, sort=True)
    cf = None 
elif dataset == 'disgenet':
    [m, y] = pkl.load(f)
    yuniques = y.columns.values
    y = y.values
    cf = None
else:
    sys.exit(f'unrecognized dataset {dataset}')
f.close()


since = time.time()
if use_knn:
    H, pathway_idx, knn_idx = hgo.construct_Hexp_KNN_inductive(fn_H, m.iloc[train_idx])
elif use_subj_edge:
    H, pathway_idx, subj_idx = hgo.construct_Hexp_inductive(fn_H, m.iloc[train_idx], tfidf = tfidf_H)
else:
    H = hgo.construct_H(fn_H, tfidf = tfidf_H)
time_elapsed = time.time() - since
print(f'Constructing H complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
nfts = np.identity(H.shape[0]) 
efts = np.identity(H.shape[1]) 
device = torch.device(devstr)

H = torch.Tensor(np.array(H)).float().to(device)
y = torch.Tensor(y).squeeze().long().to(device)
m = torch.Tensor(np.array(m)).float().to(device)
if cf is None:
    dcf = 0
else:
    cf = torch.Tensor(np.array(cf)).float().to(device)
    dcf = cf.shape[1]
nfts = torch.Tensor(nfts).to(device)
efts = torch.Tensor(efts).to(device)

def _main():
    print(f"Classification on {cfg['on_dataset']} dataset!!! classes: {yuniques}")
    print('Configuration -> Start')
    pp.pprint(cfg)
    print('Configuration -> End')

    print(f'train_idx.shape: {train_idx.shape}')
    
    fres = open(f'{rdn}/{fnres}', 'w')
    fres.write(f'n_hid,lr,weight_decay,gamma,patience,seed,best_train_{metric},best_val_{metric},test_{metric},best_train_loss,best_val_loss,test_loss,best_train_cm,best_val_cm,test_cm\n')
    for n_hid in n_hid_list:
        for lr in lr_list:
            for gamma in gamma_list:
                for patience in patience_list:
                    for weight_decay in weight_decay_list:
                        for seed in seeds:
                            fn = f'{rdn}/nhid{n_hid}_lr{lr}_wd{weight_decay}_gamma{gamma}_p{patience}_s{seed}'
                            if os.path.isfile(f'{fn}.ckpt'):
                                print(f'Using existing {fn}.ckpt')
                            else:
                                np.random.seed(123456789)
                                model = SHINE(H = H,
                                               yuniques = yuniques,
                                               in_ch_n = nfts.shape[1],
                                               train_idx = train_idx,
                                               val_idx = val_idx,
                                               test_idx = test_idx,
                                               n_hid = n_hid,
                                               jk = jk,
                                               dcf = dcf,
                                               dropout = cfg['drop_out'],
                                               atype = atype,
                                               fn = fn,
                                               metric = metric,
                                               seed = seed,
                                               dataset = dataset)
                                model = model.to(device)

                                optimizer = optim.Adam(model.parameters(), lr=lr,
                                                       weight_decay=weight_decay)
                                schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                 factor=gamma,
                                                                                 patience=patience)
                                if dataset == 'MC3':
                                    cls_loss = torch.nn.CrossEntropyLoss(reduction='sum')
                                elif dataset == 'disgenet':
                                    cls_loss = torch.nn.MultiLabelSoftMarginLoss(reduction='sum') 

                                model = model.fit(nfts, m, cf, y, cls_loss, optimizer, schedular, cfg['max_epoch'], print_freq=cfg['print_freq']) 
                            ckpt = torch.load(f'{fn}.ckpt')
                            best_train_score = ckpt['best_train_score']
                            best_train_loss = ckpt['best_train_loss']
                            best_train_cm = ckpt['best_train_cm']
                            best_val_score = ckpt['best_val_score']
                            best_val_loss = ckpt['best_val_loss']
                            best_val_cm = ckpt['best_val_cm']
                            test_score = ckpt['test_score']
                            test_loss = ckpt['test_loss']
                            test_cm = ckpt['test_cm']
                            fres.write(f'{n_hid},{lr},{weight_decay},{gamma},{patience},{seed},{best_train_score},{best_val_score},{test_score},{best_train_loss},{best_val_loss},{test_loss},\"{best_train_cm}\",\"{best_val_cm}\",\"{test_cm}\"\n')
                            print(f'nhid: {n_hid}, lr: {lr}, wd: {weight_decay}, gamma: {gamma}, p: {patience}, s: {seed}')
                            print(f'Best train {metric}: {best_train_score:4f}, val {metric}: {best_val_score:4f}, test {metric}: {test_score:4f}')

    fres.close()
    
if __name__ == '__main__':
    _main()


