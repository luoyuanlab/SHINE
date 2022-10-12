import os
import random
import numpy as np
import copy
import torch
import sys
import torch.optim as optim
import pprint as pp
import pandas as pd
import utils.hg_ops as hgo
import pickle as pkl
from models import SHyGNN
from config import get_config
import getopt

if len(sys.argv) < 2:
    sys.exit("Use: python precompute_G.py -c config/mc3g/c2cp/subj_edge/config.mc3g.c2cp.subj_edge.shygnn200.yaml")


opts, extraparams = getopt.getopt(sys.argv[1:], 's:c:',
                                  ['seed=', 'config='])

for o,p in opts:
    if o in ['-s', '--seed']:
        seed = int(p)
    if o in ['-c', '--config']:
        fncfg = p

cfg = get_config(fncfg)

ddn = cfg['data_root']
fn_H = cfg['fn_H']
fn_G = cfg['fn_G']
fn_m = cfg['fn_m']
fn_train = cfg['fn_train']
dataset = cfg['on_dataset']
use_subj_edge = cfg['use_subj_edge']
tfidf_H = cfg['tfidf_H']
if 'transductive' in cfg:
    transductive = cfg['transductive']
else:
    transductive = False


train_idx = pd.read_csv(fn_train, header=None)[0] 

f = open(fn_m, 'rb')
if dataset == 'MC3':
    [m, cf, y] = pkl.load(f)
elif dataset == 'disgenet':
    [m, y] = pkl.load(f)
else:
    sys.exit(f'unrecognized dataset {dataset}')
    
f.close()

print(f'Computing G...')
if transductive:
    if dataset == 'MC3':
        H, gene_idx, subj_idx, cf, y, yuniques = hgo.construct_Hexp(fn_H, fn_m, dataset)
    elif dataset == 'disgenet':
        H, gene_idx, subj_idx, y, yuniques = hgo.construct_Hexp(fn_H, fn_m, dataset)
elif use_subj_edge:
    H, pathway_idx, subj_idx = hgo.construct_Hexp_inductive(fn_H, m.iloc[train_idx])
else:
    H = hgo.construct_H(fn_H, tfidf = tfidf_H)
    
G = hgo.generate_G_from_H(H)

f = open(fn_G, 'wb')

if transductive:
    if dataset == 'MC3':
        pkl.dump([G, gene_idx, subj_idx, cf, y, yuniques], f, -1)
    elif dataset == 'disgenet':
        pkl.dump([G, gene_idx, subj_idx, y, yuniques], f, -1)
elif use_subj_edge:
    pkl.dump([G, pathway_idx, subj_idx], f, -1)
else:
    pkl.dump([G], f, -1)
f.close()
