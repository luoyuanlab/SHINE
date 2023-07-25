import math
from torch import nn
from layers import HGAT_sparse, HGNN_fc, weighted_sum, masked_sum, HGNN_sg_attn
import utils.hg_ops as hgo
import torch.nn.functional as F
import torch
import time
import copy
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torch.nn.parameter import Parameter

class SHINEreg(nn.Module):
    def __init__(self, H, yuniques, in_ch_n, train_idx, val_idx, test_idx, n_hid, dcf, G=None, dropout=0.5, fn=None, seed=0, atype='additive', metric='f1', fc_dropout=0.5, dataset='MC3', threshold=0.5, jk=False): 
        super(SHINEreg, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        HT = H.T
        if G is None:
            self.G = torch.mm(H, HT) 
        else:
            self.G = G
            
        self.e_degs = H.sum(0)
        self.n_degs = H.sum(1)

        self.HTa = HT / HT.sum(1, keepdim=True)
            
        self.pair = HT.nonzero(as_tuple=False).t()
        
        self.train_idx = torch.Tensor(train_idx).long()
        self.val_idx = torch.Tensor(val_idx).long()
        self.test_idx = torch.Tensor(test_idx).long()
        self.fn = fn
        self.yuniques = yuniques
        n_class = len(yuniques)

        self.hgc1 = HGAT_sparse(in_ch_n, n_hid, dropout=dropout, alpha=0.2, transfer = True, bias = True, concat=False) 
        self.hgc2 = HGAT_sparse(n_hid, n_hid, dropout=dropout, alpha=0.2, transfer = True, bias = True, concat=False) 

        self.sga_dropout = nn.Dropout(dropout)
        self.jk = jk # jumping knowledge
        if self.jk:
            sg_hid = n_hid *2
        else:
            sg_hid = n_hid
        self.sga = HGNN_sg_attn(sg_hid, sg_hid, atype)
        l_hid = 2*sg_hid // 3

        self.fc = HGNN_fc(sg_hid+dcf, l_hid)
        self.fc2 = HGNN_fc(l_hid, l_hid)
        self.fc3 = HGNN_fc(l_hid, n_class) 
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2_dropout = nn.Dropout(fc_dropout)        
        self.metric = metric
        
        self.report = defaultdict(list)
        self.dataset = dataset
        self.threshold = threshold
        
        self.a = nn.Parameter(torch.zeros(size=(n_hid, 1)))   
        self.a2 = nn.Parameter(torch.zeros(size=(n_hid, 1)))  
        
        stdv = 1. / math.sqrt(n_hid)
        self.a.data.uniform_(-stdv, stdv)
        self.a2.data.uniform_(-stdv, stdv)        

    def forward(self, x, xe, sgs, cf=None): 
        x1, xe = self.hgc1(x, xe, self.pair, self.a) 
        x, xe = self.hgc2(x1, xe, self.pair, self.a2) 
        
        if self.jk:
            x = torch.cat((x, x1), 1) 

        x2 = x / torch.norm(x, p=2, dim=1, keepdim=True)
        xxt = torch.mm(x2, x2.T) 
        node_loss = ( (2 -2*xxt)*self.G ).sum() / self.G.shape[0]
        
        # attention layer to calcualte subgraph representation
        xsg = self.sga(x, sgs)
        xsg = self.sga_dropout(xsg)
        
        if cf is None:
            x = F.elu(self.fc(xsg))
        else:
            x = F.elu(self.fc(torch.cat([xsg, cf], 1)))

        x = self.fc_dropout(x)
        x = F.elu(self.fc2(x))
        x = self.fc2_dropout(x)
        x = self.fc3(x)
        return x, node_loss, xsg

    def to(self, device):
        self.pair = self.pair.to(device)
        self.HTa = self.HTa.to(device)
        
        self.train_idx = self.train_idx.to(device)
        self.val_idx = self.val_idx.to(device)
        self.test_idx = self.test_idx.to(device)
        return super(SHINEreg, self).to(device)
    
    def fit(self, x, sgs, cf, y, cls_loss, nratio, optimizer, scheduler, num_epochs=25, print_freq=500): 
        # sgs, each sg (subgraph) is a list of node ids in x
        # cf - confounding variables for future extension
        since = time.time()

        best_model_wts = copy.deepcopy(self.state_dict())
        best_val_score = 0.0

        xe = self.HTa.mm(x) 

        for epoch in range(num_epochs):
            if epoch % print_freq == 0:
                print('-' * 20)
                print(f'Epoch {epoch}/{num_epochs - 1}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.train()  
                else:
                    self.eval()  
                    
                idx = self.train_idx if phase == 'train' else self.val_idx

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if cf is None:
                        cf_sel = None
                    else:
                        cf_sel = cf.index_select(0,idx)
                    
                    outputs, node_loss, xsg = self.forward(x,
                                                      xe,
                                                      sgs.index_select(0,idx),
                                                      cf_sel)
                    clf_loss = cls_loss(outputs, y[idx]) 
                    loss = clf_loss / len(idx) + nratio*node_loss
                    if self.dataset == 'MC3':
                        _, preds = torch.max(outputs, 1)
                    elif self.dataset == 'disgenet':
                        preds = 1*(outputs > self.threshold)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=1)
                        grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in self.parameters()]))                        
                        self.report['grad_norm'].append(grad_norm.detach().cpu().numpy())
                        optimizer.step()

                # statistics
                y_phase_pred = preds.detach().cpu().numpy()
                y_phase = y[idx].detach().cpu().numpy()
                if self.dataset == 'MC3':
                    epoch_cm = confusion_matrix(y_phase, y_phase_pred)
                elif self.dataset == 'disgenet':
                    epoch_cm = multilabel_confusion_matrix(y_phase, y_phase_pred)

                epoch_loss = loss.item() 
                if self.metric == 'f1':
                    epoch_score = f1_score(y_phase, y_phase_pred, average='micro')
                elif self.metric == 'acc':
                    epoch_score = accuracy_score(y_phase, y_phase_pred)
                else:
                    sys.exit(f'unsupported metric {self.metric}')
                    
                self.report['epoch'].append(epoch)
                if phase == 'train':
                    self.report['train_loss'].append(epoch_loss)
                    self.report['train_score'].append(epoch_score)
                    y_tr_pred, y_tr = y_phase_pred, y_phase
                    train_cm = epoch_cm
                    train_loss = epoch_loss
                    train_score = epoch_score
                    train_xsg = xsg
                else:
                    self.report['val_loss'].append(epoch_loss)
                    self.report['val_score'].append(epoch_score)
                    val_xsg = xsg
                    scheduler.step(epoch_loss)
                

                if epoch % print_freq == 0:
                    print(f'{phase} Loss: {epoch_loss:.4f}, clf_loss: {clf_loss.item()/len(idx):.4f}, node_loss: {node_loss.item():.4f}, {self.metric}: {epoch_score:.4f}') 
                # deep copy the model
                if phase == 'val' and epoch_score > best_val_score:
                    best_epoch = epoch+1
                    best_train_score = train_score
                    best_train_loss =  train_loss
                    best_train_cm = train_cm
                    best_val_score = epoch_score
                    best_val_loss = epoch_loss
                    best_val_cm = epoch_cm
                    best_model_wts = copy.deepcopy(self.state_dict())
                    best_train_xsg = train_xsg
                    best_val_xsg = val_xsg
                    if cf is None:
                        cf_sel = None
                    else:
                        cf_sel = cf.index_select(0,self.test_idx)
                    
                    pred, prob, test_xsg = self.predict(x,
                                              xe, 
                                              sgs.index_select(0,self.test_idx),
                                              cf_sel)
                    test_loss = cls_loss(prob, y[self.test_idx]) 
                    test_loss = test_loss.item() / len(self.test_idx)
                    best_y_tr_pred, best_y_tr = y_tr_pred, y_tr
                    y_val_pred, y_val = y_phase_pred, y_phase
                    y_test_pred = pred.detach().cpu().numpy()
                    y_test = y[self.test_idx].detach().cpu().numpy()
                    if self.metric == 'f1':
                        test_score = f1_score(y_test, y_test_pred, average='micro')
                    elif self.metric == 'acc':
                        test_score = accuracy_score(y_test, y_test_pred)
                    else:
                        sys.exit(f'unsupported metric {self.metric}')

                    if self.dataset == 'MC3':
                        test_cm = confusion_matrix(y_test, y_test_pred)
                    elif self.dataset == 'disgenet':
                        test_cm = multilabel_confusion_matrix(y_test, y_test_pred)
                    
                    print(f'Updating val {self.metric}: {best_val_score:4f}; test {self.metric}: {test_score:4f}')
                    print(f'{self.yuniques}')
                    print(f'{train_cm}')
                    print(f'{epoch_cm}')
                    print(f'{test_cm}')


        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # save checkpoint
        if self.fn is not None:
            torch.save({'epoch': best_epoch,
                        'state_dict': self.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'yuniques': self.yuniques,
                        'best_train_score': best_train_score,
                        'best_train_loss': best_train_loss,
                        'best_train_cm': best_train_cm,
                        'best_train_xsg': best_train_xsg,
                        'best_val_score': best_val_score,
                        'best_val_loss': best_val_loss,
                        'best_val_cm': best_val_cm,
                        'best_val_xsg': best_val_xsg,
                        'test_score': test_score,
                        'test_loss': test_loss,
                        'test_cm': test_cm,
                        'test_xsg': test_xsg,
                        'y_test': y_test,
                        'y_test_pred': y_test_pred,
                        'y_val': y_val,
                        'y_val_pred': y_val_pred,
                        'y_tr': best_y_tr,
                        'y_tr_pred': best_y_tr_pred,
                        'report': self.report,
            }, f'{self.fn}.ckpt')

        fig = plt.figure()
        plt.plot(self.report['train_loss'], label='Train loss')
        plt.plot(self.report['val_loss'], label='Val loss')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_loss.pdf', bbox_inches='tight')
        plt.close()
        
        fig = plt.figure()
        plt.plot(self.report['train_score'], label=f'Train {self.metric}')
        plt.plot(self.report['val_score'], label=f'Val {self.metric}')
        plt.legend()
        plt.grid()
        plt.show()
        fig.savefig(f'{self.fn}_{self.metric}.pdf', bbox_inches='tight')
        plt.close()
        
        # load best model weights
        self.load_state_dict(best_model_wts)
        return self

    def predict(self, x, xe, sgs, cf):   
        self.eval()  # set to evaluate mode

        outputs, node_loss, xsg = self.forward(x, xe, sgs, cf)
        if self.dataset == 'MC3':
            _, preds = torch.max(outputs, 1)
        elif self.dataset == 'disgenet':
            preds = 1*(outputs > self.threshold)        

        return preds, outputs, xsg

    def show_report(self):
        return pd.DataFrame(self.report)    
