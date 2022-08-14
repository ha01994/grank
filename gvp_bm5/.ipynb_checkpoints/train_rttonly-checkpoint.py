import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import os, sys, glob, tqdm
import numpy as np
import datetime
import scipy.stats as stats
from atom3d_datasets import LMDBDataset
import gvp_bm5 as m
from utils import *
from sklearn import metrics
import csv
torch.manual_seed(0)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


pdbs = [x.split('/')[-1].split('.')[0] for x in glob.glob('/home/ha01994/struct_pred/deeprank_bm5/refs/*')]
pdbs.sort()

fnat_binary_dic = {}
for pdb in pdbs:
    with open(os.path.join('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models/%s/Fnat.dat'%pdb), 'r') as f:
        r = csv.reader(f, delimiter=' ')
        next(r)
        for line in r:
            if float(line[-1]) < 0.3: fnat_binary_dic[line[0]] = 0
            else: fnat_binary_dic[line[0]] = 1

                

nw = 8
bsz = 16
lr = 1e-4


foldnums = range(1,11)
device_num = 0
device = 'cuda:%d'%device_num if torch.cuda.is_available() else 'cpu'

            

for foldnum in foldnums:
    print('======================== FOLD %d ======================='%foldnum)
    
    train_dataset = LMDBDataset('datasets_rttonly/fold%d/train/'%foldnum, transform = m.Transform())
    val_dataset = LMDBDataset('datasets_rttonly/fold%d/valid/'%foldnum, transform = m.Transform())
    test_dataset = LMDBDataset('datasets_rttonly/test/', transform=m.Transform())

    #print('len(train_dataset)', len(train_dataset))
    #print('len(val_dataset)', len(val_dataset))
    #print('len(test_dataset)', len(test_dataset))
    

    train_dataloader = tg.loader.DataLoader(train_dataset, num_workers=nw, batch_size=bsz, shuffle=True)
    val_dataloader = tg.loader.DataLoader(val_dataset, num_workers=nw, batch_size=bsz, shuffle=False)
    test_dataloader = tg.loader.DataLoader(test_dataset, num_workers=nw, batch_size=bsz, shuffle=False)

    model = m.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    loss_fn = nn.MSELoss()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', pytorch_total_params)

        
    #==============================================================================================#
    valloss_dict = {}
    for epoch in range(10):
        print('-----------epoch %d-----------'%epoch)
        total_loss, total_count = 0, 0

        for batch in train_dataloader:        
            model.train()
            optimizer.zero_grad()        
            pred = model(batch.to(device))
            label = batch.label.type(torch.FloatTensor).to(device)        
            loss = loss_fn(pred, label)
            total_loss += float(loss)
            total_count += 1
            loss.backward()
            optimizer.step()

        print('train_loss: %.5f'%(total_loss / total_count))    
        torch.save(model.state_dict(), "model_dir_rttonly/f%d_%d.pt"%(foldnum, epoch))

        model.eval()
        with torch.no_grad():
            preds, labels = [], []
            for batch in val_dataloader:
                pred = model(batch.to(device))
                label = batch.label.type(torch.FloatTensor).to(device)                
                preds.extend(list(pred.cpu().detach().numpy()))
                labels.extend(list(label.cpu().detach().numpy()))                   

            val_loss = loss_fn(torch.from_numpy(np.array(preds)).to(device), 
                               torch.from_numpy(np.array(labels)).to(device))
            print('val_loss: %.5f'%val_loss)
            valloss_dict[epoch] = val_loss
    
    
    min_valloss = min([valloss_dict[key] for key in valloss_dict.keys()])    
    min_valloss_epoch = [key for key in valloss_dict.keys() if valloss_dict[key] == min_valloss][0]    
    print('min_valloss_epoch', min_valloss_epoch)
    #==============================================================================================#
    
    
    
    model.load_state_dict(torch.load('model_dir_rttonly/f%d_%d.pt'%(foldnum, min_valloss_epoch)))
        
    model.eval()
    with torch.no_grad():
        ids, preds, labels = [], [], []
        for batch in test_dataloader:
            pred = model(batch.to(device))
            label = batch.label.type(torch.FloatTensor).to(device)
            ids.extend([x.split('/')[-1] for x in list(batch.id)])
            preds.extend(list(pred.cpu().detach().numpy()))
            labels.extend(list(label.cpu().detach().numpy()))            
    
    binary_labels = [fnat_binary_dic[f] for f in ids]
    test_roc_auc = metrics.roc_auc_score(binary_labels, preds)    
    print('test_roc_auc: %.3f'%test_roc_auc)
    
    
    with open('test_results_rttonly/f%d_%d.csv'%(foldnum, min_valloss_epoch), 'w') as fw:
        for i,j,k in zip(ids, binary_labels, preds):
            fw.write('%s,%d,%.6f\n'%(i,j,k))
    


