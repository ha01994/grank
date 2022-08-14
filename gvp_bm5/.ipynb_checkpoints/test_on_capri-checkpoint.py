import torch
import torch_geometric as tg
import os, sys, glob, tqdm
import csv
import numpy as np
from sklearn import metrics
from atom3d_datasets import LMDBDataset
import gvp_bm5 as m
torch.manual_seed(0)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import scipy.stats as stats


nw = 8
bsz = 16
device_num = 1
device = 'cuda:%d'%device_num if torch.cuda.is_available() else 'cpu'



for foldnum in range(1,11):
    print('===================== fold %d ==================='%foldnum)
    
    z = [x for x in glob.glob('test_results_rttonly/*') if '.csv' in x and 'f%d_'%foldnum in x][0]
    epoch = int(z.split('/')[-1].split('.')[0].split('_')[1])
    print(epoch)
    
    model_path = './model_dir_rttonly/f%d_%d.pt'%(foldnum,epoch)
    print(model_path)

    capri_dataset = LMDBDataset('../gvp_capri/datasets', transform = m.Transform())
    capri_dataloader = tg.loader.DataLoader(capri_dataset, num_workers=nw, batch_size=bsz, shuffle=False)

    model = m.Model().to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        ids, preds, labels = [], [], []
        t = tqdm.tqdm(capri_dataloader)
        for batch in t:
            pred = model(batch.to(device))
            label = batch.label.type(torch.LongTensor).to(device)
            ids.extend([x.split('/')[-1] for x in list(batch.id)])
            labels.extend(list(label.cpu().detach().numpy()))
            preds.extend(list(pred.cpu().detach().numpy()))

    capri_roc_auc = metrics.roc_auc_score(labels, preds)
    print('capri_roc_auc: %.3f'%capri_roc_auc)

    with open('test_predictions_capri/fold%d.csv'%foldnum, 'w') as fw:
        for i, j, k in zip(ids, labels, preds):
            fw.write('%s,%d,%.6f\n'%(i,j,k))

