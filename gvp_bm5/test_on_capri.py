import torch
import torch_geometric as tg
import os, sys, glob, tqdm
import csv
import numpy as np
from sklearn import metrics
from atom3d_datasets import LMDBDataset
import gvp_bm5 as m
torch.manual_seed(0)


nw = 8
bsz = 16
device = 'cuda:2' 


for foldnum in range(1,11):
    
    if os.path.exists('test_predictions_bm5/fold%d.csv'%foldnum):

        print('===================== fold %d ==================='%foldnum)

        capri_dataset = LMDBDataset('../_gvp_capri/datasets', transform = m.Transform())
        capri_dataloader = tg.loader.DataLoader(capri_dataset, num_workers=nw, batch_size=bsz, shuffle=False)

        #--------------------------------------------------------------------------------------------#
        valloss_dict = {}
        with open('results/fold%d.csv'%foldnum, 'r') as f:
            r = csv.reader(f)
            for line in r:
                valloss_dict[int(line[0])] = float(line[2])

        min_valloss = min([valloss_dict[key] for key in valloss_dict.keys()])    
        min_valloss_epoch = [key for key in valloss_dict.keys() if valloss_dict[key] == min_valloss][0]    
        print('min_valloss_epoch', min_valloss_epoch)
        #--------------------------------------------------------------------------------------------#

        model = m.Model().to(device)
        model.load_state_dict(torch.load('model_dir/f%d_%d.pt'%(foldnum, min_valloss_epoch), map_location=device))

        model.eval()
        with torch.no_grad():        
            t = tqdm.tqdm(capri_dataloader)
            ids, preds, labels = [], [], []
            for batch in t:
                pred = model(batch.to(device))
                label = batch.label.type(torch.LongTensor).to(device)
                ids.extend([x.split('/')[-1] for x in list(batch.id)])
                labels.extend(list(label.cpu().detach().numpy()))
                preds.extend(list(pred.cpu().detach().numpy()))

        with open('test_predictions_capri/fold%d.csv'%foldnum, 'w') as fw:
            for i, j, k in zip(ids, labels, preds):
                fw.write('%s,%d,%.6f\n'%(i,j,k))

