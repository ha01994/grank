import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import os, sys, glob, tqdm
import numpy as np
from atom3d_datasets import LMDBDataset
import gvp_bm5 as m
from sklearn import metrics
import time, csv
torch.manual_seed(0)



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
bsz = 32
lr = 1e-4
epochs = 20


k = int(sys.argv[1])

if k == 0:
    foldnums = [1,2,3,4,5]
    device_num = 0
if k == 1:
    foldnums = [6,7,8,9,10]
    device_num = 1

    
device = 'cuda:%d'%device_num if torch.cuda.is_available() else 'cpu'


for foldnum in foldnums:
    
    start_time = time.time()
    
    print('==================================== FOLD %d ===================================='%foldnum)
    
    train_dataset = LMDBDataset('datasets/fold%d/train/'%foldnum, transform = m.Transform())
    val_dataset = LMDBDataset('datasets/fold%d/valid/'%foldnum, transform = m.Transform())
    test_dataset = LMDBDataset('datasets/test/', transform=m.Transform())

    print('len(train_dataset)', len(train_dataset))
    print('len(val_dataset)', len(val_dataset))
    print('len(test_dataset)', len(test_dataset))    

    train_dataloader = tg.loader.DataLoader(train_dataset, num_workers=nw, batch_size=bsz, shuffle=True)
    val_dataloader = tg.loader.DataLoader(val_dataset, num_workers=nw, batch_size=bsz, shuffle=False)
    test_dataloader = tg.loader.DataLoader(test_dataset, num_workers=nw, batch_size=bsz, shuffle=False)

    model = m.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)    
    loss_fn = nn.MSELoss()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', pytorch_total_params)
        
    #======================================================================================#
    
    for epoch in range(epochs):
        print('-----------epoch %d-----------'%epoch)
        total_loss, total_count = 0, 0
        train_tqdm = tqdm.tqdm(train_dataloader)
        
        for batch in train_tqdm:                    
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
        torch.save(model.state_dict(), "model_dir/f%d_%d.pt"%(foldnum, epoch))
        
        train_loss = total_loss / total_count

        model.eval()
        with torch.no_grad():
            val_tqdm = tqdm.tqdm(val_dataloader)
            
            ids, preds, labels = [], [], []
            for batch in val_tqdm:
                pred = model(batch.to(device))
                label = batch.label.type(torch.FloatTensor).to(device)     
                ids.extend([x.split('/')[-1] for x in list(batch.id)])
                preds.extend(list(pred.cpu().detach().numpy()))
                labels.extend(list(label.cpu().detach().numpy()))                   

            val_loss = loss_fn(torch.from_numpy(np.array(preds)).to(device), 
                               torch.from_numpy(np.array(labels)).to(device))
            
            binary_labels = [fnat_binary_dic[f] for f in ids]
            val_roc_auc = metrics.roc_auc_score(binary_labels, preds)    
            print('val_loss: %.5f'%val_loss)
            print('val_roc_auc: %.3f'%val_roc_auc)
            
        with open('results/fold%d.csv'%foldnum, 'a') as fw:
            fw.write('%d,%.5f,%.5f,%.3f\n'%(epoch, train_loss, val_loss, val_roc_auc))
    
    #======================================================================================#
    
    end_time = time.time()
    
    minutes = (end_time - start_time) / 60    
    minutes = minutes / epochs 
    
    with open('minutes_took_per_epoch.csv', 'a') as fw:
        fw.write('fold%d,%.3f\n'%(foldnum, minutes))
        
        
    #################################################################################################
    
    valloss_dict = {}
    with open('results/fold%d.csv'%foldnum, 'r') as f:
        r = csv.reader(f)
        for line in r:
            valloss_dict[int(line[0])] = float(line[2])
    
    min_valloss = min([valloss_dict[key] for key in valloss_dict.keys()])    
    min_valloss_epoch = [key for key in valloss_dict.keys() if valloss_dict[key] == min_valloss][0]    
    print('min_valloss_epoch', min_valloss_epoch)
        
    
    model.load_state_dict(torch.load('model_dir/f%d_%d.pt'%(foldnum, min_valloss_epoch)))

    model.eval()
    with torch.no_grad():        
        t = tqdm.tqdm(test_dataloader)
        
        ids, preds, labels = [], [], []
        for batch in t:
            pred = model(batch.to(device))
            label = batch.label.type(torch.FloatTensor).to(device)
            ids.extend([x.split('/')[-1] for x in list(batch.id)])
            labels.extend(list(label.cpu().detach().numpy()))
            preds.extend(list(pred.cpu().detach().numpy()))
            
    binary_labels = [fnat_binary_dic[f] for f in ids]

    with open('test_predictions_bm5/fold%d.csv'%foldnum, 'w') as fw:
        for i, j, k in zip(ids, binary_labels, preds):
            fw.write('%s,%d,%.6f\n'%(i,j,k))


