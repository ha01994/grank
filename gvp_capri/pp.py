import os, sys, glob, random, csv
import atom3d_datasets as da
import numpy as np
from random import randint
import pickle



binary_dic = {}
with open('../drgnn/results_capri.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        if float(line[5]) < 0.3: 
            binary_dic[line[2]+'.pdb'] = 0
        else: 
            binary_dic[line[2]+'.pdb'] = 1
print(len(binary_dic.keys()))            


'''
print('capri target,#all,#pos,#neg')

targets_total = []
for i in [29, 30, 32, 35, 37, 39, 40, 41, 46, 47, 50, 53, 54]:    

    targets = []    
    zz = [f for f in binary_dic.keys() if 'T%d_'%i in f]
    for file in zz:
        targets.append(binary_dic[file])
        targets_total.append(binary_dic[file])
    
    n_pos = sum(targets)
    n_neg = len(targets) - sum(targets)
    print('Target%d,%d,%d,%d'%(i,len(targets),n_pos,n_neg))

n_pos = sum(targets_total)
n_neg = len(targets_total) - sum(targets_total)
print('TOTAL,%d,%d,%d'%(len(targets_total),n_pos,n_neg))

exit()
'''



datasets_dir = 'datasets'
input_dir = 'input_hdf5_rotated_itf'

output_path = datasets_dir
os.system('rm -rf %s'%(output_path))
os.system('mkdir %s'%(output_path))

file_list = []
for i in [29, 30, 32, 35, 37, 39, 40, 41, 46, 47, 50, 53, 54]:    
    print(i)

    zz = [f for f in binary_dic.keys() if 'T%d_'%i in f]

    for file in zz:
        #interface 인식 안되는 경우 없음
        #if os.path.exists('/home/ha01994/struct_pred/deeprank_capri/%s/T%d/%s'%(input_dir, i, file)):
        if True:
            file_list.append('/home/ha01994/struct_pred/deeprank_capri/%s/T%d/%s'%(input_dir, i, file))

dataset = da.load_dataset(file_list, 'pdb')
da.make_lmdb_dataset(dataset, output_path, binary_dic)






