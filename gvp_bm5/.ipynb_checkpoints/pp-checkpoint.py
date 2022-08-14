import os, sys, glob, random, csv
import atom3d_datasets as da
import numpy as np
from random import randint
import pickle


pdbs = [x.split('/')[-1].split('.')[0] for x in glob.glob('/home/ha01994/struct_pred/deeprank_bm5/refs/*')]
pdbs.sort()
print(len(pdbs))

fnat_cont_dic = {}
for pdb in pdbs:
    with open(os.path.join('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models/%s/Fnat.dat'%pdb), 'r') as f:
        r = csv.reader(f, delimiter=' '); next(r)
        for line in r:              
            fnat_cont_dic[line[0]] = float(line[-1])

fnat_binary_dic = {}
for pdb in pdbs:
    with open(os.path.join('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models/%s/Fnat.dat'%pdb), 'r') as f:
        r = csv.reader(f, delimiter=' '); next(r)
        for line in r:              
            if float(line[-1]) < 0.3: fnat_binary_dic[line[0]] = 0
            else: fnat_binary_dic[line[0]] = 1


                
                
                
datasets_dir = 'datasets'
    
    
    

k = int(sys.argv[1]) #1, 2, ..., 10, 11




if k == 11:

    print('test')

    output_path = os.path.join(datasets_dir, 'test/')
    os.system('rm -rf %s'%output_path)
    os.system('mkdir %s'%output_path)

    file_list = []
    with open('../drgnn/FILES/test.csv', 'r') as f:
        r = csv.reader(f)
        for line in r:
            file = line[0]
            pdb = file.split('_')[0]

            if os.path.exists('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models_itf/%s/%s'%(pdb, file)):
                file_list.append('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models_itf/%s/%s'%(pdb, file))

            elif os.path.exists('/home/ha01994/struct_pred/drgnn/input_docking_models_itf/%s/%s'%(pdb, file)):
                file_list.append('/home/ha01994/struct_pred/drgnn/input_docking_models_itf/%s/%s'%(pdb, file))

            elif os.path.exists('/home/ha01994/struct_pred/drgnn/input_docking_models_2_itf/%s/%s'%(pdb, file)):
                file_list.append('/home/ha01994/struct_pred/drgnn/input_docking_models_2_itf/%s/%s'%(pdb, file))

    print(len(file_list))
    dataset = da.load_dataset(file_list, 'pdb')
    da.make_lmdb_dataset(dataset, output_path, fnat_cont_dic)



    

else:
    for foldnum in [k]:
        
        print('FOLD %d'%foldnum)

        os.system('rm -rf %s/fold%d/'%(datasets_dir,foldnum))
        os.system('mkdir %s/fold%d/'%(datasets_dir,foldnum))

        for split in ['train', 'valid']:
            print(split)

            output_path = '%s/fold%d/%s/'%(datasets_dir, foldnum, split)
            os.system('mkdir %s'%output_path)

            file_list = []
            with open('../drgnn/FILES/fold%d_%s.csv'%(foldnum,split), 'r') as f:
                r = csv.reader(f)
                for line in r:
                    file = line[0]
                    pdb = file.split('_')[0]    

                    if os.path.exists('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models_itf/%s/%s'%(pdb, file)):
                        file_list.append('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models_itf/%s/%s'%(pdb, file))

                    elif os.path.exists('/home/ha01994/struct_pred/drgnn/input_docking_models_itf/%s/%s'%(pdb, file)):
                        file_list.append('/home/ha01994/struct_pred/drgnn/input_docking_models_itf/%s/%s'%(pdb, file))

                    elif os.path.exists('/home/ha01994/struct_pred/drgnn/input_docking_models_2_itf/%s/%s'%(pdb, file)):
                        file_list.append('/home/ha01994/struct_pred/drgnn/input_docking_models_2_itf/%s/%s'%(pdb, file))

            print('fold%d,%s,%d'%(foldnum,split,len(file_list)))
            
            dataset = da.load_dataset(file_list, 'pdb')
            da.make_lmdb_dataset(dataset, output_path, fnat_cont_dic)
            


