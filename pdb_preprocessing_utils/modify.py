import os, csv, sys, glob


all_files = []
for foldnum in range(1,11):
    for split in ['train', 'valid']:
        with open('FILES_RTTONLY/fold%d_%s.csv'%(foldnum, split), 'r') as f:
            r = csv.reader(f)
            for line in r:
                all_files.append(line[0])
                
with open('FILES_RTTONLY/test.csv', 'r') as f:
    r = csv.reader(f)
    for line in r:
        all_files.append(line[0])
        
print(len(all_files))



files_to_modify = []
for file in all_files:
    pdb = file.split('_')[0]
    if not os.path.exists('/home/ha01994/struct_pred/deeprank_bm5/input_docking_models_itf/%s/%s'%(pdb, file)):
        files_to_modify.append(file)


            

k = int(sys.argv[1])

pdbs1 = ['1ACB', '1AK4', '1ATN', '1AVX', '1AY7', '1B6C', '1BKD', '1BUH', '1BVN', '1CGI', ]
pdbs2 = ['1CLV', '1D6R', '1DFJ', '1E6E', '1E96', '1EAW', '1EFN', '1EWY', '1F34', '1F6M', ]
pdbs3 = ['1FC2', '1FFW', '1FLE', '1FQ1', '1FQJ', '1GCQ', '1GHQ', '1GL1', '1GLA', '1GPW', ]
pdbs4 = ['1GRN', '1GXD', '1H1V', '1H9D', '1HE1', '1HE8', '1I2M', '1IBR', '1J2J', '1JIW', ]
pdbs5 = ['1JK9', '1JTD', '1JTG', '1KAC', '1KTZ', '1KXP', '1KXQ', '1LFD', '1M10', '1MAH', ]
pdbs6 = ['1MQ8', '1NW9', '1OC0', '1OPH', '1OYV', '1PPE', '1PVH', '1PXV', '1QA9', '1R0R', ]
pdbs7 = ['1R6Q', '1R8S', '1RKE', '1S1Q', '1SBB', '1SYX', '1T6B', '1TMQ', '1UDI', '1US7', ]
pdbs8 = ['1WQ1', '1XD3', '1XQS', '1Y64', '1YVB', '1Z0K', '1Z5Y', '1ZHH', '1ZHI', '1ZLI', ]
pdbs9 = ['1ZM4', '2A1A', '2A5T', '2A9K', '2ABZ', '2AJF', '2AYO', '2B42', '2BTF', '2C0L', ]
pdbs10 = ['2CFH', '2FJU', '2G77', '2GAF', '2GTP', '2H7V', '2HLE', '2HQS', '2HRK', '2I25',]
pdbs11 = ['2I9B', '2IDO', '2J0T', '2J7P', '2NZ8', '2O3B', '2O8V', '2OOB', '2OT3', '2OUL',]
pdbs12 = ['2OZA', '2PCC', '2SIC', '2SNI', '2UUY', '2VDB', '2X9A', '2YVJ', '2Z0E', '3A4S',]
pdbs13 = ['3AAD', '3BIW', '3BX7', '3CPH', '3D5S', '3DAW', '3F1P', '3FN1', '3H2V', '3K75', 'BOYV']
pdbs14 = ['3PC8', '3S9D', '3SGQ', '3VLB', '4CPA', '4FZA', '4H03', '4IZ7', '4M76', '7CEI', 'BAAD']

if k==1: pdbs = pdbs1
if k==2: pdbs = pdbs2
if k==3: pdbs = pdbs3
if k==4: pdbs = pdbs4
if k==5: pdbs = pdbs5
if k==6: pdbs = pdbs6
if k==7: pdbs = pdbs7
if k==8: pdbs = pdbs8
if k==9: pdbs = pdbs9
if k==10: pdbs = pdbs10
if k==11: pdbs = pdbs11
if k==12: pdbs = pdbs12
if k==13: pdbs = pdbs13
if k==14: pdbs = pdbs14

    
    
files_to_modify = [f for f in files_to_modify if f.split('_')[0] in pdbs]


accepted = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']


for en, file in enumerate(files_to_modify):
    if en % 10000 == 0:
        print(en, '/', len(files_to_modify))

    pdb = file.split('_')[0]    
    
    e = 'input_docking_models_modified/%s/'%pdb
    f = 'input_docking_models_modified2/%s/'%pdb
    if not os.path.exists(e): os.system('mkdir %s'%e)        
    if not os.path.exists(f): os.system('mkdir %s'%f)
    
    path0 = '/home/ha01994/struct_pred/deeprank_bm5/input_docking_models/%s/%s'%(pdb, file)
    path1 = 'input_docking_models_modified/%s/%s'%(pdb, file)
    path2 = 'input_docking_models_modified2/%s/%s'%(pdb, file)

    towrites = []
    with open(path0, 'r') as f:
        for line in f:
            
            if line[0:6] == 'REMARK': pass
            elif line[0:3] == 'END': pass
            elif line[0:3] == 'TER': pass
            elif line[17:20] not in accepted: pass

            else:
                a = line[6:11].strip() #atom serial number
                b = line[12:16].strip() #atom name
                r = line[17:20].strip() #residue name                
                o = line[72:76].strip() #chain
                k = line[22:26].strip() #pos
                c = line[30:38].strip() #x
                d = line[38:46].strip() #y
                e = line[46:54].strip() #z
                y = line[54:60].strip() #occupancy
                ###################################
                if len(a) <= 4:
                    a = ' ' + ' '*(4-len(a)) + a
                elif len(a) == 5:
                    a =                        a
                ###################################
                if len(b) <= 3:
                    b = ' ' + b + ' '*(3-len(b))                    
                elif len(b) == 4:
                    b =       b
                ###################################
                if len(k) <= 3: 
                    k = ' ' + ' '*(3-len(k)) + k                    
                elif len(k) == 4:
                    k =                        k
                ###################################
                if len(c) <= 7: 
                    c = ' ' + ' '*(7-len(c)) + c
                elif len(c) == 8:
                    c =                        c
                ###################################
                if len(d) <= 7: 
                    d = ' ' + ' '*(7-len(d)) + d
                elif len(d) == 8:
                    d =                        d
                ###################################
                if len(e) <= 7: 
                    e = ' ' + ' '*(7-len(e)) + e
                elif len(e) == 8:
                    e =                        e
                ###################################

                towrite = 'ATOM  '+ a + ' '+    b   + ' '+ r + ' ' + o + k + '    ' + c + d + e + '  '+ y + '                 \n'
                towrites.append(towrite)

    with open(path1, 'w') as fw:
        for towrite in towrites:
            fw.write('%s'%towrite)

    os.system('pdb_b %s > %s'%(path1, path2))
    
    
    
    
