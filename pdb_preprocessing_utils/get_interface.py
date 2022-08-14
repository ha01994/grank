from pdb2sql import pdb2sql
from pdb2sql import interface
import os, sys, glob
import numpy as np
from utils import *


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

    

cutoff = 8.5
output_dir = 'input_docking_models_itf'


for nn, pdb in enumerate(pdbs):
    print(nn , '/', len(pdbs))

    os.system('rm -rf %s/%s'%(output_dir,pdb))
    os.system('mkdir %s/%s'%(output_dir,pdb))
    
    for file in [x.split('/')[-1] for x in glob.glob('input_docking_models_modified2/%s/*'%pdb)]:        
        try:
            db = interface('input_docking_models_modified2/%s/%s'%(pdb,file))
            db_a = db(chainID='A')
            db_b = db(chainID='B')

            contact = db.get_contact_residues(cutoff=cutoff)        
            contact_a = [x[1] for x in contact['A']]
            contact_b = [x[1] for x in contact['B']]

            db_a = db_a(resSeq = contact_a)
            db_b = db_b(resSeq = contact_b)

            db = pdb2sql_concat((db_a, db_b), chains=['A','B'])
            db.exportpdb('%s/%s/%s'%(output_dir,pdb,file))
            
        except:
            print('Failed to process', file)
            
    


