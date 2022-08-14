#author: Sungsik Kim


import os, csv, sys, glob
from pdb2sql import pdb2sql
import numpy as np
from random import randint
from Bio import pairwise2


def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)





def reres(db, start=1):
    res_seq_new = [start]
    _res_seq_new = start
    for res_seq_prev, res_seq in zip(np.array(db.get('resSeq'))[:-1], np.array(db.get('resSeq'))[1:]):
        if res_seq != res_seq_prev:
            _res_seq_new += 1
        res_seq_new.append(_res_seq_new)

    db.update_column('resSeq', values=res_seq_new)

    return db



def pdb2sql_concat(dbs, chains):
    assert len(dbs) == len(chains)

    _start = 1
    for db, chain in zip(dbs, chains):
        db.update_column('chainID', values = [chain]*len(db.get('*')))
        db = reres(db, start = _start)
        _start = max(db.get('resSeq')) + 1

    pdb_lines = []
    for db in dbs:
        pdb_lines.extend(db.sql2pdb())

    db_concat = pdb2sql(pdb_lines)
    db_concat.update_column('serial', values = range(1, len(db_concat.get('*'))+1))

    return db_concat





aa_dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
           'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

def to_1code(list_aa):
    return ''.join([aa_dict[a] for a in list_aa])