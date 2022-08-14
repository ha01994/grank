import csv, gzip
import json, io, os, sys
from pathlib import Path
import tqdm
import lmdb
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import atom3d_utils


def serialize(x, serialization_format):
    serialized = json.dumps(x, default=lambda df: json.loads(df.to_json(orient='split', double_precision=6))).encode()
    return serialized


def deserialize(x, serialization_format):
    serialized = json.loads(x)
    return serialized



class LMDBDataset(Dataset):
    
    def __init__(self, data_file, transform=None):
        
        if type(data_file) is list:
            if len(data_file) != 1: raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]
        self.data_file = Path(data_file).absolute()
        
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)
        
        self.env = None
        self.txn = None
        env = lmdb.open(str(self.data_file), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)        
        with env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = txn.get(b'serialization_format').decode()
            self._id_to_idx = deserialize(txn.get(b'id_to_idx'), self._serialization_format)
        self._env = env
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def get(self, id: str):
        idx = self.id_to_idx(id)
        return self[idx]

    def id_to_idx(self, id: str):
        if id not in self._id_to_idx: 
            raise IndexError(id)
        idx = self._id_to_idx[id]
        return idx

    def ids_to_indices(self, ids):
        return [self.id_to_idx(id) for id in ids]

    def ids(self):
        return list(self._id_to_idx.keys())

    def __getitem__(self, index: int):        
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
            
        with self._env.begin(write=False) as txn:
            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            try:
                item = deserialize(serialized, self._serialization_format)
            except:
                return None
        
        if 'types' in item.keys():
            for x in item.keys():
                if item['types'][x] == str(pd.DataFrame):
                    item[x] = pd.DataFrame(**item[x])
        else:
            print('Data types in item %i not defined. Will use basic types only.'%index)

        if self._transform:
            item = self._transform(item)

        return item




class PDBDataset(Dataset):
    def __init__(self, file_list, transform=None, store_file_path=False):
        self._file_list = [Path(x).absolute() for x in file_list]
        self._num_examples = len(self._file_list)
        self._transform = transform
        self._store_file_path = store_file_path

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples: raise IndexError(index)

        file_path = self._file_list[index]
        item = {'atoms': atom3d_utils.bp_to_df(atom3d_utils.read_pdb(file_path)),
                'id': str(file_path)}

        if self._store_file_path:
            item['file_path'] = str(file_path)
        if self._transform:
            item = self._transform(item)

        return item


    


def load_dataset(file_list, filetype, transform=None):
    if filetype == 'lmdb':
        dataset = LMDBDataset(file_list, transform=transform)
    elif filetype == 'pdb':
        dataset = PDBDataset(file_list, transform=transform)

    return dataset




def make_lmdb_dataset(dataset, output_lmdb, label_dict):
    serialization_format = 'json'
    env = lmdb.open(str(output_lmdb), map_size=int(1e13))

    with env.begin(write=True) as txn:
        try:
            id_to_idx = {}
            i = 0
            for x in tqdm.tqdm(dataset, total=len(dataset)):
                
                #######################################################################
                filename = x['id'].split('/')[-1]
                x['label'] = label_dict[filename]
                #######################################################################
                                
                x['types'] = {key: str(type(val)) for key, val in x.items()}
                x['types']['types'] = str(type(x['types']))

                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
                    f.write(serialize(x, serialization_format))
                compressed = buf.getvalue()
                result = txn.put(str(i).encode(), compressed, overwrite=False)
                if not result:
                    raise RuntimeError(f'LMDB entry {i} in {str(output_lmdb)} already exists')
                id_to_idx[x['id']] = i
                i += 1

        finally:
            txn.put(b'num_examples', str(i).encode())
            txn.put(b'serialization_format', serialization_format.encode())
            txn.put(b'id_to_idx', serialize(id_to_idx, serialization_format))




