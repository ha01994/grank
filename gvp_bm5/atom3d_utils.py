import collections as col
import gzip
import os
import re
import Bio.PDB.Atom
import Bio.PDB.Chain
import Bio.PDB.Model
import Bio.PDB.Residue
import Bio.PDB.Structure
import numpy as np
import pandas as pd



def split_df(df, key):
    """
    Split dataframe containing structure(s) based on specified key. Most commonly used to split by ensemble (`key="ensemble"`) or subunit (`key=["ensemble", "subunit"]`).

    :param df: Molecular structure(s) in ATOM3D dataframe format.
    :type df: pandas.DataFrame
    :param key: key on which to split dataframe. To split on multiple keys, provide all keys in a list. Must be compatible with dataframe hierarchy, i.e. ensemble > subunit > structure > model > chain.
    :type key: Union[str, list[str]]

    :return: List of tuples containing keys and corresponding sub-dataframes.
    :rtypes: list[tuple]
    """
    return [(x, y) for x, y in df.groupby(key)]




def bp_to_df(bp):
    """Convert biopython representation to ATOM3D dataframe representation.

    :param bp: Molecular structure in Biopython representation.
    :type bp: Bio.PDB.Structure

    :return: Molecular structure in ATOM3D dataframe format.
    :rtype: pandas.DataFrame
    """
    df = col.defaultdict(list)
    
    for atom in Bio.PDB.Selection.unfold_entities(bp, 'A'):
        residue = atom.get_parent()
        chain = residue.get_parent()
        model = chain.get_parent()
        
        df['chain'].append(chain.id)
        df['residue'].append(residue.id[1])
        df['x'].append(atom.coord[0])
        df['y'].append(atom.coord[1])
        df['z'].append(atom.coord[2])
        df['element'].append(atom.element)
    
    df = pd.DataFrame(df)
    
    return df



def df_to_bp(df_in):
    """Convert ATOM3D dataframe representation to biopython representation. Assumes dataframe contains only one structure.

    :param df_in: Molecular structure in ATOM3D dataframe format.
    :type df_in: pandas.DataFrame

    :return: Molecular structure in BioPython format.
    :rtype: Bio.PDB.Structure
    """
    all_structures = df_to_bps(df_in)
    if len(all_structures) > 1:
        raise RuntimeError('More than one structure in provided dataframe.')
    return all_structures[0]



def df_to_bps(df_in):
    """Convert ATOM3D dataframe representation containing multiple structures to list of Biopython structures. Assumes different structures are specified by `ensemble` and `structure` columns of dataframe.

    :param df_in: Molecular structures in ATOM3D dataframe format.
    :type df_in: pandas.DataFrame

    :return : List of molecular structures in BioPython format.
    :rtype: list[Bio.PDB.Structure]
    """
    df = df_in.copy()
    all_structures = []
    
    for (structure, s_atoms) in split_df(df_in, ['ensemble', 'structure']):
        new_structure = Bio.PDB.Structure.Structure(structure[1])
        
        for (model, m_atoms) in df.groupby(['model']):
            new_model = Bio.PDB.Model.Model(model)
            
            for (chain, c_atoms) in m_atoms.groupby(['chain']):
                new_chain = Bio.PDB.Chain.Chain(chain)
                
                for (residue, r_atoms) in c_atoms.groupby(
                        ['hetero', 'residue', 'insertion_code']):
                    # Take first atom as representative for residue values.
                    rep = r_atoms.iloc[0]
                    new_residue = Bio.PDB.Residue.Residue(
                        (rep['hetero'], rep['residue'], rep['insertion_code']),
                        rep['resname'], rep['segid'])
                    
                    for row, atom in r_atoms.iterrows():
                        new_atom = Bio.PDB.Atom.Atom(
                            atom['name'],
                            [atom['x'], atom['y'], atom['z']],
                            atom['bfactor'],
                            atom['occupancy'],
                            atom['altloc'],
                            atom['fullname'],
                            atom['serial_number'],
                            atom['element'])
                        new_residue.add(new_atom)
                        
                    new_chain.add(new_residue)
                    
                new_model.add(new_chain)
                
            new_structure.add(new_model)
            
        all_structures.append(new_structure)
        
    return all_structures



        

def read_pdb(pdb_file, name=None):
    
    if name is None:
        name = os.path.basename(pdb_file)
    parser = Bio.PDB.PDBParser(QUIET=True)
    bp = parser.get_structure(name, pdb_file)
    
    return bp



