import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster, torch_geometric, torch_scatter
from gvp import GVP, GVPConvLayer, LayerNorm



def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))



def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF




_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)

_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)



def _edge_features(coords, edge_index, D_max, num_rbf=16, device='cpu'):

    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)
    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


#####################################################################################################################


class BaseTransform:
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device

    def __call__(self, df):
        with torch.no_grad():            
            
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=self.device)
            
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)), dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

            edge_s, edge_v = _edge_features(coords, edge_index, self.edge_cutoff, num_rbf=self.num_rbf,
                                            device=self.device)

            return torch_geometric.data.Data(x=coords, atoms=atoms, edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)



class BaseModel(nn.Module):
    def __init__(self, num_rbf=16):
        super().__init__()
        activations = (F.relu, None)

        self.embed = nn.Embedding(_NUM_ATOM_TYPES, _NUM_ATOM_TYPES)

        self.W_e = nn.Sequential(LayerNorm((num_rbf, 1)),
                                 GVP((num_rbf, 1), _DEFAULT_E_DIM, activations=(None, None), vector_gate=True))

        self.W_v = nn.Sequential(LayerNorm((_NUM_ATOM_TYPES, 0)),
                                 GVP((_NUM_ATOM_TYPES, 0), _DEFAULT_V_DIM, activations=(None, None), vector_gate=True))

        self.layers = nn.ModuleList(GVPConvLayer(_DEFAULT_V_DIM, _DEFAULT_E_DIM, activations=activations, vector_gate=True)
                        for _ in range(5))

        ns, _ = _DEFAULT_V_DIM

        self.W_out = nn.Sequential(LayerNorm(_DEFAULT_V_DIM),
                                   GVP(_DEFAULT_V_DIM, (ns, 0), activations=activations, vector_gate=True))

        self.dense = nn.Sequential(nn.Linear(ns, 2*ns),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.1),
                                   nn.Linear(2*ns, 1))

    def forward(self, batch, scatter_mean=True, dense=True):

        h_V = self.embed(batch.atoms)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        batch_id = batch.batch
        
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)

        if scatter_mean:
            out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        if dense:
            out = self.dense(out).squeeze(-1)

        return out


#####################################################################################################################


class Transform(BaseTransform):
    def __call__(self, elem):
        df = elem['atoms']
        #df = df[df.element != 'H'].reset_index(drop=True)
        
        data = super().__call__(df) # call BaseTransform

        data.id = elem['id']
        data.label = elem['label']
        
        return data

    
    
    
Model = BaseModel



