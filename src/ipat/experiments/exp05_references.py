"""Reference models; their differences from the matched Transformer are explicit."""
import numpy as np
import torch
from torch import nn
from ipat.model.bpi17 import BPI17Assembler
from ipat.model.baseline import SemiOrthogonalProjector
from sklearn.ensemble import HistGradientBoostingClassifier


class PilotGRU(nn.Module):
    def __init__(self,cfg,stats):
        super().__init__();self.assembler=BPI17Assembler(cfg,stats,False)
        self.projector=SemiOrthogonalProjector(cfg.d_in,cfg.d_model)
        self.gru=nn.GRU(cfg.d_model,60,batch_first=True);self.norm=nn.LayerNorm(60);self.head=nn.Linear(60,1)
    def forward(self,batch):
        u,_=self.assembler(batch);z=self.projector(u,batch['valid_mask'])
        packed=nn.utils.rnn.pack_padded_sequence(z,batch['valid_mask'].sum(1).cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed)
        return {'logits':self.head(self.norm(h[-1])).squeeze(-1)}


def prefix_summaries(dataset,state):
    rows=[]
    for i in range(len(dataset)):
        x,_,_=dataset[i];row=[]
        for j in range(6):
            v=x['x_obs'][:,j][~x['missing_mask'][:,j]]
            row.extend([float(v[-1]),float(v.mean()),float(v.std()),float(v.min()),float(v.max())] if len(v) else [np.nan]*5)
            row.append(len(v)/len(x['timestamp']))
        for j,vocab in enumerate(state['vocabularies']):
            size=len(vocab)+3;ids=x['categorical'][:,j]
            row.extend((np.bincount(ids,minlength=size)[1:]/len(ids)).tolist())
            final=np.zeros(size);final[ids[-1]]=1;row.extend(final[1:].tolist())
        gaps=x['delta'];row.extend([len(gaps),float(x['timestamp'][-1]),float(gaps.mean()),float(gaps.std()),float(gaps.min()),float(gaps.max())])
        rows.append(row)
    return np.asarray(rows,dtype=np.float64)


def tree(seed):
    return HistGradientBoostingClassifier(loss='log_loss',max_iter=200,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=20,l2_regularization=0.,max_bins=255,categorical_features=None,
        early_stopping=False,class_weight=None,random_state=seed)
