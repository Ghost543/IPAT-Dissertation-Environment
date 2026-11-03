"""BPI17 assemblers that reuse frozen model forward methods and building blocks."""
from dataclasses import dataclass
import torch
from torch import nn
from ipat.model.baseline import (FixedIdentityNumericalEncoder,TemporalEncoder,
    SemiOrthogonalProjector,PreNormTransformerBlock,StandardTransformerBaseline)
from ipat.model.adaptive import (FullIPATConfig,AdaptiveNumericalEncoder,
    AdaptiveStandardTransformer,FullIPATTransformer)
from ipat.model.retention import (RetentionAwarePreNormTransformerBlock,RetentionScorer,RetentionAwareTransformer)
from ipat.training.preprocessing import BaselinePreprocessingStats


@dataclass(frozen=True)
class BPI17Config(FullIPATConfig):
    categorical_cardinalities: tuple[int,...] = (4,4,4,4,4,4,4)
    categorical_embedding_dim: int = 4
    @property
    def d_cat(self):return 7*self.categorical_embedding_dim
    def validate(self):
        super().validate()
        if self.n_num!=6:raise ValueError('BPI17 requires six numerical channels')
        if len(self.categorical_cardinalities)!=7 or min(self.categorical_cardinalities)<3:raise ValueError('Seven vocabularies including PAD/MISSING/UNK required')
        if self.categorical_embedding_dim<1:raise ValueError('Positive embedding dimension required')


def model_stats(state):
    n=state['numeric'];t=state['temporal']
    return BaselinePreprocessingStats(tuple(x['mean'] for x in n),tuple(x['scale'] for x in n),
        t['timestamp']['mean'],t['timestamp']['scale'],t['delta']['mean'],t['delta']['scale'])


def config_from_state(state,**overrides):
    if 'categorical_cardinalities' in overrides:raise ValueError('Cardinalities come from training vocabularies')
    return BPI17Config(categorical_cardinalities=tuple(len(v)+3 for v in state['vocabularies']),**overrides)


class BPI17CategoricalEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(c,cfg.categorical_embedding_dim,padding_idx=0) for c in cfg.categorical_cardinalities])
    def forward(self,ids,valid):
        if ids.shape[-1]!=7:raise ValueError('Expected seven categorical fields')
        # Invalid positions are overwritten before lookup, so padding content cannot leak.
        ids=torch.where(valid.unsqueeze(-1),ids,torch.zeros_like(ids))
        return torch.cat([e(ids[...,j]) for j,e in enumerate(self.embeddings)],dim=-1)


class BPI17Assembler(nn.Module):
    def __init__(self,cfg,stats,adaptive):
        super().__init__();self.adaptive=adaptive
        self.numeric=(AdaptiveNumericalEncoder if adaptive else FixedIdentityNumericalEncoder)(cfg,stats)
        self.categorical=BPI17CategoricalEncoder(cfg);self.temporal=TemporalEncoder(cfg,stats)
    def forward(self,batch):
        valid=batch['valid_mask']
        if self.adaptive:
            n=self.numeric.encode(batch['x_obs'],batch['missing_mask'],valid)
            num=n['block'];components={'x_std':n['x_std'],'g':n['g'],'num':num,
                'numeric_candidates':n['candidates'],'mixture_weights':n['mixture_weights'],
                'mixture_derivative':n['mixture_derivative']}
        else:
            num,x_std=self.numeric(batch['x_obs'],batch['missing_mask'],valid)
            components={'num':num,'x_std':x_std}
        cat=self.categorical(batch['categorical'],valid)
        time=self.temporal(batch['timestamp'],batch['delta'],batch['position'],valid)
        mask=batch['missing_mask'].to(num.dtype)*valid.unsqueeze(-1)
        u=torch.cat([num,cat,time,mask],dim=-1)*valid.unsqueeze(-1)
        components.update(cat=cat,time=time,mask=mask)
        return u,components


class _SchemaInit:
    adaptive=False
    retention=False
    def __init__(self,cfg,stats):
        nn.Module.__init__(self);cfg.validate();self.cfg=cfg;self.stats=stats
        self.assembler=BPI17Assembler(cfg,stats,self.adaptive)
        self.projector=SemiOrthogonalProjector(cfg.d_in,cfg.d_model)
        block=RetentionAwarePreNormTransformerBlock if self.retention else PreNormTransformerBlock
        self.blocks=nn.ModuleList([block(cfg) for _ in range(cfg.n_layers)])
        self.head_norm=nn.LayerNorm(cfg.d_model);self.classifier=nn.Linear(cfg.d_model,1)
        if self.retention:self.retention_scorer=RetentionScorer(cfg.d_model)


class BPI17Baseline(_SchemaInit,StandardTransformerBaseline):pass
class BPI17RepresentationOnly(_SchemaInit,AdaptiveStandardTransformer):adaptive=True
class BPI17AttentionOnly(_SchemaInit,RetentionAwareTransformer):retention=True
class BPI17FullIPAT(_SchemaInit,FullIPATTransformer):adaptive=True;retention=True
MODELS=(BPI17Baseline,BPI17RepresentationOnly,BPI17AttentionOnly,BPI17FullIPAT)
