"""Detached pilot metrics: explicit undefined values and application dependence."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def probabilities(logits):
    x=np.asarray(logits,dtype=np.float64)
    return np.exp(-np.logaddexp(0.,-x))


def losses(y,logits):
    y=np.asarray(y,dtype=np.float64);x=np.asarray(logits,dtype=np.float64)
    if y.ndim!=1 or x.shape!=y.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(x).all():
        raise ValueError('Finite aligned logits and nonempty binary labels required')
    return np.logaddexp(0.,x)-y*x


def reliability(y,p,equal_mass=False):
    y=np.asarray(y);p=np.asarray(p);confidence=np.maximum(p,1-p);correct=((p>=.5)==y)
    if equal_mass:
        groups=np.array_split(np.argsort(confidence,kind='stable'),15)
    else:
        bins=np.minimum((confidence*15).astype(int),14)
        groups=[np.flatnonzero(bins==i) for i in range(15)]
    rows=[]
    for g in groups:
        rows.append(dict(count=len(g),confidence=float(confidence[g].mean()) if len(g) else None,
                         accuracy=float(correct[g].mean()) if len(g) else None))
    return sum(r['count']/len(y)*abs(r['accuracy']-r['confidence']) for r in rows if r['count']),rows


def metrics(y,logits,applications=None):
    l=losses(y,logits);y=np.asarray(y,dtype=int);p=probabilities(logits);pred=p>=.5
    cm=np.array([[int(np.sum((y==i)&(pred==j))) for j in (0,1)] for i in (0,1)])
    cls={};reasons={}
    for c in (0,1):
        support=int(cm[c].sum());positive=int(cm[:,c].sum());tp=int(cm[c,c]);den=support+positive
        cls[str(c)]=dict(support=support,precision=tp/positive if positive else None,
                        recall=tp/support if support else None,f1=2*tp/den if den else None)
        if not support:reasons[f'class_{c}_recall']='No true instances of this class'
        if not positive:reasons[f'class_{c}_precision']='No predicted instances of this class'
    both=all(cls[str(c)]['support'] for c in (0,1))
    ece,bins=reliability(y,p);mass,_=reliability(y,p,True)
    result=dict(n=len(y),nll=float(l.mean()),accuracy=float((pred==y).mean()),
        balanced_accuracy=float(np.mean([cls[str(c)]['recall'] for c in (0,1)])) if both else None,
        macro_f1=float(np.mean([cls[str(c)]['f1'] for c in (0,1)])) if both else None,
        auroc=float(roc_auc_score(y,p)) if both else None,
        average_precision={str(c):float(average_precision_score(y==c,p if c else 1-p)) if both else None for c in (0,1)},
        brier=float(np.mean((p-y)**2)),ece=ece,ece_equal_mass=mass,reliability=bins,
        classes=cls,confusion_matrix=cm.tolist(),undefined_reasons=reasons)
    if not both:reasons['ranking_balanced_macro']='Both labels required by pilot convention'
    if applications is not None:
        apps=np.asarray(applications)
        if apps.shape!=y.shape:raise ValueError('Application alignment mismatch')
        _,inv=np.unique(apps,return_inverse=True);counts=np.bincount(inv)
        result.update(applications=len(counts),application_macro_nll=float(np.mean(np.bincount(inv,weights=l)/counts)))
    return result


def stratified(y,logits,apps,lengths):
    result={}
    for lo,hi in [(1,10),(11,20),(21,40),(41,None)]:
        ix=(lengths>=lo)&((lengths<=hi) if hi else True);key=f'{lo}-{hi}' if hi else '41+'
        result[key]=metrics(y[ix],logits[ix],apps[ix]) if ix.any() else {'n':0,'nll':None,'undefined_reasons':{'all':'Empty stratum'}}
    return result


def paired_cluster_interval(y,logits_a,logits_b,apps,draws=2000,seed=50501):
    """Inputs: seeds x offers; resample applications identically for all seeds/arms."""
    a=np.asarray(logits_a);b=np.asarray(logits_b);apps=np.asarray(apps)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=len(y) or apps.shape!=(len(y),) or draws<1:raise ValueError('Paired seeds/offers required')
    delta=np.mean([losses(y,x)-losses(y,z) for x,z in zip(a,b)],axis=0)
    unique,inv=np.unique(apps,return_inverse=True);n=np.bincount(inv);s=np.bincount(inv,weights=delta)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        ii=rng.integers(0,len(unique),size=len(unique));values.append(float(s[ii].sum()/n[ii].sum()))
    return dict(mean_delta=float(delta.mean()),percentile_95=np.quantile(values,[.025,.975]).tolist(),draws=draws,
                scope='Descriptive application-cluster interval conditional on fitted checkpoints; not full training uncertainty')
