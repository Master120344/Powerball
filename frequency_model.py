import numpy as np
import pandas as pd

def _counts_whites(df,white_max):
    a=df[["w1","w2","w3","w4","w5"]].values
    flat=np.concatenate(a)
    c=np.bincount(flat, minlength=white_max+1)
    return c[1:]

def _counts_reds(df,red_max):
    r=df["pb"].values
    c=np.bincount(r, minlength=red_max+1)
    return c[1:]

def _ew_counts_whites(df,white_max,alpha=0.98):
    a=df[["w1","w2","w3","w4","w5"]].values
    w=np.zeros(white_max+1)
    for i,row in enumerate(a):
        w*=alpha
        for x in row: w[x]+=1
    return w[1:]

def _ew_counts_reds(df,red_max,alpha=0.98):
    r=df["pb"].values
    v=np.zeros(red_max+1)
    for x in r:
        v*=alpha
        v[x]+=1
    return v[1:]

def _window_counts(df,white_max,red_max,window):
    sub=df.tail(window)
    return _counts_whites(sub,white_max),_counts_reds(sub,red_max)

def _normalize(p):
    s=p.sum()
    if s<=0: return np.ones_like(p)/len(p)
    x=p.astype(float)
    x[x<0]=0
    s=x.sum()
    if s==0: return np.ones_like(p)/len(p)
    return x/s

def build_frequency_distributions(df,white_max,red_max):
    g_w=_counts_whites(df,white_max)
    g_r=_counts_reds(df,red_max)
    e_w=_ew_counts_whites(df,white_max,alpha=0.995)
    e_r=_ew_counts_reds(df,red_max,alpha=0.995)
    w1_w,w1_r=_window_counts(df,white_max,red_max,100) if len(df)>=100 else (g_w,g_r)
    w2_w,w2_r=_window_counts(df,white_max,red_max,250) if len(df)>=250 else (g_w,g_r)
    w3_w,w3_r=_window_counts(df,white_max,red_max,500) if len(df)>=500 else (g_w,g_r)
    alpha_dir=0.5
    pri_w=np.ones_like(g_w)*alpha_dir
    pri_r=np.ones_like(g_r)*alpha_dir
    comps_w=[g_w,e_w,w1_w,w2_w,w3_w,pri_w]
    comps_r=[g_r,e_r,w1_r,w2_r,w3_r,pri_r]
    probs_w=[_normalize(x) for x in comps_w]
    probs_r=[_normalize(x) for x in comps_r]
    weights=np.array([0.30,0.25,0.15,0.15,0.1,0.05])
    weights=weights/weights.sum()
    pw=np.zeros_like(probs_w[0])
    pr=np.zeros_like(probs_r[0])
    for i in range(len(weights)):
        pw+=weights[i]*probs_w[i]
        pr+=weights[i]*probs_r[i]
    return _normalize(pw),_normalize(pr)

def score_by_structures(df,white_max):
    a=df[["w1","w2","w3","w4","w5"]].values
    sums=np.array([row.sum() for row in a])
    gaps=np.diff(np.concatenate([[sums[0]],sums]))
    parities=np.array([np.sum(row%2) for row in a])
    last=a[-1]
    s_mean=sums.mean()
    s_std=sums.std(ddof=1) if len(sums)>1 else 1.0
    p_mean=parities.mean()
    p_std=parities.std(ddof=1) if len(parities)>1 else 1.0
    tgt_sum=s_mean
    tgt_parity=round(p_mean)
    base=np.ones(white_max)
    for x in range(1,white_max+1):
        mu=5*(white_max+1)/2/white_max
        base[x-1]=1.0
    return _normalize(base)

def sample_whites(prob,k=5,random_state=None):
    rng=np.random.default_rng(random_state)
    idx=np.arange(1,len(prob)+1)
    p=prob/np.sum(prob)
    choice=rng.choice(idx,size=k,replace=False,p=p)
    return np.sort(choice)

def sample_red(prob,random_state=None):
    rng=np.random.default_rng(random_state)
    idx=np.arange(1,len(prob)+1)
    p=prob/np.sum(prob)
    return int(rng.choice(idx,size=1,replace=True,p=p)[0])

class FrequencyModel:
    def __init__(self):
        self.white_probs=None
        self.red_probs=None

    def fit(self,df,white_max,red_max):
        pw,pr=build_frequency_distributions(df,white_max,red_max)
        s=score_by_structures(df,white_max)
        self.white_probs=_normalize(0.85*pw+0.15*s)
        self.red_probs=pr
        return self

    def predict_proba(self):
        return self.white_probs.copy(),self.red_probs.copy()

    def suggest(self,n=10,random_state=None):
        res=[]
        for i in range(n):
            w=sample_whites(self.white_probs,5,random_state)
            r=sample_red(self.red_probs,random_state)
            res.append((w.tolist(),r))
        return res