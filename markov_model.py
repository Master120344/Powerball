import numpy as np
import pandas as pd

def _normalize(p):
    s=p.sum()
    if s<=0: return np.ones_like(p)/len(p)
    x=p.astype(float)
    x[x<0]=0
    s=x.sum()
    if s==0: return np.ones_like(p)/len(p)
    return x/s

def _cooccurrence(df,white_max):
    a=df[["w1","w2","w3","w4","w5"]].values
    mat=np.zeros((white_max+1,white_max+1))
    for row in a:
        for i in range(5):
            for j in range(i+1,5):
                x=row[i]; y=row[j]
                mat[x,y]+=1
                mat[y,x]+=1
    return mat

def _transition(df,white_max):
    a=df[["w1","w2","w3","w4","w5"]].values
    T=np.zeros((white_max+1,white_max+1))
    for i in range(len(a)-1):
        src=a[i]; dst=a[i+1]
        for x in src:
            for y in dst:
                T[x,y]+=1
    return T

def _red_transition(df,red_max):
    r=df["pb"].values
    T=np.zeros((red_max+1,red_max+1))
    for i in range(len(r)-1):
        T[r[i],r[i+1]]+=1
    return T

def _gaps(df,white_max):
    a=df[["w1","w2","w3","w4","w5"]].values
    last_seen=np.full(white_max+1,-1)
    gaps=np.zeros(white_max+1)
    for t,row in enumerate(a):
        for x in range(1,white_max+1):
            if last_seen[x]==-1: continue
        for x in row:
            last_seen[x]=t
    t=len(a)-1
    for x in range(1,white_max+1):
        if last_seen[x]==-1: gaps[x]=t+1
        else: gaps[x]=t-last_seen[x]
    return gaps[1:]

def _hazard_from_gaps(gaps):
    g=np.array(gaps,dtype=float)
    g[g<1]=1
    h=1/(g)
    return _normalize(h)

def markov_predict(df,white_max,red_max):
    co=_cooccurrence(df,white_max)
    tr=_transition(df,white_max)
    fr=df[["w1","w2","w3","w4","w5"]].values[-1]
    v=np.zeros(white_max+1)
    for x in fr:
        v+=tr[x]
        v+=co[x]
    v=v[1:]
    gaps=_gaps(df,white_max)
    hz=_hazard_from_gaps(gaps)
    freq=np.zeros(white_max)
    flat=np.concatenate(df[["w1","w2","w3","w4","w5"]].values)
    cnt=np.bincount(flat, minlength=white_max+1)[1:]
    freq=_normalize(cnt)
    pw=_normalize(0.55*_normalize(v)+0.25*hz+0.20*freq)
    rT=_red_transition(df,red_max)
    last_r=df["pb"].values[-1]
    rv=rT[last_r][1:]
    rcnt=np.bincount(df["pb"].values, minlength=red_max+1)[1:]
    rprob=_normalize(0.70*_normalize(rv)+0.30*_normalize(rcnt))
    return pw,rprob

class MarkovModel:
    def __init__(self):
        self.white_probs=None
        self.red_probs=None

    def fit(self,df,white_max,red_max):
        pw,pr=markov_predict(df,white_max,red_max)
        self.white_probs=pw
        self.red_probs=pr
        return self

    def predict_proba(self):
        return self.white_probs.copy(),self.red_probs.copy()