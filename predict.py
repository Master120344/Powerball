import argparse
import numpy as np
import pandas as pd
from data_loader import DataRepository
from frequency_model import FrequencyModel, sample_whites, sample_red
from markov_model import MarkovModel

def normalize(p):
    s=p.sum()
    if s<=0: return np.ones_like(p)/len(p)
    x=p.astype(float); x[x<0]=0
    s=x.sum()
    if s==0: return np.ones_like(p)/len(p)
    return x/s

def ensemble_probs(p_list, w_list=None):
    if w_list is None:
        w_list=[1.0/len(p_list)]*len(p_list)
    w=np.array(w_list)/np.sum(w_list)
    p=np.zeros_like(p_list[0])
    for i in range(len(p_list)):
        p+=w[i]*normalize(p_list[i])
    return normalize(p)

def pick_set(pw,pr,n_lines=10,random_state=None):
    res=[]
    rng=np.random.default_rng(random_state)
    for i in range(n_lines):
        ws=sample_whites(pw,5,rng.integers(0,1<<32))
        rr=sample_red(pr,rng.integers(0,1<<32))
        res.append((ws.tolist(),int(rr)))
    return res

def metrics_topk(pw,pr,actual_whites,actual_red,ks=(5,10,15)):
    idx=np.argsort(pw)[::-1]+1
    rdx=np.argsort(pr)[::-1]+1
    m={}
    for k in ks:
        top=set(idx[:k])
        hits=len([x for x in actual_whites if x in top])
        m[f"whites_in_top_{k}"]=hits
    m["red_top_1"]=int(rdx[0]==actual_red)
    m["red_top_3"]=int(actual_red in set(rdx[:3]))
    m["red_top_5"]=int(actual_red in set(rdx[:5]))
    return m

def walk_forward_eval(repo, w_freq=0.5, w_markov=0.5, min_size=300, step=1):
    rows=[]
    for train,target in repo.walk_forward(min_size=min_size,step=step):
        wm=train["white_max"].iloc[-1]
        rm=train["red_max"].iloc[-1]
        f=FrequencyModel().fit(train,wm,rm)
        m=MarkovModel().fit(train,wm,rm)
        fw,fr=f.predict_proba()
        mw,mr=m.predict_proba()
        pw=ensemble_probs([fw,mw],[w_freq,w_markov])
        pr=ensemble_probs([fr,mr],[w_freq,w_markov])
        tw=target[["w1","w2","w3","w4","w5"]].values[0].tolist()
        tr=int(target["pb"].values[0])
        mt=metrics_topk(pw,pr,tw,tr)
        row={"date":target["date"].values[0],"era":target["era"].values[0]}
        row.update(mt)
        rows.append(row)
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="powerball.csv")
    ap.add_argument("--lines",type=int,default=20)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--eval",action="store_true")
    ap.add_argument("--min_size",type=int,default=300)
    ap.add_argument("--w_freq",type=float,default=0.55)
    ap.add_argument("--w_markov",type=float,default=0.45)
    args=ap.parse_args()
    repo=DataRepository(args.csv)
    df=repo.get_all()
    wm=df["white_max"].iloc[-1]
    rm=df["red_max"].iloc[-1]
    f=FrequencyModel().fit(df,wm,rm)
    m=MarkovModel().fit(df,wm,rm)
    fw,fr=f.predict_proba()
    mw,mr=m.predict_proba()
    pw=ensemble_probs([fw,mw],[args.w_freq,args.w_markov])
    pr=ensemble_probs([fr,mr],[args.w_freq,args.w_markov])
    picks=pick_set(pw,pr,n_lines=args.lines,random_state=args.seed)
    print("PREDICTED_DISTRIBUTIONS_WHITES_TOP10:", (np.argsort(pw)[-10:]+1).tolist())
    print("PREDICTED_DISTRIBUTIONS_RED_TOP5:", (np.argsort(pr)[-5:]+1).tolist())
    print("SUGGESTED_LINES:")
    for w,r in picks:
        print(w,r)
    if args.eval:
        res=walk_forward_eval(repo,w_freq=args.w_freq,w_markov=args.w_markov,min_size=args.min_size,step=1)
        if not res.empty:
            agg={}
            for c in [x for x in res.columns if str(x).startswith("whites_in_top_")]:
                agg[c]=res[c].mean()
            agg["red_top_1"]=res["red_top_1"].mean()
            agg["red_top_3"]=res["red_top_3"].mean()
            agg["red_top_5"]=res["red_top_5"].mean()
            print("BACKTEST_MEAN_METRICS:", {k:round(float(v),4) for k,v in agg.items()})
            last=res.tail(10)
            print("RECENT_10:", last.to_dict(orient="records"))

if __name__=="__main__":
    main()