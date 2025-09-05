import pandas as pd
import numpy as np
from dataclasses import dataclass

DEFAULT_WHITE_MAX=69
DEFAULT_RED_MAX=26

def _find_columns(df):
    cols={c.lower():c for c in df.columns}
    date_col=None
    for k in ["date","draw_date","drawdate","draw time","draw"]:
        if k in cols: date_col=cols[k]; break
    wcols=[]
    for k in ["w1","w2","w3","w4","w5"]:
        if k in cols: wcols.append(cols[k])
    if len(wcols)!=5:
        alt=[c for c in df.columns if str(c).lower().startswith("w")]
        alt_sorted=sorted(alt,key=lambda x:int("".join([ch for ch in str(x) if ch.isdigit()]) or 0))
        if len(alt_sorted)>=5: wcols=alt_sorted[:5]
    pb_col=None
    for k in ["pb","powerball","red","r","power_ball"]:
        if k in cols: pb_col=cols[k]; break
    return date_col,wcols,pb_col

def _coerce_int(x):
    try:
        return int(x)
    except:
        try:
            return int(float(str(x).strip().replace(",","")))
        except:
            return np.nan

def _detect_era_bounds(df):
    eras=[]
    if df.empty: return eras
    d1=pd.Timestamp("2015-10-07")
    d2=pd.Timestamp("2021-08-23")
    eras.append(("pre_2015",None,d1))
    eras.append(("2015_to_2021",d1,d2))
    eras.append(("post_2021",d2,None))
    return eras

def _era_params(name):
    if name=="pre_2015": return 59,35
    if name=="2015_to_2021": return 69,26
    if name=="post_2021": return 69,26
    return DEFAULT_WHITE_MAX,DEFAULT_RED_MAX

@dataclass
class DrawData:
    df: pd.DataFrame
    eras: pd.DataFrame

class DataRepository:
    def __init__(self, path="powerball.csv"):
        self.raw=pd.read_csv(path)
        date_col,wcols,pb_col=_find_columns(self.raw)
        if date_col is None or pb_col is None or len(wcols)!=5:
            raise ValueError("powerball.csv missing required columns")
        df=self.raw.copy()
        df[date_col]=pd.to_datetime(df[date_col])
        for c in wcols+[pb_col]:
            df[c]=df[c].apply(_coerce_int)
        df=df.dropna(subset=wcols+[pb_col])
        df=df.astype({c:int for c in wcols+[pb_col]})
        df=df.sort_values(date_col).reset_index(drop=True)
        df=df.drop_duplicates(subset=[date_col]+wcols+[pb_col])
        df["date"]=df[date_col]
        df["w1"],df["w2"],df["w3"],df["w4"],df["w5"]=df[wcols[0]],df[wcols[1]],df[wcols[2]],df[wcols[3]],df[wcols[4]]
        df["pb"]=df[pb_col]
        df=df[["date","w1","w2","w3","w4","w5","pb"]]
        eras=_detect_era_bounds(df)
        era_list=[]
        for name,start,end in eras:
            mask=True
            if start is not None: mask&=df["date"]>=start
            if end is not None: mask&=df["date"]<end
            sub=df.loc[mask].copy()
            if not sub.empty:
                wm,rm=_era_params(name)
                sub["white_max"]=wm
                sub["red_max"]=rm
                sub["era"]=name
                era_list.append(sub)
        if not era_list:
            df["white_max"]=DEFAULT_WHITE_MAX
            df["red_max"]=DEFAULT_RED_MAX
            df["era"]="unknown"
            era_df=df
        else:
            era_df=pd.concat(era_list,ignore_index=True).sort_values("date").reset_index(drop=True)
        era_df["white_set"]=era_df[["w1","w2","w3","w4","w5"]].values.tolist()
        self.data=DrawData(df=era_df,eras=era_df[["date","era","white_max","red_max"]].copy())

    def get_all(self):
        return self.data.df.copy()

    def get_arrays(self):
        d=self.data.df
        whites=d[["w1","w2","w3","w4","w5"]].values
        reds=d["pb"].values
        dates=d["date"].values
        white_max=d["white_max"].iloc[-1] if not d.empty else DEFAULT_WHITE_MAX
        red_max=d["red_max"].iloc[-1] if not d.empty else DEFAULT_RED_MAX
        return whites,reds,dates,white_max,red_max

    def rolling_windows(self, min_size=100, step=1):
        d=self.data.df
        for i in range(min_size,len(d),step):
            yield d.iloc[:i].copy()

    def walk_forward(self, min_size=200, step=1):
        d=self.data.df.reset_index(drop=True)
        for i in range(min_size,len(d)-1,step):
            train=d.iloc[:i].copy()
            target=d.iloc[i:i+1].copy()
            yield train,target

    def last_draw(self):
        return self.data.df.iloc[-1].copy()

    def last_k(self,k=10):
        return self.data.df.iloc[-k:].copy()