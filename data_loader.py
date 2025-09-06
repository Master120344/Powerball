import os
import re
import pandas as pd

class DataRepository:
    def __init__(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = self._load_csv(path)
        elif ext == ".txt":
            df = self._load_txt(path)
        else:
            raise ValueError("Unsupported file format")
        self.raw = self._coerce(df)
        self._validate()

    def _load_csv(self, path):
        df = pd.read_csv(path, dtype=str, engine="python")
        cols = {c.lower().strip(): c for c in df.columns}
        date_col = None
        for k in ("date","draw date","draw_date","drawing","draw"):
            if k in cols:
                date_col = cols[k]
                break
        num_col = None
        for k in ("winning numbers","numbers","nums","balls","winning_numbers"):
            if k in cols:
                num_col = cols[k]
                break
        pb_col = None
        for k in ("powerball","pb","power ball"):
            if k in cols:
                pb_col = cols[k]
                break
        bcols = []
        for i in range(1,6):
            for k in (f"ball{i}",f"b{i}",f"n{i}"):
                if k in cols:
                    bcols.append(cols[k])
                    break
        rows = []
        if num_col is not None and len(bcols)==0:
            tmp = df[[x for x in [date_col,num_col,pb_col] if x is not None]].copy()
            for _, r in tmp.iterrows():
                d = r[date_col] if date_col in r else None
                s = str(r[num_col]) if num_col in r else ""
                s = s.replace(",", " ").replace("-", " ").replace("|", " ")
                nums = [int(x) for x in re.findall(r"\d+", s)]
                if len(nums) >= 5:
                    balls = nums[:5]
                    pb = None
                    if pb_col is not None:
                        pb = int(str(r[pb_col])) if str(r[pb_col]).strip().isdigit() else None
                    else:
                        if len(nums) >= 6:
                            pb = nums[5]
                    if d is not None and pb is not None:
                        rows.append([d]+balls+[pb])
        else:
            need = []
            if date_col: need.append(date_col)
            need += bcols
            if pb_col: need.append(pb_col)
            if len(bcols)==5 and pb_col and date_col and all(c in df.columns for c in need):
                sub = df[need].copy()
                for _, r in sub.iterrows():
                    d = r[date_col]
                    balls = []
                    ok = True
                    for c in bcols:
                        v = re.findall(r"\d+", str(r[c]))
                        if not v: ok=False; break
                        balls.append(int(v[0]))
                    if not ok: continue
                    pv = re.findall(r"\d+", str(r[pb_col]))
                    if not pv: continue
                    rows.append([d]+balls+[int(pv[0])])
        return pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])

    def _load_txt(self, path):
        rows = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", s)
                if not m:
                    parts = s.replace(",", " ").replace("|", " ").split()
                    if len(parts)==7 and re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", parts[0]):
                        d = parts[0]
                        nums = [int(x) for x in parts[1:]]
                        rows.append([d]+nums[:5]+[nums[5]])
                        continue
                    continue
                d = m.group(1)
                rest = (s[:m.start()] + " " + s[m.end():]).replace(",", " ").replace("-", " ").replace("|", " ")
                nums = [int(x) for x in re.findall(r"\d+", rest)]
                if len(nums) >= 6:
                    balls = nums[:5]
                    pb = nums[5]
                    rows.append([d]+balls+[pb])
        return pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])

    def _coerce(self, df):
        df = df.dropna()
        df = df.copy()
        df["date"] = df["date"].astype(str).str.strip()
        for c in ["ball1","ball2","ball3","ball4","ball5","powerball"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        df = df.dropna().astype({"ball1":"int64","ball2":"int64","ball3":"int64","ball4":"int64","ball5":"int64","powerball":"int64"})
        try:
            dt = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
            df = df.loc[~dt.isna()].copy()
            df["date"] = dt.dt.strftime("%m/%d/%Y")
        except Exception:
            pass
        df = df.drop_duplicates(subset=["date","ball1","ball2","ball3","ball4","ball5","powerball"])
        return df.reset_index(drop=True)

    def _validate(self):
        req = {"date","ball1","ball2","ball3","ball4","ball5","powerball"}
        if not req.issubset(self.raw.columns):
            raise ValueError("Data missing required columns")
        if len(self.raw)==0:
            raise ValueError("No valid rows parsed")

    def get_numbers(self):
        return self.raw[["ball1","ball2","ball3","ball4","ball5"]].values

    def get_powerballs(self):
        return self.raw["powerball"].values