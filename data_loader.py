import os
import re
import pandas as pd
from tqdm import tqdm

def _ints(s):
    return [int(x) for x in re.findall(r"\d+", s)]

class DataRepository:
    def __init__(self, path, verbose=True):
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            self.raw = self._load_pdf(path, verbose=verbose)
        elif ext == ".txt":
            self.raw = self._load_txt(path, verbose=verbose)
        elif ext == ".csv":
            self.raw = self._load_csv(path, verbose=verbose)
        else:
            raise ValueError("Unsupported file format")
        self._coerce()
        self._validate()

    def _load_csv(self, path, verbose=True):
        df = pd.read_csv(path, dtype=str, engine="python")
        cols = {c.lower().strip(): c for c in df.columns}
        date_col = None
        for k in ("date","draw date","draw_date","drawing","draw"):
            if k in cols:
                date_col = cols[k]
                break
        num_col = None
        for k in ("winning numbers","numbers","winning_numbers","balls"):
            if k in cols:
                num_col = cols[k]
                break
        pb_col = None
        for k in ("powerball","pb","power ball"):
            if k in cols:
                pb_col = cols[k]
                break
        rows = []
        if num_col is not None:
            tmp = df[[x for x in [date_col,num_col,pb_col] if x is not None]].copy()
            for _, r in tmp.iterrows():
                d = str(r[date_col]) if date_col is not None else None
                nums = _ints(str(r[num_col]))
                pb = None
                if pb_col is not None:
                    pbv = _ints(str(r[pb_col]))
                    if pbv:
                        pb = pbv[0]
                elif len(nums) >= 6:
                    pb = nums[5]
                if d and len(nums) >= 5 and pb is not None:
                    rows.append([d]+nums[:5]+[pb])
            return pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])
        bcols = []
        for i in range(1,6):
            for k in (f"ball{i}",f"b{i}",f"n{i}"):
                if k in cols:
                    bcols.append(cols[k])
                    break
        if date_col and pb_col and len(bcols)==5:
            sub = df[[date_col]+bcols+[pb_col]].copy()
            for _, r in sub.iterrows():
                d = str(r[date_col])
                nums = []
                ok = True
                for c in bcols:
                    v = _ints(str(r[c]))
                    if not v:
                        ok=False
                        break
                    nums.append(v[0])
                if not ok:
                    continue
                pv = _ints(str(r[pb_col]))
                if not pv:
                    continue
                rows.append([d]+nums+[pv[0]])
        return pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])

    def _load_txt(self, path, verbose=True):
        rows = []
        with open(path,"r") as f:
            lines = f.readlines()
        for s in tqdm(lines, desc="Parsing TXT", disable=not verbose):
            s = s.strip()
            if not s:
                continue
            m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", s)
            if not m:
                continue
            d = m.group(1)
            rest = (s[:m.start()]+" "+s[m.end():]).replace(",", " ").replace("|", " ").replace("—"," ").replace("-"," ")
            nums = _ints(rest)
            if len(nums) >= 6:
                rows.append([d]+nums[:5]+[nums[5]])
        return pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])

    def _load_pdf(self, path, verbose=True):
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer, LTTextLine
        rows = []
        pages = list(extract_pages(path))
        for page in tqdm(pages, desc="Parsing PDF pages", disable=not verbose):
            text_lines = []
            for el in page:
                if isinstance(el, LTTextContainer):
                    for line in el:
                        if isinstance(line, LTTextLine):
                            s = line.get_text().strip()
                            if s:
                                text_lines.append(s)
            for s in text_lines:
                if not re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s):
                    continue
                dmatch = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", s)
                d = dmatch.group(1)
                rest = (s[:dmatch.start()]+" "+s[dmatch.end():]).replace(",", " ").replace("|", " ").replace("—"," ").replace("-"," ")
                nums = _ints(rest)
                if len(nums) >= 6:
                    rows.append([d]+nums[:5]+[nums[5]])
        return pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])

    def _coerce(self):
        df = self.raw.dropna().copy()
        df["date"] = df["date"].astype(str).str.strip()
        for c in ["ball1","ball2","ball3","ball4","ball5","powerball"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        try:
            dt = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
            df = df.loc[~dt.isna()].copy()
            df["date"] = dt.dt.strftime("%m/%d/%Y")
        except Exception:
            pass
        df = df.astype({"ball1":"int64","ball2":"int64","ball3":"int64","ball4":"int64","ball5":"int64","powerball":"int64"})
        df = df.drop_duplicates(subset=["date","ball1","ball2","ball3","ball4","ball5","powerball"]).reset_index(drop=True)
        self.raw = df

    def _validate(self):
        req = {"date","ball1","ball2","ball3","ball4","ball5","powerball"}
        if not req.issubset(set(self.raw.columns)):
            raise ValueError("Data missing required columns")
        if len(self.raw) == 0:
            raise ValueError("No valid rows parsed")

    def get_numbers(self):
        return self.raw[["ball1","ball2","ball3","ball4","ball5"]].values

    def get_powerballs(self):
        return self.raw["powerball"].values