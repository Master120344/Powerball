import re
import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm

def _ints(s):
    return [int(x) for x in re.findall(r"\d+", s)]

class DataRepository:
    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose
        if path.lower().endswith(".pdf"):
            self.raw = self._load_pdf(path, verbose=verbose)
        elif path.lower().endswith(".csv"):
            self.raw = pd.read_csv(path)
        elif path.lower().endswith(".txt"):
            self.raw = self._load_txt(path, verbose=verbose)
        else:
            raise ValueError("Unsupported file format")
        self._validate()

    def _load_pdf(self, path, verbose=True):
        rows = []
        reader = PdfReader(path)
        for page in tqdm(reader.pages, desc="Parsing PDF pages", disable=not verbose):
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                line = line.strip()
                dmatch = re.match(r"^(\d{2}/\d{2}/\d{4})", line)
                if not dmatch:
                    continue
                date = dmatch.group(1)
                rest = line[dmatch.end():]
                nums = _ints(rest)
                if len(nums) >= 6:
                    rows.append([date] + nums[:5] + [nums[5]])
        df = pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])
        print(f"✅ Parsed {len(df)} draws from {path}")
        return df

    def _load_txt(self, path, verbose=True):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Parsing TXT lines", disable=not verbose):
                line = line.strip()
                dmatch = re.match(r"^(\d{2}/\d{2}/\d{4})", line)
                if not dmatch:
                    continue
                date = dmatch.group(1)
                rest = line[dmatch.end():]
                nums = _ints(rest)
                if len(nums) >= 6:
                    rows.append([date] + nums[:5] + [nums[5]])
        df = pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])
        print(f"✅ Parsed {len(df)} draws from {path}")
        return df

    def _validate(self):
        if self.raw.empty:
            raise ValueError("No valid rows parsed")
        required = ["date","ball1","ball2","ball3","ball4","ball5","powerball"]
        for col in required:
            if col not in self.raw.columns:
                raise ValueError("Missing required columns in parsed data")
        self.raw["date"] = pd.to_datetime(self.raw["date"], errors="coerce")
        self.raw = self.raw.dropna(subset=["date"]).reset_index(drop=True)

    def get_draws(self):
        return self.raw.copy()

    def get_numbers_only(self):
        return self.raw[["ball1","ball2","ball3","ball4","ball5","powerball"]].values.tolist()