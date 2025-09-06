import re
import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm

def _ints(s):
    return [int(x) for x in re.findall(r"\d+", s)]

def _parse_whites_recursive(s, count, min_val):
    if count == 0:
        return [] if s == "" else None
    if not s or count < 0:
        return None
    if len(s) >= 1:
        num = int(s[0])
        if num > min_val and 1 <= num <= 69:
            result = _parse_whites_recursive(s[1:], count - 1, num)
            if result is not None:
                return [num] + result
    if len(s) >= 2:
        num = int(s[:2])
        if num > min_val and 1 <= num <= 69:
            result = _parse_whites_recursive(s[2:], count - 1, num)
            if result is not None:
                return [num] + result
    return None

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
        for page_index, page in enumerate(tqdm(reader.pages, desc="Parsing PDF pages", disable=not verbose)):
            text = page.extract_text()
            if not text:
                continue
            if verbose and page_index == 0:
                print("\n--- DEBUG PREVIEW (first 10 lines from PDF) ---")
                for preview_line in text.splitlines()[:10]:
                    print("DEBUG:", preview_line)
                print("--- END DEBUG PREVIEW ---\n")
            for line in text.splitlines():
                line = line.strip()
                dmatch = re.match(r"^(\d{1,2}\s*[/\- ]\s*\d{1,2}\s*[/\- ]\s*\d{4})", line)
                if not dmatch:
                    continue
                date = re.sub(r"\s+", "", dmatch.group(1).replace(" ", "").replace("-", "/"))
                rest = line[dmatch.end():].strip()
                parts = rest.split()
                if not parts:
                    continue
                num_blob = "".join(re.findall(r"\d+", parts[0]))
                parsed_successfully = False
                if len(num_blob) > 7:
                    try:
                        pb = int(num_blob[-2:])
                        whites_blob = num_blob[:-2]
                        if 1 <= pb <= 26:
                            whites = _parse_whites_recursive(whites_blob, 5, 0)
                            if whites and len(whites) == 5:
                                rows.append([date] + whites + [pb])
                                parsed_successfully = True
                    except (ValueError, IndexError):
                        pass
                if not parsed_successfully and len(num_blob) > 5:
                    try:
                        pb = int(num_blob[-1])
                        whites_blob = num_blob[:-1]
                        if 1 <= pb <= 26:
                            whites = _parse_whites_recursive(whites_blob, 5, 0)
                            if whites and len(whites) == 5:
                                rows.append([date] + whites + [pb])
                    except (ValueError, IndexError):
                        pass
        df = pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])
        print(f"✅ Parsed {len(df)} draws from {path}")
        return df

    def _load_txt(self, path, verbose=True):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Parsing TXT lines", disable=not verbose):
                line = line.strip()
                dmatch = re.match(r"^(\d{1,2}\s*[/\- ]\s*\d{1,2}\s*[/\- ]\s*\d{4})", line)
                if not dmatch:
                    continue
                date = re.sub(r"\s+", "", dmatch.group(1).replace(" ", "").replace("-", "/"))
                rest = line[dmatch.end():].strip()
                nums = _ints(rest)
                if len(nums) >= 6:
                    rows.append([date] + nums[:5] + [nums[5]])
                    continue
                else:
                    parts = rest.split()
                    if not parts:
                        continue
                    num_blob = "".join(re.findall(r"\d+", parts[0]))
                    parsed_successfully = False
                    if len(num_blob) > 7:
                        try:
                            pb = int(num_blob[-2:])
                            whites_blob = num_blob[:-2]
                            if 1 <= pb <= 26:
                                whites = _parse_whites_recursive(whites_blob, 5, 0)
                                if whites and len(whites) == 5:
                                    rows.append([date] + whites + [pb])
                                    parsed_successfully = True
                        except (ValueError, IndexError):
                            pass
                    if not parsed_successfully and len(num_blob) > 5:
                        try:
                            pb = int(num_blob[-1])
                            whites_blob = num_blob[:-1]
                            if 1 <= pb <= 26:
                                whites = _parse_whites_recursive(whites_blob, 5, 0)
                                if whites and len(whites) == 5:
                                    rows.append([date] + whites + [pb])
                        except (ValueError, IndexError):
                            pass
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