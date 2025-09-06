import pandas as pd
import os

class DataRepository:
    def __init__(self, path):
        if path.endswith(".csv"):
            self.raw = pd.read_csv(path)
        elif path.endswith(".txt"):
            self.raw = self._load_txt(path)
        else:
            raise ValueError("Unsupported file format")
        self._validate()

    def _load_txt(self, path):
        rows = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 7:
                    date = parts[0]
                    balls = list(map(int, parts[1:6]))
                    powerball = int(parts[6])
                    rows.append([date] + balls + [powerball])
        return pd.DataFrame(rows, columns=["date","ball1","ball2","ball3","ball4","ball5","powerball"])

    def _validate(self):
        required = {"date","ball1","ball2","ball3","ball4","ball5","powerball"}
        if not required.issubset(set(self.raw.columns)):
            raise ValueError("Data missing required columns")

    def get_numbers(self):
        return self.raw[["ball1","ball2","ball3","ball4","ball5"]].values

    def get_powerballs(self):
        return self.raw["powerball"].values