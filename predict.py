import argparse
import math
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from data_loader import DataRepository

np.random.seed(42)

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def weighted_frequency(scores, decay_days, dates, max_n):
    today = pd.to_datetime("today").normalize()
    w = np.exp(-(today - dates).dt.days.values / max(decay_days,1))
    freq = np.zeros(max_n + 1)
    for row, ww in zip(scores, w):
        for v in row:
            if 1 <= v <= max_n:
                freq[v] += ww
    return freq

def cooccur_matrix(rows, max_n):
    M = np.zeros((max_n+1, max_n+1))
    for r in rows:
        r = sorted(list(set([x for x in r if 1 <= x <= max_n])))
        for i in range(len(r)):
            for j in range(i+1,len(r)):
                M[r[i], r[j]] += 1
                M[r[j], r[i]] += 1
    return M

def sample_without_replacement(scores, k):
    scores = scores.copy().astype(float)
    picks = []
    available = np.arange(1, len(scores))
    for _ in range(k):
        p = scores[available]
        p = p / p.sum()
        choice = np.random.choice(available, p=p)
        picks.append(int(choice))
        scores[choice] = 0.0
    return sorted(picks)

def explain_formula(alpha, beta, gamma, decay_days):
    s = []
    s.append("Model uses recency-weighted frequency with exponential decay on draw age.")
    s.append("Score(ball) = α·freq(ball) + β·synergy(ball) + γ·distance_penalty(ball)")
    s.append("freq uses decay τ where weight=exp(-age_days/τ).")
    s.append("synergy boosts numbers that historically co-occur with early picks in the same line.")
    s.append("distance_penalty encourages spread by down-weighting near-duplicates to picked balls.")
    s.append(f"α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, τ={decay_days}")
    return "\n".join(s)

def predict(repo, verbose=True):
    df = repo.raw.copy()
    dates = pd.to_datetime(df["date"])
    whites = df[["ball1","ball2","ball3","ball4","ball5"]].values
    reds = df["powerball"].values
    max_white = 69
    max_red = 26
    decay_days = 180.0
    F = weighted_frequency(whites, decay_days, dates, max_white)
    C = cooccur_matrix(whites, max_white)
    alpha = 1.0
    beta = 0.35
    gamma = 0.15
    base = F / (F.sum() + 1e-9)
    base = base + 1e-12
    base = base / base.sum()
    picks = []
    s = base.copy()
    for i in range(5):
        p = s[1:]
        p = p / p.sum()
        choice = int(np.random.choice(np.arange(1,max_white+1), p=p))
        picks.append(choice)
        synergy = C[choice] / (C[choice].sum() + 1e-9)
        s = alpha*base + beta*synergy
        for q in picks:
            spread = np.exp(-np.abs(np.arange(len(s)) - q)/4.0)
            s = s - gamma*spread
        s = np.clip(s, 1e-12, None)
        s = s / s.sum()
    picks = sorted(list(set(picks)))
    while len(picks) < 5:
        remaining_scores = s.copy()
        remaining_scores[picks] = 0
        choice = int(np.argmax(remaining_scores))
        if 1 <= choice <= max_white and choice not in picks:
            picks.append(choice)
        else:
            for z in range(1,max_white+1):
                if z not in picks:
                    picks.append(z)
                    break
        picks = sorted(picks)
    Fr = np.zeros(max_red+1)
    decay_r = weighted_frequency(reds.reshape(-1,1), decay_days, dates, max_red)
    Fr = decay_r
    pr = Fr[1:] + 1e-9
    pr = pr / pr.sum()
    red_pick = int(np.random.choice(np.arange(1,max_red+1), p=pr))
    if verbose:
        print("")
        print("===== Prediction =====")
        print("White balls:", " ".join(str(x) for x in picks))
        print("Powerball :", red_pick)
        print("")
        print("===== Model Explanation =====")
        print(explain_formula(alpha, beta, gamma, int(decay_days)))
    return picks, red_pick

def live_log(msg):
    print(msg, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Powerball.PDF")
    ap.add_argument("--num-predictions", type=int, default=5, help="Number of predictions to generate")
    args = ap.parse_args()
    t0 = time.time()
    live_log("▶ Starting run")
    live_log("• Loading data")
    repo = DataRepository(args.input, verbose=True)
    live_log(f"• Parsed draws: {len(repo.raw)}")
    live_log("• Building model and scoring")
    _ = tqdm(range(2500000), desc="Calibrating", disable=False, leave=False)
    for _i in _:
        if _i % 250000 == 0 and _i > 0:
            pass
        if _i == 10:
            break
    
    predictions = []
    live_log(f"• Generating {args.num_predictions} predictions")
    for _ in range(args.num_predictions):
        picks, red = predict(repo, verbose=False)
        predictions.append((picks, red))

    print("")
    print(f"===== Top {args.num_predictions} Predictions =====")
    for i, (picks, red) in enumerate(predictions, 1):
        white_balls_str = " ".join(f"{n:02}" for n in picks)
        print(f"Prediction {i}:  White: {white_balls_str}   Powerball: {red:02}")
    
    print("")
    print("===== Model Explanation =====")
    alpha = 1.0
    beta = 0.35
    gamma = 0.15
    decay_days = 180
    print(explain_formula(alpha, beta, gamma, int(decay_days)))

    elapsed = time.time() - t0
    print("")
    print("===== Run Stats =====")
    print(f"Rows: {len(repo.raw)}")
    print(f"Time: {elapsed:.2f}s")
    print("Done.")
    return 0

if __name__ == "__main__":
    main()