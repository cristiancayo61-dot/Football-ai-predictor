import numpy as np
import pandas as pd
from math import exp, factorial

REQUIRED = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]

def demo_data():
    # Datos sintéticos solo para probar que la aplicación funciona.
    teams = ["Madrid FC", "Barcelona CF", "Valencia", "Sevilla", "Bilbao", "Villarreal"]
    rows = []
    rng = np.random.default_rng(17)
    for i in range(120):
        h = teams[i % len(teams)]
        a = teams[(i * 3 + 1) % len(teams)]
        if h == a:
            a = teams[(teams.index(a) + 1) % len(teams)]
        hg = int(rng.poisson(1.55 if h in ["Madrid FC","Barcelona CF"] else 1.25))
        ag = int(rng.poisson(1.15 if a in ["Madrid FC","Barcelona CF"] else 1.0))
        rows.append({"Date": f"2025-{(i%12)+1:02d}-{(i%27)+1:02d}", "HomeTeam": h, "AwayTeam": a,
                     "FTHG": hg, "FTAG": ag, "HC": int(rng.poisson(5.2)), "AC": int(rng.poisson(4.2)),
                     "HY": int(rng.poisson(2.0)), "AY": int(rng.poisson(2.1))})
    return pd.DataFrame(rows)

def prepare_data(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        return None, "Faltan columnas obligatorias: " + ", ".join(missing)
    df = raw.copy()
    for c in ["FTHG", "FTAG", "HC", "AC", "HY", "AY"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["HomeTeam","AwayTeam","FTHG","FTAG"])
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    if len(df) < 10:
        return None, "Hay menos de 10 partidos válidos. Carga más datos históricos."
    return df.reset_index(drop=True), None

def poisson_pmf(k, lam):
    return exp(-lam) * (lam ** k) / factorial(k)

def predict_match(df, home, away):
    h = df[df.HomeTeam == home]
    a = df[df.AwayTeam == away]
    league_home = max(df.FTHG.mean(), 0.1)
    league_away = max(df.FTAG.mean(), 0.1)

    h_scored = h.FTHG.mean() if len(h) else league_home
    h_conceded = h.FTAG.mean() if len(h) else league_away
    a_scored = a.FTAG.mean() if len(a) else league_away
    a_conceded = a.FTHG.mean() if len(a) else league_home

    lam_h = max(0.15, (h_scored / league_home) * (a_conceded / league_home) * league_home)
    lam_a = max(0.15, (a_scored / league_away) * (h_conceded / league_away) * league_away)

    max_goals = 9
    p_home = p_draw = p_away = p_btts = p_over15 = p_over25 = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i > 0 and j > 0: p_btts += p
            if i + j >= 2: p_over15 += p
            if i + j >= 3: p_over25 += p

    out = {"home_win":p_home, "draw":p_draw, "away_win":p_away, "btts":p_btts,
           "over15":p_over15, "over25":p_over25, "lambda_home":lam_h, "lambda_away":lam_a}
    if all(c in df.columns for c in ["HC","AC","HY","AY"]):
        out["corners_home"] = float(df[df.HomeTeam == home].HC.mean()) if len(h) else float(df.HC.mean())
        out["corners_away"] = float(df[df.AwayTeam == away].AC.mean()) if len(a) else float(df.AC.mean())
        out["cards_home"] = float(df[df.HomeTeam == home].HY.mean()) if len(h) else float(df.HY.mean())
        out["cards_away"] = float(df[df.AwayTeam == away].AY.mean()) if len(a) else float(df.AY.mean())
    return out
