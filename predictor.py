import numpy as np
import pandas as pd
from math import exp, factorial

REQUIRED = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def demo_data():
    """Datos sintéticos solo para comprobar la aplicación. No usar para apostar."""
    teams = ["Madrid FC", "Barcelona CF", "Valencia", "Sevilla", "Bilbao", "Villarreal"]
    rows = []
    rng = np.random.default_rng(17)
    for i in range(120):
        home = teams[i % len(teams)]
        away = teams[(i * 3 + 1) % len(teams)]
        if home == away:
            away = teams[(teams.index(away) + 1) % len(teams)]
        hg = int(rng.poisson(1.55 if home in ["Madrid FC", "Barcelona CF"] else 1.25))
        ag = int(rng.poisson(1.15 if away in ["Madrid FC", "Barcelona CF"] else 1.0))
        rows.append({
            "Date": f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag,
            "HC": int(rng.poisson(5.2)), "AC": int(rng.poisson(4.2)),
            "HY": int(rng.poisson(2.0)), "AY": int(rng.poisson(2.1)),
        })
    return pd.DataFrame(rows)


def prepare_data(raw):
    missing = [col for col in REQUIRED if col not in raw.columns]
    if missing:
        return None, "Faltan columnas obligatorias: " + ", ".join(missing)
    df = raw.copy()
    for col in ["FTHG", "FTAG", "HC", "AC", "HY", "AY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df = df[(df["HomeTeam"] != "") & (df["AwayTeam"] != "")]
    if "Date" in df.columns:
        df["_date_sort"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.sort_values("_date_sort", na_position="first").drop(columns=["_date_sort"])
    if len(df) < 10:
        return None, "Hay menos de 10 partidos válidos. Carga más datos históricos."
    return df.reset_index(drop=True), None


def poisson_pmf(k, lam):
    return exp(-lam) * (lam ** k) / factorial(k)


def compare_odds(probability, decimal_odds):
    probability = float(np.clip(probability, 0.0, 1.0))
    decimal_odds = max(float(decimal_odds), 1.000001)
    return {
        "implied_probability": 1.0 / decimal_odds,
        "fair_odds": (1.0 / probability) if probability > 0 else float("inf"),
        "expected_value": probability * decimal_odds - 1.0,
    }


def predict_match(df, home, away):
    """Modelo Poisson sencillo a partir de medias históricas local/visitante."""
    home_rows = df[df.HomeTeam == home].tail(10)
    away_rows = df[df.AwayTeam == away].tail(10)

    league_home = max(float(df.FTHG.mean()), 0.1)
    league_away = max(float(df.FTAG.mean()), 0.1)

    h_scored = float(home_rows.FTHG.mean()) if len(home_rows) else league_home
    h_conceded = float(home_rows.FTAG.mean()) if len(home_rows) else league_away
    a_scored = float(away_rows.FTAG.mean()) if len(away_rows) else league_away
    a_conceded = float(away_rows.FTHG.mean()) if len(away_rows) else league_home

    lam_home = max(0.15, (h_scored / league_home) * (a_conceded / league_home) * league_home)
    lam_away = max(0.15, (a_scored / league_away) * (h_conceded / league_away) * league_away)

    max_goals = 10
    p_home = p_draw = p_away = p_btts = 0.0
    p_over15 = p_over25 = p_over35 = 0.0
    score_probs = []

    for goals_home in range(max_goals + 1):
        for goals_away in range(max_goals + 1):
            prob = poisson_pmf(goals_home, lam_home) * poisson_pmf(goals_away, lam_away)
            score_probs.append((f"{goals_home}-{goals_away}", prob))
            if goals_home > goals_away:
                p_home += prob
            elif goals_home == goals_away:
                p_draw += prob
            else:
                p_away += prob
            if goals_home > 0 and goals_away > 0:
                p_btts += prob
            total = goals_home + goals_away
            if total >= 2:
                p_over15 += prob
            if total >= 3:
                p_over25 += prob
            if total >= 4:
                p_over35 += prob

    out = {
        "home_win": p_home, "draw": p_draw, "away_win": p_away,
        "double_chance_1x": p_home + p_draw,
        "double_chance_x2": p_draw + p_away,
        "double_chance_12": p_home + p_away,
        "btts": p_btts, "over15": p_over15, "over25": p_over25,
        "over35": p_over35, "lambda_home": lam_home, "lambda_away": lam_away,
        "top_scores": sorted(score_probs, key=lambda x: x[1], reverse=True)[:5],
    }

    if all(col in df.columns for col in ["HC", "AC", "HY", "AY"]):
        home_all = df[df.HomeTeam == home]
        away_all = df[df.AwayTeam == away]
        out["corners_home"] = float(home_all.HC.tail(10).mean()) if len(home_all) else float(df.HC.mean())
        out["corners_away"] = float(away_all.AC.tail(10).mean()) if len(away_all) else float(df.AC.mean())
        out["cards_home"] = float(home_all.HY.tail(10).mean()) if len(home_all) else float(df.HY.mean())
        out["cards_away"] = float(away_all.AY.tail(10).mean()) if len(away_all) else float(df.AY.mean())

    return out
