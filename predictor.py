"""Motor de predicción de Football AI Predictor.

Los equipos se identifican por los ID oficiales de API-Football, no por nombres.
Las probabilidades son estimaciones Poisson y no garantizan rentabilidad.
"""
import math
import pandas as pd


def _poisson(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_over(lam, line):
    if lam is None or pd.isna(lam) or float(lam) < 0:
        return None
    cutoff = math.floor(float(line))
    return max(0.0, min(1.0, 1.0 - sum(_poisson(k, float(lam)) for k in range(cutoff + 1))))


def prepare_api_history(records):
    """Convierte fixtures finalizados de API-Football a la tabla del modelo."""
    rows = []
    for item in records or []:
        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        goals = item.get("goals") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        if home.get("id") is None or away.get("id") is None:
            continue
        if goals.get("home") is None or goals.get("away") is None:
            continue
        rows.append({
            "FixtureID": fixture.get("id"),
            "Date": fixture.get("date"),
            "HomeID": int(home["id"]), "AwayID": int(away["id"]),
            "HomeTeam": str(home.get("name") or home["id"]),
            "AwayTeam": str(away.get("name") or away["id"]),
            "FTHG": int(goals["home"]), "FTAG": int(goals["away"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    return df.dropna(subset=["HomeID", "AwayID", "FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)


def predict_match(df, home_id, away_id):
    if df is None or df.empty:
        raise ValueError("No hay resultados históricos disponibles para esta competición.")
    h = df[df["HomeID"] == int(home_id)]
    a = df[df["AwayID"] == int(away_id)]
    if h.empty or a.empty:
        raise ValueError("No hay suficientes partidos recientes del local en casa o del visitante fuera.")

    league_home = max(float(df["FTHG"].mean()), 0.15)
    league_away = max(float(df["FTAG"].mean()), 0.15)
    def blend(values, baseline):
        n = len(values)
        mean = float(values.mean()) if n else baseline
        weight = min(n / 8.0, 1.0)
        return mean * weight + baseline * (1 - weight)

    h_scored = blend(h["FTHG"], league_home)
    h_conceded = blend(h["FTAG"], league_away)
    a_scored = blend(a["FTAG"], league_away)
    a_conceded = blend(a["FTHG"], league_home)
    lam_h = max(0.15, min(4.0, (h_scored / league_home) * (a_conceded / league_home) * league_home))
    lam_a = max(0.15, min(4.0, (a_scored / league_away) * (h_conceded / league_away) * league_away))

    matrix = []
    p_home = p_draw = p_away = p_btts = p_over15 = p_over25 = p_over35 = 0.0
    for i in range(10):
        for j in range(10):
            p = _poisson(i, lam_h) * _poisson(j, lam_a)
            matrix.append((f"{i}-{j}", p))
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i and j: p_btts += p
            if i + j >= 2: p_over15 += p
            if i + j >= 3: p_over25 += p
            if i + j >= 4: p_over35 += p
    total = p_home + p_draw + p_away
    return {
        "home_win": p_home / total, "draw": p_draw / total, "away_win": p_away / total,
        "double_chance_1x": (p_home + p_draw) / total,
        "double_chance_x2": (p_draw + p_away) / total,
        "double_chance_12": (p_home + p_away) / total,
        "btts": p_btts, "over15": p_over15, "over25": p_over25, "over35": p_over35,
        "lambda_home": lam_h, "lambda_away": lam_a,
        "top_scores": sorted(matrix, key=lambda x: x[1], reverse=True)[:8],
        "home_matches": len(h), "away_matches": len(a), "auxiliary": {
            "corners": None, "fouls": None, "yellow_cards": None,
            "shots_on_target": None, "saves": None,
        },
    }


def compare_odds(probability, decimal_odds):
    probability = max(0.0, min(1.0, float(probability)))
    odds = max(1.01, float(decimal_odds))
    return {
        "implied_probability": 1.0 / odds,
        "fair_odds": (1.0 / probability) if probability > 0 else float("inf"),
        "expected_value": probability * odds - 1.0,
    }
