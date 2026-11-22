"""Segundo modelo de predicción 1X2 basado en Machine Learning.

Construye características únicamente con partidos anteriores a cada encuentro.
La validación utiliza un corte cronológico; no se mezclan partidos futuros en el
entrenamiento de la evaluación.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "home_gf", "home_ga", "home_points", "home_win_rate",
    "home_venue_gf", "home_venue_ga", "home_n",
    "away_gf", "away_ga", "away_points", "away_win_rate",
    "away_venue_gf", "away_venue_ga", "away_n",
    "league_home_goals", "league_away_goals", "league_n",
]
CLASS_TO_KEY = {2: "home_win", 1: "draw", 0: "away_win"}


def _safe_mean(values, fallback):
    return float(np.mean(values)) if values else float(fallback)


def _team_features(history, venue, fallback_gf, fallback_ga):
    items = list(history)
    if not items:
        return {
            "gf": fallback_gf, "ga": fallback_ga, "points": 1.0,
            "win_rate": 1 / 3, "venue_gf": fallback_gf,
            "venue_ga": fallback_ga, "n": 0,
        }
    venue_items = [x for x in items if x[3] == venue]
    if not venue_items:
        venue_items = items
    return {
        "gf": _safe_mean([x[0] for x in items], fallback_gf),
        "ga": _safe_mean([x[1] for x in items], fallback_ga),
        "points": _safe_mean([x[2] for x in items], 1.0),
        "win_rate": float(sum(x[2] == 3 for x in items) / len(items)),
        "venue_gf": _safe_mean([x[0] for x in venue_items], fallback_gf),
        "venue_ga": _safe_mean([x[1] for x in venue_items], fallback_ga),
        "n": len(items),
    }


def _features_for_match(league, home, away, histories, league_stats):
    count, sum_home, sum_away = league_stats
    fallback_home = sum_home / count if count else 1.35
    fallback_away = sum_away / count if count else 1.10
    h = _team_features(histories[(league, home)], True, fallback_home, fallback_away)
    a = _team_features(histories[(league, away)], False, fallback_away, fallback_home)
    return {
        "league": str(league),
        "home_gf": h["gf"], "home_ga": h["ga"],
        "home_points": h["points"], "home_win_rate": h["win_rate"],
        "home_venue_gf": h["venue_gf"], "home_venue_ga": h["venue_ga"],
        "home_n": h["n"],
        "away_gf": a["gf"], "away_ga": a["ga"],
        "away_points": a["points"], "away_win_rate": a["win_rate"],
        "away_venue_gf": a["venue_gf"], "away_venue_ga": a["venue_ga"],
        "away_n": a["n"],
        "league_home_goals": fallback_home,
        "league_away_goals": fallback_away,
        "league_n": count,
    }


def _build_dataset(raw):
    required = {"League", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if raw is None or raw.empty or not required.issubset(raw.columns):
        raise ValueError("El histórico no contiene todas las columnas necesarias para entrenar el modelo.")
    df = raw[list(required)].copy()
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in ("FTHG", "FTAG"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_date", "League", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = df[(df["FTHG"] >= 0) & (df["FTAG"] >= 0)].sort_values("_date")
    if df.empty:
        raise ValueError("No hay resultados históricos fechados y válidos.")
    histories = defaultdict(lambda: deque(maxlen=10))
    league_stats = defaultdict(lambda: [0, 0.0, 0.0])
    records, labels, dates = [], [], []

    # Todos los partidos de una misma fecha se featurizan antes de actualizar
    # el historial: evita que un resultado del mismo día filtre información.
    for (league, match_date), day in df.groupby(["League", "_date"], sort=True):
        for _, row in day.iterrows():
            home, away = str(row["HomeTeam"]), str(row["AwayTeam"])
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            stats = league_stats[str(league)]
            hf = _features_for_match(str(league), home, away, histories, stats)
            if hf["home_n"] >= 3 and hf["away_n"] >= 3:
                records.append(hf)
                labels.append(2 if hg > ag else (1 if hg == ag else 0))
                dates.append(match_date)
        # Actualizar al terminar el bloque completo de esa fecha.
        for _, row in day.iterrows():
            home, away = str(row["HomeTeam"]), str(row["AwayTeam"])
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            hp = 3 if hg > ag else (1 if hg == ag else 0)
            ap = 3 if ag > hg else (1 if hg == ag else 0)
            histories[(str(league), home)].append((hg, ag, hp, True))
            histories[(str(league), away)].append((ag, hg, ap, False))
            stats = league_stats[str(league)]
            stats[0] += 1
            stats[1] += hg
            stats[2] += ag

    X = pd.DataFrame(records)
    y = np.asarray(labels, dtype=int)
    if len(X) < 120 or len(set(y.tolist())) < 3:
        raise ValueError(
            f"Muestra insuficiente para entrenar con garantías básicas: {len(X)} partidos útiles. "
            "Se necesitan al menos 120 y presencia de los tres resultados 1X2."
        )
    latest_states = (histories, dict(league_stats))
    return X, y, pd.to_datetime(pd.Series(dates)), latest_states


def _new_model():
    preprocess = ColumnTransformer(
        [("league", OneHotEncoder(handle_unknown="ignore"), ["league"])],
        remainder="passthrough",
    )
    classifier = RandomForestClassifier(
        n_estimators=350,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("features", preprocess), ("model", classifier)])


@st.cache_resource(show_spinner=False)
def _train_cached(raw):
    X, y, dates, latest_states = _build_dataset(raw)
    order = np.argsort(dates.to_numpy())
    X, y = X.iloc[order].reset_index(drop=True), y[order]
    dates = dates.iloc[order].reset_index(drop=True)
    split = max(1, min(len(X) - 1, int(len(X) * 0.80)))
    cutoff = dates.iloc[split]
    train_mask = dates < cutoff
    test_mask = dates >= cutoff
    if train_mask.sum() < 80 or test_mask.sum() < 20:
        train_mask = np.arange(len(X)) < split
        test_mask = ~train_mask
    else:
        train_mask = np.asarray(train_mask)
        test_mask = np.asarray(test_mask)

    validation_model = _new_model()
    validation_model.fit(X.loc[train_mask], y[train_mask])
    test_prob = validation_model.predict_proba(X.loc[test_mask])
    test_pred = validation_model.predict(X.loc[test_mask])
    metrics = {
        "accuracy": float(accuracy_score(y[test_mask], test_pred)),
        "log_loss": float(log_loss(y[test_mask], test_prob, labels=[0, 1, 2])),
        "validation_matches": int(test_mask.sum()),
        "training_matches": int(train_mask.sum()),
        "cutoff": str(cutoff.date()) if hasattr(cutoff, "date") else str(cutoff),
    }
    final_model = _new_model()
    final_model.fit(X, y)
    return final_model, metrics, latest_states, len(X)


def predict_match(raw, league, home, away):
    """Devuelve probabilidades 1X2, evaluación temporal y tamaño de muestra."""
    model, metrics, latest_states, sample_size = _train_cached(raw)
    histories, league_stats = latest_states
    features = _features_for_match(str(league), str(home), str(away), histories, league_stats)
    if features["home_n"] == 0 or features["away_n"] == 0:
        raise ValueError("No hay historial suficiente de uno de los equipos en esta competición.")
    row = pd.DataFrame([features])
    probabilities = model.predict_proba(row)[0]
    class_probs = {int(c): float(p) for c, p in zip(model.classes_, probabilities)}
    result = {
        CLASS_TO_KEY[c]: class_probs.get(c, 0.0) for c in (2, 1, 0)
    }
    result["home_matches"] = int(features["home_n"])
    result["away_matches"] = int(features["away_n"])
    result["training_matches"] = int(sample_size)
    result["validation"] = metrics
    return result
