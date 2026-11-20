import json
import math
import re
import unicodedata
from pathlib import Path
import pandas as pd


def _normalise(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "", value)


ALIASES = {
    "athleticbilbao": ["athbilbao", "athleticclub", "bilbao"],
    "atleticodemadrid": ["atleticomadrid", "atletico"],
    "realbetis": ["betis"],
    "realsociedad": ["sociedad"],
    "deportivoalaves": ["alaves"],
    "leganes": ["cdleganes"],
    "fcandorra": ["andorra"],
    "sportinggijon": ["sporting"],
    "realvalladolid": ["valladolid"],
    "manchestercity": ["mancity"],
    "manchesterunited": ["manutd"],
    "tottenhamhotspur": ["tottenham", "spurs"],
    "newcastleunited": ["newcastle"],
    "nottinghamforest": ["nottmforest"],
    "brightonandhovealbion": ["brighton"],
    "parissaintgermain": ["psg"],
    "internazionale": ["inter"],
    "acmilan": ["milan"],
    "asroma": ["roma"],
    "bayernmunich": ["bayern"],
    "borussiadortmund": ["dortmund"],
    "rbleipzig": ["leipzig"],
    "psveindhoven": ["psv"],
    "ajax": ["afcajax"],
    "feyenoord": ["feyenoordrotterdam"],
}

def _load_alias_groups():
    """Carga equivalencias externas desde Data/equipos_equivalencias.json."""
    groups = []
    json_path = Path(__file__).resolve().parent / "Data" / "equipos_equivalencias.json"
    try:
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        for league_mapping in payload.get("leagues", {}).values():
            for canonical, variants in league_mapping.items():
                names = {str(canonical)}
                names.update(str(item) for item in variants if item)
                normalized = {_normalise(item) for item in names if _normalise(item)}
                if normalized:
                    groups.append(normalized)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        # Si el archivo no existe o está dañado, se mantienen las equivalencias antiguas.
        pass

    # Conserva también las equivalencias históricas que ya tenía el programa.
    for canonical, variants in ALIASES.items():
        names = {canonical}
        names.update(str(item) for item in variants)
        normalized = {_normalise(item) for item in names if _normalise(item)}
        if normalized:
            groups.append(normalized)
    return groups


def _alias_candidates(target):
    return [group for group in _load_alias_groups() if target in group]



def prepare_data(raw):
    if raw is None or len(raw) == 0:
        return pd.DataFrame(), "El archivo no contiene partidos."
    df = raw.copy()
    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame(), "Faltan columnas obligatorias: " + ", ".join(missing)
    numeric = ["FTHG", "FTAG", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR", "HST", "AST"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    if "Date" in df.columns:
        df["_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.sort_values("_date")
    return df.reset_index(drop=True), None


def match_team_name(name, df):
    """Devuelve el nombre exacto usado por el histórico, usando equivalencias seguras."""
    if df is None or df.empty or not name:
        return None

    teams = sorted(set(df["HomeTeam"].dropna()).union(set(df["AwayTeam"].dropna())))
    target = _normalise(name)
    if not target:
        return None

    normalized_teams = {_normalise(team): team for team in teams}

    # 1) Coincidencia exacta tras quitar tildes, espacios y signos.
    if target in normalized_teams:
        return normalized_teams[target]

    # 2) Coincidencia por cualquiera de las variantes del archivo JSON o aliases antiguos.
    groups = _alias_candidates(target)
    candidate_names = set()
    for group in groups:
        candidate_names.update(group)
    candidates = {normalized_teams[item] for item in candidate_names if item in normalized_teams}
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        return None

    # 3) Fallback parcial solo si existe una única coincidencia inequívoca.
    partial = [
        team for team in teams
        if target and (target in _normalise(team) or _normalise(team) in target)
    ]
    return partial[0] if len(partial) == 1 else None



def _poisson(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_over(lam, line):
    """Probability that a Poisson count is strictly greater than a half-goal line."""
    if lam is None or pd.isna(lam) or lam < 0:
        return None
    cutoff = math.floor(float(line))
    cumulative = sum(_poisson(k, float(lam)) for k in range(cutoff + 1))
    return max(0.0, min(1.0, 1.0 - cumulative))


def _blended_mean(team_values, league_values, minimum=8):
    team_values = pd.to_numeric(team_values, errors="coerce").dropna()
    league_values = pd.to_numeric(league_values, errors="coerce").dropna()
    if len(league_values) < 20 or len(team_values) < 4:
        return None
    weight = min(len(team_values) / minimum, 1.0)
    return float(team_values.mean() * weight + league_values.mean() * (1 - weight))


def _auxiliary_markets(df, home, away):
    """Team-level estimates from Football-Data fields; never fabricates player props."""
    definitions = {
        "corners": ("HC", "AC", "Córners"),
        "fouls": ("HF", "AF", "Faltas cometidas"),
        "yellow_cards": ("HY", "AY", "Tarjetas amarillas"),
        "shots_on_target": ("HST", "AST", "Tiros a puerta"),
    }
    home_rows = df[df["HomeTeam"] == home]
    away_rows = df[df["AwayTeam"] == away]
    result = {}
    for key, (home_col, away_col, label) in definitions.items():
        if home_col not in df.columns or away_col not in df.columns:
            result[key] = None
            continue
        home_mean = _blended_mean(home_rows[home_col], df[home_col])
        away_mean = _blended_mean(away_rows[away_col], df[away_col])
        if home_mean is None or away_mean is None:
            result[key] = None
            continue
        total = home_mean + away_mean
        result[key] = {"label": label, "home": home_mean, "away": away_mean, "total": total}

    # Proxy for saves: opponent shots on target minus goals conceded.
    if "HST" in df.columns and "AST" in df.columns:
        home_saves_values = (pd.to_numeric(home_rows["AST"], errors="coerce") - pd.to_numeric(home_rows["FTAG"], errors="coerce")).clip(lower=0)
        away_saves_values = (pd.to_numeric(away_rows["HST"], errors="coerce") - pd.to_numeric(away_rows["FTHG"], errors="coerce")).clip(lower=0)
        home_saves = _blended_mean(home_saves_values, (pd.to_numeric(df["AST"], errors="coerce") - pd.to_numeric(df["FTAG"], errors="coerce")).clip(lower=0))
        away_saves = _blended_mean(away_saves_values, (pd.to_numeric(df["HST"], errors="coerce") - pd.to_numeric(df["FTHG"], errors="coerce")).clip(lower=0))
        result["saves"] = {"label": "Paradas (estimación por equipo)", "home": home_saves, "away": away_saves, "total": (home_saves + away_saves) if home_saves is not None and away_saves is not None else None} if home_saves is not None and away_saves is not None else None
    else:
        result["saves"] = None
    return result


def predict_match(df, home, away):
    if df.empty:
        raise ValueError("No hay resultados históricos para esta competición.")
    h = df[df["HomeTeam"] == home]
    a = df[df["AwayTeam"] == away]
    league_home = max(float(df["FTHG"].mean()), 0.15)
    league_away = max(float(df["FTAG"].mean()), 0.15)
    h_scored = (float(h["FTHG"].mean()) * min(len(h) / 8, 1) + league_home * max(0, 1 - len(h) / 8)) if len(h) else league_home
    h_conceded = (float(h["FTAG"].mean()) * min(len(h) / 8, 1) + league_away * max(0, 1 - len(h) / 8)) if len(h) else league_away
    a_scored = (float(a["FTAG"].mean()) * min(len(a) / 8, 1) + league_away * max(0, 1 - len(a) / 8)) if len(a) else league_away
    a_conceded = (float(a["FTHG"].mean()) * min(len(a) / 8, 1) + league_home * max(0, 1 - len(a) / 8)) if len(a) else league_home
    lam_h = max(0.15, min(4.0, (h_scored / league_home) * (a_conceded / league_home) * league_home))
    lam_a = max(0.15, min(4.0, (a_scored / league_away) * (h_conceded / league_away) * league_away))

    matrix = []
    p_home = p_draw = p_away = p_btts = p_over15 = p_over25 = p_over35 = 0.0
    for i in range(9):
        for j in range(9):
            p = _poisson(i, lam_h) * _poisson(j, lam_a)
            matrix.append((f"{i}-{j}", p))
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i > 0 and j > 0: p_btts += p
            if i + j >= 2: p_over15 += p
            if i + j >= 3: p_over25 += p
            if i + j >= 4: p_over35 += p
    total = p_home + p_draw + p_away
    aux = _auxiliary_markets(df, home, away)
    return {
        "home_win": p_home / total, "draw": p_draw / total, "away_win": p_away / total,
        "double_chance_1x": (p_home + p_draw) / total,
        "double_chance_x2": (p_draw + p_away) / total,
        "double_chance_12": (p_home + p_away) / total,
        "btts": p_btts, "over15": p_over15, "over25": p_over25, "over35": p_over35,
        "lambda_home": lam_h, "lambda_away": lam_a,
        "top_scores": sorted(matrix, key=lambda x: x[1], reverse=True)[:8],
        "home_matches": len(h), "away_matches": len(a), "auxiliary": aux,
    }


def compare_odds(probability, decimal_odds):
    probability = max(0.0, min(1.0, float(probability)))
    odds = max(1.01, float(decimal_odds))
    return {"implied_probability": 1.0 / odds, "fair_odds": (1.0 / probability) if probability > 0 else float("inf"), "expected_value": probability * odds - 1.0}


def load_historical_data():
    return pd.DataFrame()
