import io
import json
import re
import unicodedata
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

DATA_ZIP = "https://github.com/oladeeayo/open-matchday/archive/refs/heads/master.zip"
SEASON = "2627"

def _norm(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(c for c in value if not unicodedata.combining(c)).lower()
    value = re.sub(r"\b(fc|cf|sc|ac|afc|club|de|the)\b", " ", value)
    return re.sub(r"[^a-z0-9]", "", value)

@st.cache_data(ttl=21600, show_spinner=False)
def load_official_results():
    """Carga resultados de la temporada actual desde el ZIP público de open-matchday."""
    response = requests.get(DATA_ZIP, timeout=90, headers={"User-Agent": "Football-AI-Predictor/1.0"})
    response.raise_for_status()
    frames = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            if not name.endswith(f"/results_{SEASON}.csv") or "/data/" not in name:
                continue
            try:
                df = pd.read_csv(archive.open(name))
                if not {"date", "home", "away", "ft_home", "ft_away"}.issubset(df.columns):
                    continue
                folder = name.split("/data/", 1)[1].split("/", 1)[0]
                df["competition"] = _competition_name(folder)
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
                for col in df.columns:
                    if col.startswith(("home_", "away_", "ft_")):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result.dropna(subset=["date", "home", "away", "ft_home", "ft_away"])
    result = result.drop_duplicates(subset=["date", "home", "away"])
    return result.sort_values("date", ascending=False).reset_index(drop=True)

def _competition_name(folder):
    parts = folder.split("_", 2)
    slug = parts[-1] if parts else folder
    return slug.replace("-", " ").replace("_", " ").title()



def _team_keys(team):
    target = _norm(team)
    keys = {target} if target else set()
    alias_path = Path(__file__).resolve().parent / "Data" / "equipos_equivalencias.json"
    try:
        payload = json.loads(alias_path.read_text(encoding="utf-8"))
        matched_groups = set()
        for mapping in payload.get("leagues", {}).values():
            for canonical, variants in mapping.items():
                group = {_norm(canonical)}
                group.update(_norm(v) for v in variants if v)
                if target in group:
                    matched_groups.add(frozenset(group))
        # Solo ampliar alias cuando la coincidencia identifica un único grupo.
        if len(matched_groups) == 1:
            keys.update(next(iter(matched_groups)))
    except Exception:
        pass
    return {x for x in keys if x}

def _team_match_rows(df, team):
    keys = _team_keys(team)
    if not keys or df.empty:
        return pd.DataFrame()
    home_norm = df["home"].astype(str).map(_norm)
    away_norm = df["away"].astype(str).map(_norm)
    exact = df[home_norm.isin(keys) | away_norm.isin(keys)].copy()
    if not exact.empty:
        return exact
    # Fallback prudente: solo se acepta una coincidencia claramente única.
    names = pd.unique(pd.concat([df["home"], df["away"]]).dropna().astype(str))
    scores = sorted(
        ((max(SequenceMatcher(None, key, _norm(name)).ratio() for key in keys), name)
         for name in names),
        reverse=True
    )
    if not scores or scores[0][0] < .94:
        return pd.DataFrame()
    if len(scores) > 1 and scores[0][0] - scores[1][0] < .035:
        return pd.DataFrame()
    chosen = _norm(scores[0][1])
    return df[(home_norm == chosen) | (away_norm == chosen)].copy()

def last_five(team, df=None):
    df = load_official_results() if df is None else df
    rows = _team_match_rows(df, team)
    if rows.empty:
        return pd.DataFrame()
    rows = rows.sort_values("date", ascending=False).head(5).copy()
    output = []
    for _, r in rows.iterrows():
        is_home = _norm(r["home"]) in _team_keys(team)
        gf = r["ft_home"] if is_home else r["ft_away"]
        ga = r["ft_away"] if is_home else r["ft_home"]
        result = "G" if gf > ga else ("E" if gf == ga else "P")
        item = {
            "Fecha": r["date"].strftime("%d/%m/%Y"),
            "Competición": r.get("competition", "No disponible"),
            "Local": r["home"], "Visitante": r["away"],
            "Casa/Fuera": "Casa" if is_home else "Fuera",
            "Marcador": f'{int(r["ft_home"])} - {int(r["ft_away"])}',
            "GF": int(gf), "GC": int(ga), "Resultado": result,
        }
        stat_pairs = [
            ("Córners", "home_corners", "away_corners"),
            ("Tiros a puerta", "home_target", "away_target"),
            ("Faltas", "home_fouls", "away_fouls"),
            ("Amarillas", "home_yellow", "away_yellow"),
            ("Rojas", "home_red", "away_red"),
        ]
        for label, hc, ac in stat_pairs:
            value = r.get(hc) if is_home else r.get(ac)
            item[label] = value if pd.notna(value) else None
        output.append(item)
    return pd.DataFrame(output)

def form_summary(matches):
    if matches is None or matches.empty:
        return {"played": 0, "points": 0, "goals_for": None, "goals_against": None, "wins": 0, "draws": 0, "losses": 0}
    return {
        "played": len(matches),
        "points": int((matches["Resultado"] == "G").sum() * 3 + (matches["Resultado"] == "E").sum()),
        "goals_for": float(matches["GF"].mean()),
        "goals_against": float(matches["GC"].mean()),
        "wins": int((matches["Resultado"] == "G").sum()),
        "draws": int((matches["Resultado"] == "E").sum()),
        "losses": int((matches["Resultado"] == "P").sum()),
    }

def stat_average(matches, column):
    if matches is None or matches.empty or column not in matches:
        return None, 0
    values = pd.to_numeric(matches[column], errors="coerce").dropna()
    if values.empty:
        return None, 0
    return float(values.mean()), int(len(values))
