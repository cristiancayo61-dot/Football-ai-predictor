import re
import unicodedata
from difflib import SequenceMatcher
import requests
import streamlit as st

API_ROOT = "https://api.the-odds-api.com/v4/sports"
SPORT_KEYS = {
    "LaLiga (España)": "soccer_spain_la_liga",
    "Segunda División (España)": "soccer_spain_segunda_division",
    "Premier League (Inglaterra)": "soccer_epl",
    "Bundesliga (Alemania)": "soccer_germany_bundesliga",
    "Serie A (Italia)": "soccer_italy_serie_a",
    "Ligue 1 (Francia)": "soccer_france_ligue_one",
    "Eredivisie (Países Bajos)": "soccer_netherlands_eredivisie",
}


def _api_key():
    try:
        return st.secrets.get("THE_ODDS_API_KEY", "")
    except Exception:
        return ""


def _norm(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = "".join(c for c in value if not unicodedata.combining(c)).lower()
    value = re.sub(r"\b(fc|cf|sc|ac|afc|club|de|the)\b", " ", value)
    return re.sub(r"[^a-z0-9]", "", value)


def _same_team(a, b):
    x, y = _norm(a), _norm(b)
    return bool(x and y and (x == y or x in y or y in x or SequenceMatcher(None, x, y).ratio() >= .88))


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_league(sport_key, key):
    response = requests.get(
        f"{API_ROOT}/{sport_key}/odds",
        params={"regions": "eu", "markets": "h2h,totals",
                "oddsFormat": "decimal", "dateFormat": "iso"},
        headers={"x-api-key": key}, timeout=25
    )
    if response.status_code == 401:
        raise ValueError("La API Key no es válida o no está autorizada.")
    if response.status_code == 403:
        raise ValueError("El plan no permite consultar esta competición o mercado.")
    if response.status_code == 429:
        raise ValueError("Se ha alcanzado el límite de consultas. Prueba más tarde.")
    response.raise_for_status()
    return response.json(), response.headers.get("x-requests-remaining")


def get_match_odds(competition, home, away):
    key = _api_key()
    if not key:
        return {"event": None, "error": "No encuentro THE_ODDS_API_KEY en los Secrets de Streamlit."}
    sport = SPORT_KEYS.get(competition)
    if not sport:
        return {"event": None, "error": "Competición sin identificador de cuotas configurado."}
    try:
        events, remaining = _fetch_league(sport, key)
    except Exception as exc:
        return {"event": None, "error": str(exc)}
    for event in events:
        if _same_team(event.get("home_team"), home) and _same_team(event.get("away_team"), away):
            return {"event": event, "error": None, "remaining": remaining}
    return {"event": None, "error": None, "remaining": remaining}


def odds_for_market(event, market_label, home, away):
    if not event:
        return []
    label = market_label.lower()
    if label == "victoria local":
        market_key, target = "h2h", home
    elif label == "empate":
        market_key, target = "h2h", "draw"
    elif label == "victoria visitante":
        market_key, target = "h2h", away
    elif label.startswith("más de ") and " goles" in label:
        market_key, target = "totals", "over"
        try:
            line = float(label.split("más de ", 1)[1].split(" ", 1)[0].replace(",", "."))
        except Exception:
            return []
    else:
        return []
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes", []):
                name = str(outcome.get("name", "")).lower()
                if market_key == "h2h":
                    match = name in ("draw", "tie", "empate") if target == "draw" else _same_team(name, target)
                else:
                    match = name == "over" and abs(float(outcome.get("point", -999)) - line) < .001
                if match:
                    try:
                        price = float(outcome["price"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if price > 1:
                        rows.append({"Casa": book.get("title", book.get("key", "Desconocida")),
                                     "Cuota": round(price, 3),
                                     "Actualizada": market.get("last_update", "No indicada")})
    return sorted(rows, key=lambda x: x["Cuota"], reverse=True)
