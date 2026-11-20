import streamlit as st
import pandas as pd
import requests
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from predictor import load_historical_data, predict_match, compare_odds, match_team_name

st.set_page_config(page_title="Football AI Predictor", page_icon="⚽", layout="wide")

API_URL = "https://api.api-ninjas.com/v1/soccermatches"
MADRID_TZ = ZoneInfo("Europe/Madrid")

# API Ninjas codes for the calendar; Football-Data codes for historical results.
COMPETITIONS = {
    "LaLiga (España)": {"api": "laliga", "fd": "SP1"},
    "Segunda División (España)": {"api": "laliga2", "fd": "SP2"},
    "Premier League (Inglaterra)": {"api": "epl", "fd": "E0"},
    "Bundesliga (Alemania)": {"api": "bundesliga", "fd": "D1"},
    "Serie A (Italia)": {"api": "seriea", "fd": "I1"},
    "Ligue 1 (Francia)": {"api": "ligue1", "fd": "F1"},
    "Eredivisie (Países Bajos)": {"api": "eredivisie", "fd": "N1"},
    "Champions League": {"api": "ucl", "fd": None},
    "Europa League": {"api": "uel", "fd": None},
}
SEASONS = ["2223", "2324", "2425", "2526", "2627"]


def madrid_today():
    return datetime.now(MADRID_TZ).date()


def get_api_key():
    try:
        return st.secrets.get("API_NINJAS_KEY", "")
    except Exception:
        return ""


def get_team_name(value, fallback):
    if isinstance(value, dict):
        return str(value.get("name") or fallback)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def parse_api_datetime(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(MADRID_TZ)
    except (ValueError, TypeError):
        return None


def madrid_time(value):
    dt = parse_api_datetime(value)
    return dt.strftime("%H:%M") if dt else "Hora pendiente"


@st.cache_data(ttl=900, show_spinner=False)
def get_fixtures(league_code, iso_day, api_key):
    response = requests.get(
        API_URL,
        params={"league": league_code, "date": iso_day},
        headers={"X-Api-Key": api_key},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("La API devolvió un formato inesperado.")
    return payload


@st.cache_data(ttl=21600, show_spinner=False)
def get_all_history():
    """Descarga CSV oficiales de football-data.co.uk; 6 horas de caché."""
    frames, problems = [], []
    for label, cfg in COMPETITIONS.items():
        code = cfg["fd"]
        if not code:
            continue
        for season in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                response = requests.get(
                    url, timeout=20,
                    headers={"User-Agent": "Football-AI-Predictor-personal-use/1.0"}
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                from io import BytesIO
                raw = pd.read_csv(BytesIO(response.content), encoding="latin-1")
                raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
                required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
                if not required.issubset(raw.columns):
                    problems.append(f"{label} {season}: faltan columnas básicas.")
                    continue
                raw["League"] = label
                raw["Season"] = season
                frames.append(raw)
            except Exception as exc:
                problems.append(f"{label} {season}: {type(exc).__name__}")
    if not frames:
        return pd.DataFrame(), problems
    return pd.concat(frames, ignore_index=True, sort=False), problems


def show_calendar():
    st.title("⚽ Football AI Predictor")
    st.caption("Calendario de partidos y análisis estadístico · Horario de España peninsular")
    st.subheader("Calendario")

    today = madrid_today()
    if "calendar_date" not in st.session_state:
        st.session_state.calendar_date = today

    cols = st.columns(4)
    actions = [
        ("Hoy", today),
        ("Mañana", today + timedelta(days=1)),
        ("Sábado", today + timedelta(days=(5 - today.weekday()) % 7)),
        ("Domingo", today + timedelta(days=(6 - today.weekday()) % 7)),
    ]
    for col, (label, value) in zip(cols, actions):
        if col.button(label, use_container_width=True):
            st.session_state.calendar_date = value
            st.rerun()

    chosen_date = st.date_input(
        "Fecha", value=st.session_state.calendar_date,
        key="calendar_date", format="DD/MM/YYYY"
    )
    selected_competitions = st.multiselect(
        "Competiciones", list(COMPETITIONS.keys()),
        default=list(COMPETITIONS.keys()), key="calendar_competitions"
    )
    api_key = get_api_key()
    if not api_key:
        st.error("No encuentro API_NINJAS_KEY en los Secrets de Streamlit.")
        return

    fixtures, errors = [], []
    with st.spinner("Consultando el calendario..."):
        for comp in selected_competitions:
            try:
                payload = get_fixtures(COMPETITIONS[comp]["api"], chosen_date.isoformat(), api_key)
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    start = item.get("start_time") or item.get("date")
                    home = get_team_name(item.get("home_team"), "Local")
                    away = get_team_name(item.get("away_team"), "Visitante")
                    dt = parse_api_datetime(start)
                    hs, aws = item.get("home_score"), item.get("away_score")
                    score = f"{hs} - {aws}" if hs is not None and aws is not None else "—"
                    fixtures.append({
                        "Hora": dt.strftime("%H:%M") if dt else "Pendiente",
                        "Competición": comp,
                        "Local": home,
                        "Visitante": away,
                        "Estado": item.get("status", "Sin estado"),
                        "Resultado": score,
                    })
            except Exception as exc:
                errors.append(f"{comp}: {exc}")

    if errors:
        with st.expander(f"Incidencias ({len(errors)})"):
            for e in errors:
                st.warning(e)

    if fixtures:
        fixtures.sort(key=lambda x: (x["Hora"], x["Competición"], x["Local"]))
        st.success(f"{len(fixtures)} partidos encontrados.")
        st.dataframe(pd.DataFrame(fixtures), use_container_width=True, hide_index=True)
        labels = [
            f"{x['Hora']} · {x['Competición']} · {x['Local']} vs {x['Visitante']}"
            for x in fixtures
        ]
        chosen = st.selectbox("Elige el partido que quieres analizar", labels, key="fixture_choice")
        if st.button("Analizar este partido", type="primary"):
            st.session_state.selected_fixture = fixtures[labels.index(chosen)]
            st.session_state.last_prediction = None
            st.rerun()
    else:
        st.info(f"No se han encontrado partidos para el {chosen_date:%d/%m/%Y} en las competiciones seleccionadas.")
    st.caption("El calendario depende de la cobertura y los límites de API Ninjas.")


def render_odds(result):
    st.markdown("#### Comparador de cuotas Bet365")
    st.caption("Introduce manualmente la cuota decimal que aparece en Bet365.")
    markets = {
        "Victoria local": result["home_win"],
        "Empate": result["draw"],
        "Victoria visitante": result["away_win"],
        "Doble oportunidad 1X": result["double_chance_1x"],
        "Doble oportunidad X2": result["double_chance_x2"],
        "Doble oportunidad 12": result["double_chance_12"],
        "Más de 1,5 goles": result["over15"],
        "Más de 2,5 goles": result["over25"],
        "Más de 3,5 goles": result["over35"],
        "Ambos equipos marcan - Sí": result["btts"],
    }
    market = st.selectbox("Mercado", list(markets.keys()))
    odd = st.number_input("Cuota Bet365", min_value=1.01, max_value=1000.0, value=2.00, step=0.05)
    comp = compare_odds(markets[market], odd)
    a, b, c = st.columns(3)
    a.metric("Probabilidad estimada", f"{markets[market]:.1%}")
    b.metric("Probabilidad implícita", f"{comp['implied_probability']:.1%}")
    c.metric("Valor esperado estimado", f"{comp['expected_value']:+.1%}")
    st.write(f"Cuota justa estimada: **{comp['fair_odds']:.2f}**")
    if comp["expected_value"] > 0:
        st.success("La cuota supera la cuota justa estimada. No garantiza que la apuesta sea ganadora.")
    else:
        st.warning("Con esta estimación, la cuota no ofrece valor esperado positivo.")


def main():
    show_calendar()
    st.divider()
    st.header("Análisis y apuestas")

    with st.sidebar:
        st.header("Datos históricos")
        st.caption("Los resultados se descargan automáticamente de Football-Data.co.uk. No uses datos sintéticos para apostar.")
        if st.button("Actualizar histórico ahora"):
            get_all_history.clear()
            st.rerun()

    selected = st.session_state.get("selected_fixture")
    raw, download_errors = get_all_history()
    if raw.empty:
        st.error("No se pudo cargar el histórico. Comprueba la conexión o inténtalo más tarde.")
        st.stop()

    if selected:
        st.info(f"Partido seleccionado: **{selected['Local']} vs {selected['Visitante']}** · {selected['Competición']}")
        fd_comp = COMPETITIONS[selected["Competición"]]["fd"]
        if not fd_comp:
            st.warning("El calendario muestra este torneo, pero todavía no hemos incorporado su histórico. Champions y Europa League se añadirán en una siguiente fase.")
            st.stop()
        league_df = raw[raw["League"] == selected["Competición"]].copy()
        home = match_team_name(selected["Local"], league_df)
        away = match_team_name(selected["Visitante"], league_df)
        if not home or not away:
            st.warning("No he encontrado una coincidencia suficientemente segura para los nombres de los dos equipos en el histórico. No voy a inventar una equivalencia.")
            st.write("Equipos disponibles en esta competición:")
            teams = sorted(set(league_df["HomeTeam"]).union(set(league_df["AwayTeam"])))
            st.selectbox("Equipo local", teams, key="manual_home")
            st.selectbox("Equipo visitante", teams, key="manual_away")
            if st.button("Analizar equipos seleccionados"):
                home = st.session_state.manual_home
                away = st.session_state.manual_away
            else:
                st.stop()
        if home == away:
            st.error("Los equipos local y visitante no pueden ser el mismo.")
            st.stop()
        result = predict_match(league_df, home, away)
        st.session_state.last_prediction = {"home": home, "away": away, "result": result}
    else:
        st.info("Selecciona primero un partido del calendario para ver su análisis automático.")
        st.caption(f"Histórico cargado: {len(raw):,} resultados en {raw['League'].nunique()} competiciones.")
        with st.expander("Estado de las fuentes de datos"):
            for e in download_errors[:30]:
                st.write(e)
        st.stop()

    saved = st.session_state.get("last_prediction")
    if not saved:
        st.stop()
    result = saved["result"]
    st.subheader(f"{saved['home']} vs {saved['away']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Victoria local", f"{result['home_win']:.1%}")
    c2.metric("Empate", f"{result['draw']:.1%}")
    c3.metric("Victoria visitante", f"{result['away_win']:.1%}")

    st.markdown("#### Mercados de goles")
    a, b, c, d = st.columns(4)
    a.metric("Más de 1,5", f"{result['over15']:.1%}")
    b.metric("Más de 2,5", f"{result['over25']:.1%}")
    c.metric("Más de 3,5", f"{result['over35']:.1%}")
    d.metric("Ambos marcan", f"{result['btts']:.1%}")

    st.markdown("#### Doble oportunidad")
    x, y, z = st.columns(3)
    x.metric("1X", f"{result['double_chance_1x']:.1%}")
    y.metric("X2", f"{result['double_chance_x2']:.1%}")
    z.metric("12", f"{result['double_chance_12']:.1%}")

    st.write(f"Goles esperados: **{saved['home']} {result['lambda_home']:.2f}** · **{saved['away']} {result['lambda_away']:.2f}**")
    st.markdown("#### Marcadores exactos más probables")
    scores = pd.DataFrame(result["top_scores"], columns=["Marcador", "Probabilidad"])
    scores["Probabilidad"] = scores["Probabilidad"].map(lambda p: f"{p:.1%}")
    st.dataframe(scores, use_container_width=True, hide_index=True)

    if result.get("home_matches") is not None:
        st.caption(f"Muestra utilizada: {result['home_matches']} partidos del local en casa y {result['away_matches']} del visitante fuera.")
    render_odds(result)
    st.warning("Modelo Poisson básico, no validado como sistema rentable. No incorpora alineaciones, lesiones, xG ni cuotas en tiempo real. Las probabilidades son estimaciones, no garantías.")

    with st.expander("Sobre el histórico y sus limitaciones"):
        st.write("Se consultan temporadas 2022/23 a 2026/27 cuando los archivos están disponibles. Los CSV pueden tener estadísticas de córners/tarjetas incompletas y los nombres de clubes pueden variar.")
        st.write("Fuente: Football-Data.co.uk. Datos destinados a uso personal; consulta sus condiciones antes de redistribuirlos o utilizarlos comercialmente.")
        for e in download_errors[:30]:
            st.write(e)


if __name__ == "__main__":
    main()
