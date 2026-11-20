import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from predictor import prepare_api_history, predict_match, compare_odds, poisson_over

st.set_page_config(page_title="Football AI Predictor", page_icon="⚽", layout="wide")
BASE_URL = "https://v3.football.api-sports.io"
MADRID_TZ = ZoneInfo("Europe/Madrid")
CURRENT_SEASON = 2026  # API-Football identifica la temporada por el año de inicio.
COMPETITIONS = {
    "LaLiga (España)": {"id": 140, "country": "Spain"},
    "Segunda División (España)": {"id": 141, "country": "Spain"},
    "Premier League (Inglaterra)": {"id": 39, "country": "England"},
    "Bundesliga (Alemania)": {"id": 78, "country": "Germany"},
    "Serie A (Italia)": {"id": 135, "country": "Italy"},
    "Ligue 1 (Francia)": {"id": 61, "country": "France"},
    "Eredivisie (Países Bajos)": {"id": 88, "country": "Netherlands"},
}


def madrid_today():
    return datetime.now(MADRID_TZ).date()


def get_api_key():
    try:
        return st.secrets.get("API_FOOTBALL_KEY", "")
    except Exception:
        return ""


def api_get(endpoint, params, api_key):
    response = requests.get(
        f"{BASE_URL}/{endpoint}", params=params,
        headers={"x-apisports-key": api_key}, timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("errors")
    if errors:
        raise RuntimeError(f"API-Football: {errors}")
    return payload


@st.cache_data(ttl=900, show_spinner=False)
def get_fixtures(iso_day, selected_comps, api_key):
    all_items, errors = [], []
    for label in selected_comps:
        cfg = COMPETITIONS[label]
        try:
            payload = api_get("fixtures", {"date": iso_day, "league": cfg["id"], "season": CURRENT_SEASON}, api_key)
            for item in payload.get("response", []):
                item["_competition_label"] = label
                all_items.append(item)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    return all_items, errors


@st.cache_data(ttl=21600, show_spinner=False)
def get_historical_data(competition, api_key):
    """Una consulta para la competición seleccionada: últimos 120 días y página 1."""
    start_day = (madrid_today() - timedelta(days=120)).isoformat()
    end_day = madrid_today().isoformat()
    cfg = COMPETITIONS[competition]
    payload = api_get("fixtures", {
        "league": cfg["id"], "season": CURRENT_SEASON,
        "from": start_day, "to": end_day, "status": "FT", "page": 1,
    }, api_key)
    records = payload.get("response", [])
    for item in records:
        item["_competition_label"] = competition
    pagination = payload.get("paging", {})
    pages = pagination.get("total", 1) if isinstance(pagination, dict) else 1
    coverage = [f"{competition}: {len(records)} partidos finalizados" + (f" (página 1 de {pages})" if pages and pages > 1 else "")]
    problems = []
    if pages and pages > 1:
        problems.append(f"API-Football indica {pages} páginas; se ha cargado solo la primera para controlar el consumo de consultas gratuitas.")
    return records, problems, coverage


def fixture_view(item):
    fixture = item.get("fixture") or {}
    teams = item.get("teams") or {}
    home, away = teams.get("home") or {}, teams.get("away") or {}
    league = item.get("league") or {}
    goals = item.get("goals") or {}
    try:
        dt = datetime.fromisoformat(str(fixture.get("date")).replace("Z", "+00:00")).astimezone(MADRID_TZ)
        time_text = dt.strftime("%H:%M")
    except Exception:
        dt, time_text = None, "Hora pendiente"
    score = f"{goals.get('home')} - {goals.get('away')}" if goals.get("home") is not None and goals.get("away") is not None else "—"
    status = ((fixture.get("status") or {}).get("long")) or "Sin estado"
    return {
        "fixture_id": fixture.get("id"), "home_id": (home.get("id")), "away_id": (away.get("id")),
        "Hora": time_text, "Competición": item.get("_competition_label") or league.get("name", "Competición"),
        "Local": home.get("name", "Local"), "Visitante": away.get("name", "Visitante"),
        "Estado": status, "Resultado": score,
    }


def show_calendar(api_key):
    st.title("⚽ Football AI Predictor")
    st.caption("Calendario y análisis estadístico · Horario de España peninsular")
    st.subheader("Calendario")
    today = madrid_today()
    if "calendar_date" not in st.session_state:
        st.session_state.calendar_date = today
    cols = st.columns(4)
    actions = [("Hoy", today), ("Mañana", today + timedelta(days=1)),
               ("Sábado", today + timedelta(days=(5-today.weekday()) % 7)),
               ("Domingo", today + timedelta(days=(6-today.weekday()) % 7))]
    for col, (label, value) in zip(cols, actions):
        if col.button(label, use_container_width=True):
            st.session_state.calendar_date = value
            st.rerun()
    chosen_date = st.date_input("Fecha", value=st.session_state.calendar_date, key="calendar_date", format="DD/MM/YYYY")
    selected_comps = st.multiselect("Competiciones", list(COMPETITIONS), default=list(COMPETITIONS), key="calendar_competitions")
    if not selected_comps:
        st.info("Selecciona al menos una competición.")
        return
    fixtures, errors = [], []
    with st.spinner("Consultando el calendario de API-Football..."):
        all_items, errors = get_fixtures(chosen_date.isoformat(), tuple(selected_comps), api_key)
    for item in all_items:
        view = fixture_view(item)
        if view["Competición"] in selected_comps:
            fixtures.append(view)
    if errors:
        with st.expander(f"Incidencias de calendario ({len(errors)})"):
            for error in errors:
                st.warning(error)
    if not fixtures:
        st.info(f"No se han encontrado partidos para el {chosen_date:%d/%m/%Y} en las competiciones seleccionadas.")
        return
    fixtures.sort(key=lambda x: (x["Hora"], x["Competición"], x["Local"]))
    st.session_state["day_fixtures"] = fixtures
    st.success(f"{len(fixtures)} partidos encontrados.")
    st.dataframe(pd.DataFrame([{k:v for k,v in x.items() if k not in ("fixture_id", "home_id", "away_id")} for x in fixtures]), use_container_width=True, hide_index=True)
    labels = [f"{x['Hora']} · {x['Competición']} · {x['Local']} vs {x['Visitante']}" for x in fixtures]
    chosen = st.selectbox("Elige el partido que quieres analizar", labels, key="fixture_choice")
    if st.button("Analizar este partido", type="primary"):
        st.session_state.selected_fixture = fixtures[labels.index(chosen)]
        st.session_state.last_prediction = None
        st.rerun()


def render_odds(result, fixture_name):
    st.markdown("#### Comparador de cuotas Bet365")
    st.caption("Introduce manualmente la cuota decimal que aparece en Bet365.")
    markets = {
        "Victoria local": result["home_win"], "Empate": result["draw"], "Victoria visitante": result["away_win"],
        "Doble oportunidad 1X": result["double_chance_1x"], "Doble oportunidad X2": result["double_chance_x2"],
        "Doble oportunidad 12": result["double_chance_12"], "Más de 1,5 goles": result["over15"],
        "Más de 2,5 goles": result["over25"], "Más de 3,5 goles": result["over35"], "Ambos equipos marcan - Sí": result["btts"],
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
        st.success("La cuota supera la cuota justa estimada; no garantiza una apuesta ganadora.")
    else:
        st.warning("Con esta estimación, la cuota no ofrece valor esperado positivo.")
    if st.button("Registrar apuesta en el seguimiento", type="primary"):
        entry = {"Fecha registro": datetime.now(MADRID_TZ).strftime("%Y-%m-%d %H:%M"), "Partido": fixture_name,
                 "Mercado": market, "Probabilidad modelo": round(float(markets[market]), 6),
                 "Cuota Bet365": float(odd), "Cuota justa": round(float(comp["fair_odds"]), 4),
                 "Valor esperado": round(float(comp["expected_value"]), 6), "Resultado": "Pendiente"}
        st.session_state.setdefault("bet_log", []).append(entry)
        st.success("Apuesta registrada en esta sesión. Descarga el CSV para conservar el historial.")


def render_bet_tracker():
    st.divider()
    st.subheader("Seguimiento de apuestas")
    st.caption("Puedes descargar el registro e importarlo en futuras sesiones para conservarlo.")
    log = st.session_state.setdefault("bet_log", [])
    if log:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        st.download_button("Descargar historial CSV", data=df_log.to_csv(index=False).encode("utf-8-sig"),
                           file_name="football_ai_historial_apuestas.csv", mime="text/csv")
        pending = [i for i, x in enumerate(log) if x.get("Resultado") == "Pendiente"]
        if pending:
            idx = st.selectbox("Apuesta que quieres liquidar", pending,
                format_func=lambda i: f"{log[i]['Partido']} · {log[i]['Mercado']} · cuota {log[i]['Cuota Bet365']}")
            result_status = st.selectbox("Resultado", ["Ganada", "Perdida", "Anulada"])
            if st.button("Guardar resultado"):
                log[idx]["Resultado"] = result_status
                st.session_state["bet_log"] = log
                st.rerun()
        settled = [x for x in log if x.get("Resultado") in ("Ganada", "Perdida", "Anulada")]
        if settled:
            wins = sum(x["Resultado"] == "Ganada" for x in settled)
            losses = sum(x["Resultado"] == "Perdida" for x in settled)
            valid = [x for x in settled if x.get("Resultado") != "Anulada"]
            profit = sum((float(x.get("Cuota Bet365", 0))-1) if x.get("Resultado") == "Ganada" else -1 for x in valid)
            st.write(f"Liquidadas: **{len(settled)}** · Ganadas: **{wins}** · Perdidas: **{losses}** · Anuladas: **{len(settled)-wins-losses}**")
            st.write(f"Balance teórico (1 unidad por apuesta): **{profit:+.2f}** · ROI: **{(profit/len(valid) if valid else 0):+.1%}**")
    else:
        st.info("Todavía no has registrado apuestas.")
    uploaded = st.file_uploader("Importar historial CSV anterior", type=["csv"], key="bet_log_upload")
    if uploaded is not None and st.button("Importar historial"):
        try:
            imported = pd.read_csv(uploaded).fillna("")
            required = {"Partido", "Mercado", "Cuota Bet365", "Resultado"}
            if not required.issubset(imported.columns):
                st.error("El CSV no contiene las columnas necesarias.")
            else:
                st.session_state["bet_log"] = imported.to_dict("records")
                st.success(f"Se han importado {len(imported)} registros.")
                st.rerun()
        except Exception as exc:
            st.error(f"No se pudo importar el CSV: {exc}")


def main():
    st.title("Football AI Predictor · Nueva conexión")
    api_key = get_api_key()
    if not api_key:
        st.error("No encuentro API_FOOTBALL_KEY en los Secrets de Streamlit.")
        st.code('API_FOOTBALL_KEY = "TU_CLAVE_PRIVADA"', language="toml")
        st.stop()
    with st.sidebar:
        st.header("Fuente de datos")
        st.write("API-Football v3 · IDs oficiales de competiciones y equipos")
        st.caption("Plan gratuito: el consumo depende de las consultas y páginas disponibles. El histórico se limita inicialmente a una consulta por liga para controlar el cupo.")
        if st.button("Comprobar conexión API"):
            try:
                status = api_get("status", {}, api_key)
                account = status.get("response", {}).get("account", {}).get("firstname", "Cuenta conectada")
                st.success(f"Conexión correcta: {account}")
            except Exception as exc:
                st.error(f"No se pudo conectar: {exc}")
        if st.button("Actualizar datos históricos"):
            get_historical_data.clear()
            st.rerun()
    show_calendar(api_key)
    st.divider()
    st.header("Análisis y apuestas")
    selected = st.session_state.get("selected_fixture")
    if not selected:
        st.info("Selecciona un partido del calendario para ver su análisis.")
        render_bet_tracker()
        return
    with st.spinner(f"Cargando histórico reciente de {selected['Competición']}..."):
        records, history_errors, coverage = get_historical_data(selected["Competición"], api_key)
    raw = prepare_api_history(records)
    st.caption(f"Histórico de {selected['Competición']}: {len(raw)} partidos finalizados en {raw['HomeID'].nunique() if not raw.empty else 0} equipos.")
    with st.expander("Cobertura y diagnóstico del histórico"):
        for line in coverage:
            st.write("• " + line)
        for error in history_errors:
            st.warning(error)
        st.caption("Histórico inicial: temporada 2026 y últimos 120 días. Se carga solo la primera página de la competición seleccionada para limitar el consumo del plan gratuito.")
    st.info(f"Partido seleccionado: **{selected['Local']} vs {selected['Visitante']}** · {selected['Competición']}")
    league_df = raw.copy()
    try:
        result = predict_match(league_df, selected["home_id"], selected["away_id"])
    except Exception as exc:
        st.warning(f"Todavía no se puede calcular este partido con rigor: {exc}")
        st.caption("No se buscarán equivalencias por nombre ni se generarán datos ficticios. Prueba a actualizar el histórico cuando haya más partidos disponibles.")
        render_bet_tracker()
        return
    st.subheader(f"{selected['Local']} vs {selected['Visitante']}")
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
    st.write(f"Goles esperados: **{selected['Local']} {result['lambda_home']:.2f}** · **{selected['Visitante']} {result['lambda_away']:.2f}**")
    st.markdown("#### Marcadores exactos más probables")
    scores = pd.DataFrame(result["top_scores"], columns=["Marcador", "Probabilidad"])
    scores["Probabilidad"] = scores["Probabilidad"].map(lambda p: f"{p:.1%}")
    st.dataframe(scores, use_container_width=True, hide_index=True)
    st.caption(f"Muestra: {result['home_matches']} partidos del local en casa y {result['away_matches']} del visitante fuera.")
    st.warning("El modelo Poisson es básico y no está validado como sistema rentable. Las probabilidades son estimaciones, no garantías.")
    st.info("Córners, faltas, tarjetas, tiros a puerta y paradas se mantienen desactivados en esta migración inicial: requieren estadísticas de partido verificadas. Los mercados individuales de jugadores también quedan pendientes de cobertura real.")
    render_odds(result, f"{selected['Local']} vs {selected['Visitante']}")
    render_bet_tracker()


if __name__ == "__main__":
    main()
