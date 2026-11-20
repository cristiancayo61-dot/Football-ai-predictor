import streamlit as st
import pandas as pd
import requests
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from predictor import prepare_data, predict_match, demo_data, compare_odds

st.set_page_config(page_title="Football AI Predictor", page_icon="⚽", layout="wide")

API_URL = "https://api.api-ninjas.com/v1/soccermatches"
MADRID_TZ = ZoneInfo("Europe/Madrid")

COMPETITIONS = {
    "LaLiga (España)": "laliga",
    "Segunda División (España)": "laliga2",
    "Premier League (Inglaterra)": "epl",
    "Bundesliga (Alemania)": "bundesliga",
    "Serie A (Italia)": "seriea",
    "Ligue 1 (Francia)": "ligue1",
    "Champions League": "ucl",
    "Europa League": "uel",
}


def madrid_today():
    return datetime.now(MADRID_TZ).date()


def get_api_key():
    try:
        return st.secrets.get("API_NINJAS_KEY", "")
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def get_fixtures(league_code, match_date, api_key):
    """Obtiene partidos de una competición y fecha. El resultado queda cacheado 15 min."""
    response = requests.get(
        API_URL,
        params={"league": league_code, "date": match_date},
        headers={"X-Api-Key": api_key},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("La API devolvió un formato de datos inesperado.")
    return payload


def team_name(value, fallback):
    """La API puede devolver el equipo como texto o como objeto con name."""
    if isinstance(value, dict):
        return str(value.get("name") or fallback)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def madrid_time(iso_time):
    if not iso_time:
        return "Hora pendiente"
    try:
        parsed = datetime.fromisoformat(str(iso_time).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(MADRID_TZ).strftime("%H:%M")
    except (ValueError, TypeError):
        return "Hora no disponible"


def fixture_date(iso_time, fallback):
    if not iso_time:
        return fallback.strftime("%d/%m/%Y")
    try:
        parsed = datetime.fromisoformat(str(iso_time).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(MADRID_TZ).strftime("%d/%m/%Y")
    except (ValueError, TypeError):
        return fallback.strftime("%d/%m/%Y")


def spanish_status(status):
    translations = {
        "scheduled": "Programado", "in_progress": "En juego",
        "final": "Finalizado", "postponed": "Aplazado",
        "canceled": "Cancelado", "cancelled": "Cancelado",
        "suspended": "Suspendido", "delayed": "Retrasado",
    }
    return translations.get(str(status).lower(), status or "Sin estado")


@st.cache_data(ttl=900, show_spinner=False)
def load_day(iso_day, selected_names, api_key):
    """Consulta las competiciones seleccionadas para una fecha."""
    matches, errors = [], []
    day = date.fromisoformat(iso_day)
    for competition_name in selected_names:
        league_code = COMPETITIONS[competition_name]
        try:
            payload = get_fixtures(league_code, iso_day, api_key)
            for match in payload:
                if not isinstance(match, dict):
                    continue
                start_time = match.get("start_time") or match.get("date")
                home = team_name(match.get("home_team"), "Local")
                away = team_name(match.get("away_team"), "Visitante")
                home_score, away_score = match.get("home_score"), match.get("away_score")
                score = (f"{home_score} - {away_score}"
                         if home_score is not None and away_score is not None else "—")
                matches.append({
                    "Fecha": fixture_date(start_time, day),
                    "Hora": madrid_time(start_time),
                    "Competición": competition_name,
                    "Local": home,
                    "Visitante": away,
                    "Estado": spanish_status(match.get("status")),
                    "Resultado": score,
                    "ID": str(match.get("game_id") or ""),
                })
        except requests.HTTPError as exc:
            detail = exc.response.text[:180] if exc.response is not None else str(exc)
            errors.append(f"{competition_name}: {detail}")
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"{competition_name}: {exc}")
    matches.sort(key=lambda x: (x["Hora"], x["Competición"], x["Local"]))
    return matches, errors


def show_calendar():
    st.title("⚽ Football AI Predictor")
    st.caption("Calendario y análisis de partidos · Horario peninsular español")
    st.subheader("Calendario de partidos")

    today = madrid_today()
    if "calendar_date" not in st.session_state:
        st.session_state.calendar_date = today

    quick1, quick2, quick3, quick4 = st.columns(4)
    if quick1.button("Hoy", use_container_width=True):
        st.session_state.calendar_date = today
        st.rerun()
    if quick2.button("Mañana", use_container_width=True):
        st.session_state.calendar_date = today + timedelta(days=1)
        st.rerun()
    if quick3.button("Sábado", use_container_width=True):
        st.session_state.calendar_date = today + timedelta(days=(5 - today.weekday()) % 7)
        st.rerun()
    if quick4.button("Domingo", use_container_width=True):
        st.session_state.calendar_date = today + timedelta(days=(6 - today.weekday()) % 7)
        st.rerun()

    chosen_date = st.date_input(
        "Fecha de los partidos",
        value=st.session_state.calendar_date,
        key="calendar_date",
        format="DD/MM/YYYY",
    )

    all_competitions = list(COMPETITIONS.keys())
    chosen_competitions = st.multiselect(
        "Competiciones",
        all_competitions,
        default=all_competitions,
        key="calendar_competitions",
    )

    api_key = get_api_key()
    if not api_key:
        st.error("No encuentro API_NINJAS_KEY en los Secrets de Streamlit. Revisa Manage app → Settings → Secrets.")
        return []

    if not chosen_competitions:
        st.info("Selecciona al menos una competición.")
        return []

    with st.spinner(f"Buscando partidos del {chosen_date:%d/%m/%Y}..."):
        fixtures, errors = load_day(chosen_date.isoformat(), tuple(chosen_competitions), api_key)

    if errors:
        with st.expander(f"Incidencias de consulta ({len(errors)})"):
            for err in errors:
                st.warning(err)

    if fixtures:
        st.success(f"{len(fixtures)} partido(s) encontrado(s) para el {chosen_date:%d/%m/%Y}.")
        visible = pd.DataFrame(fixtures)[["Hora", "Competición", "Local", "Visitante", "Estado", "Resultado"]]
        st.dataframe(visible, use_container_width=True, hide_index=True)
        labels = [
            f"{x['Hora']} · {x['Competición']} · {x['Local']} vs {x['Visitante']}"
            for x in fixtures
        ]
        selected_label = st.selectbox("Selecciona un partido para analizarlo", labels, key="selected_fixture_label")
        if st.button("Analizar partido seleccionado", type="primary"):
            idx = labels.index(selected_label)
            st.session_state.selected_fixture = fixtures[idx]
            st.session_state.analysis_home = fixtures[idx]["Local"]
            st.session_state.analysis_away = fixtures[idx]["Visitante"]
            st.success("Partido seleccionado. Baja a «Análisis y apuestas» para continuar.")
    else:
        st.info(
            f"No hay partidos disponibles para el {chosen_date:%d/%m/%Y} "
            "en las competiciones seleccionadas, o la API no ha devuelto encuentros."
        )
    st.caption("El calendario se consulta automáticamente al abrir la aplicación y al cambiar la fecha. "
               "Los resultados dependen de la cobertura y los límites de API Ninjas.")
    return fixtures


def odds_panel(result):
    st.markdown("#### Comparador manual de cuotas Bet365")
    st.caption("Introduce la cuota decimal que ves en Bet365. No se consulta ni se extrae automáticamente de Bet365.")
    market_options = {
        "Victoria local": result["home_win"],
        "Empate": result["draw"],
        "Victoria visitante": result["away_win"],
        "Local o empate (1X)": result["double_chance_1x"],
        "Empate o visitante (X2)": result["double_chance_x2"],
        "Local o visitante (12)": result["double_chance_12"],
        "Más de 1,5 goles": result["over15"],
        "Más de 2,5 goles": result["over25"],
        "Más de 3,5 goles": result["over35"],
        "Ambos equipos marcan: Sí": result["btts"],
    }
    market = st.selectbox("Mercado que quieres comparar", list(market_options.keys()))
    odd = st.number_input("Cuota decimal de Bet365", min_value=1.01, max_value=1000.0,
                          value=2.00, step=0.05, format="%.2f")
    probability = market_options[market]
    comparison = compare_odds(probability, odd)
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad del modelo", f"{probability:.1%}")
    c2.metric("Probabilidad implícita", f"{comparison['implied_probability']:.1%}")
    c3.metric("Valor esperado estimado", f"{comparison['expected_value']:+.1%}")
    if comparison["expected_value"] > 0:
        st.success("La cuota supera la cuota justa estimada por el modelo. No significa que la apuesta vaya a ganar.")
    else:
        st.warning("Con esta estimación, la cuota no ofrece valor esperado positivo.")
    st.caption(f"Cuota justa estimada: {comparison['fair_odds']:.2f} · "
               "El cálculo depende de la calidad y cantidad de los datos históricos.")


def main():
    fixtures = show_calendar()
    st.divider()

    st.header("Análisis y apuestas")
    with st.sidebar:
        st.header("Datos históricos")
        mode = st.radio("Fuente de datos", ["Datos de demostración", "Subir CSV de resultados"])
        st.caption("Los datos de demostración son sintéticos y no sirven para tomar decisiones reales.")

    raw = None
    if mode == "Subir CSV de resultados":
        uploaded = st.sidebar.file_uploader("Sube un CSV de resultados históricos", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"No se pudo leer el CSV: {exc}")
    else:
        raw = demo_data()

    if raw is None:
        st.info("Carga un CSV histórico en la barra lateral para calcular probabilidades.")
        st.stop()

    df, errors = prepare_data(raw)
    if errors:
        st.error(errors)
        st.stop()

    st.success(f"Histórico cargado: {len(df)} partidos · {df['HomeTeam'].nunique()} equipos")
    tab_analysis, tab_data, tab_help = st.tabs(["📈 Análisis y apuestas", "📁 Datos", "ℹ️ Cómo funciona"])

    with tab_analysis:
        historical_teams = sorted(set(df.HomeTeam).union(set(df.AwayTeam)))
        selected = st.session_state.get("selected_fixture")
        if selected:
            st.info(f"Partido del calendario: {selected['Local']} vs {selected['Visitante']} · {selected['Competición']}")
        default_home = st.session_state.get("analysis_home", historical_teams[0])
        if default_home not in historical_teams:
            default_home = historical_teams[0]
        home = st.selectbox("Equipo local (según tu histórico)", historical_teams,
                            index=historical_teams.index(default_home), key="analysis_home_select")
        away_options = [x for x in historical_teams if x != home]
        default_away = st.session_state.get("analysis_away", away_options[0])
        if default_away not in away_options:
            default_away = away_options[0]
        away = st.selectbox("Equipo visitante (según tu histórico)", away_options,
                            index=away_options.index(default_away), key="analysis_away_select")

        if selected and (selected["Local"] not in historical_teams or selected["Visitante"] not in historical_teams):
            st.warning("Uno o ambos equipos del calendario no coinciden exactamente con los nombres de tu CSV. "
                       "Selecciona manualmente los equipos equivalentes arriba. El modelo no puede analizar "
                       "automáticamente un equipo sin historial.")

        if st.button("Calcular análisis", type="primary"):
            result = predict_match(df, home, away)
            st.session_state.last_prediction = {"home": home, "away": away, "result": result}

        saved = st.session_state.get("last_prediction")
        if saved:
            result = saved["result"]
            st.subheader(f"{saved['home']} vs {saved['away']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Victoria local", f"{result['home_win']:.1%}")
            c2.metric("Empate", f"{result['draw']:.1%}")
            c3.metric("Victoria visitante", f"{result['away_win']:.1%}")
            st.markdown("#### Mercados principales")
            a, b, c, d = st.columns(4)
            a.metric("Más de 1,5 goles", f"{result['over15']:.1%}")
            b.metric("Más de 2,5 goles", f"{result['over25']:.1%}")
            c.metric("Más de 3,5 goles", f"{result['over35']:.1%}")
            d.metric("Ambos marcan", f"{result['btts']:.1%}")
            st.markdown("#### Doble oportunidad")
            x, y, z = st.columns(3)
            x.metric("1X", f"{result['double_chance_1x']:.1%}")
            y.metric("X2", f"{result['double_chance_x2']:.1%}")
            z.metric("12", f"{result['double_chance_12']:.1%}")
            st.write(f"Goles esperados: **{saved['home']} {result['lambda_home']:.2f}** · "
                     f"**{saved['away']} {result['lambda_away']:.2f}**")
            st.markdown("#### Marcadores más probables")
            score_df = pd.DataFrame(result["top_scores"], columns=["Marcador", "Probabilidad"])
            score_df["Probabilidad"] = score_df["Probabilidad"].map(lambda p: f"{p:.1%}")
            st.dataframe(score_df, use_container_width=True, hide_index=True)
            if "corners_home" in result:
                st.markdown("#### Promedios descriptivos de córners y tarjetas")
                q1, q2 = st.columns(2)
                q1.metric("Córners estimados", f"{result['corners_home'] + result['corners_away']:.1f}")
                q2.metric("Amarillas estimadas", f"{result['cards_home'] + result['cards_away']:.1f}")
            odds_panel(result)
            st.warning("Estas son estimaciones, no garantías. El modelo actual es una referencia estadística básica; "
                       "valídalo con temporadas históricas antes de utilizarlo para apostar.")

    with tab_data:
        st.subheader("Datos históricos")
        st.write("Columnas mínimas: HomeTeam, AwayTeam, FTHG y FTAG. Recomendadas: Date, HC, AC, HY y AY.")
        st.download_button("Descargar plantilla CSV", df.head(0).to_csv(index=False).encode("utf-8"),
                           "plantilla_partidos.csv", "text/csv")
        st.link_button("Descargar resultados históricos en football-data.co.uk",
                       "https://www.football-data.co.uk/")
        st.dataframe(df.tail(20), use_container_width=True, hide_index=True)

    with tab_help:
        st.markdown("""
        **Qué incluye esta versión**
        - Calendario como pantalla inicial, con consulta automática de la fecha actual.
        - Accesos rápidos a hoy, mañana, sábado y domingo.
        - Filtros de competiciones y selección de partido.
        - Probabilidades Poisson 1X2, doble oportunidad y mercados básicos de goles.
        - Marcadores más probables y estadísticas descriptivas cuando el CSV incluye córners y tarjetas.
        - Comparación manual de cuotas de Bet365 mediante probabilidad implícita, cuota justa y valor esperado.

        **Limitaciones importantes**
        - Bet365 no ofrece una API pública oficial para esta integración; por eso las cuotas se introducen manualmente.
        - El calendario depende de API Ninjas, su cobertura y el límite de tu plan.
        - El modelo no incorpora actualmente alineaciones, lesiones, xG, calendario de fatiga ni cuotas de mercado.
        - Las probabilidades no son una garantía ni una recomendación automática de apuesta.
        - Los datos de demostración son ficticios. Para análisis real, sube resultados históricos reales y suficientes.
        """)


if __name__ == "__main__":
    main()
