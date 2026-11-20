import streamlit as st
import pandas as pd
import requests
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from predictor import prepare_data, predict_match, demo_data

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


@st.cache_data(ttl=1800, show_spinner=False)
def get_fixtures(league_code, match_date, api_key):
    """Consulta los partidos de una competición para un día concreto."""
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


def get_api_key():
    try:
        return st.secrets.get("API_NINJAS_KEY", "")
    except Exception:
        return ""


def madrid_time(iso_time):
    if not iso_time:
        return "Hora pendiente"
    try:
        parsed = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        return parsed.astimezone(MADRID_TZ).strftime("%H:%M")
    except (ValueError, TypeError):
        return "Hora no disponible"


def spanish_status(status):
    return {
        "scheduled": "Programado",
        "in_progress": "En juego",
        "final": "Finalizado",
        "postponed": "Aplazado",
        "canceled": "Cancelado",
        "suspended": "Suspendido",
        "delayed": "Retrasado",
    }.get(status, status or "Sin estado")


st.set_page_config(page_title="Football AI Predictor FREE", page_icon="⚽", layout="wide")
st.title("⚽ Football AI Predictor")
st.caption("Versión gratuita · Prototipo estadístico · No garantiza resultados")

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Fuente de datos", ["Datos de demostración", "Subir CSV de resultados"])
    st.markdown("---")
    st.caption("Los datos de demostración son sintéticos. Para predicciones reales, carga resultados históricos.")

if mode == "Subir CSV de resultados":
    uploaded = st.file_uploader("Sube un CSV de resultados históricos", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        st.info("Sube un CSV para comenzar. Puedes descargar la plantilla desde la sección «Datos».")
        raw = None
else:
    raw = demo_data()

if raw is not None:
    df, errors = prepare_data(raw)
    if errors:
        st.error(errors)
        st.stop()

    # Si se ha elegido un partido en el calendario, preparar los selectores
    # antes de que Streamlit cree los controles de la pestaña Predicción.
    teams = sorted(set(df.HomeTeam).union(set(df.AwayTeam)))
    pending = st.session_state.pop("pending_fixture", None)
    if pending:
        pending_home, pending_away = pending
        if pending_home in teams and pending_away in teams and pending_home != pending_away:
            st.session_state["pred_home"] = pending_home
            st.session_state["pred_away"] = pending_away
            st.session_state["calendar_notice"] = (
                f"Partido seleccionado: {pending_home} vs {pending_away}. "
                "Entra en la pestaña Predicción para calcularlo."
            )
        else:
            st.session_state["calendar_notice"] = (
                "Los equipos de ese encuentro no están en tus resultados históricos. "
                "Puedes ver el partido en el calendario, pero el modelo actual no tiene datos "
                "suficientes para calcular esa predicción."
            )

    st.success(f"Datos cargados: {len(df)} partidos · {df['HomeTeam'].nunique()} equipos")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen", "🔮 Predicción", "📁 Datos", "ℹ️ Cómo funciona", "⚽ Calendario"
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos", len(df))
        c2.metric("Equipos", df["HomeTeam"].nunique())
        c3.metric("Goles por partido", round((df.FTHG + df.FTAG).mean(), 2))
        st.subheader("Últimos resultados del archivo")
        st.dataframe(df.tail(15).sort_index(ascending=False), use_container_width=True)
        st.subheader("Goles por equipo")
        team_stats = pd.concat([
            df.groupby("HomeTeam").agg(goles_favor=("FTHG", "mean"), goles_contra=("FTAG", "mean")).rename_axis("Equipo"),
            df.groupby("AwayTeam").agg(goles_favor=("FTAG", "mean"), goles_contra=("FTHG", "mean")).rename_axis("Equipo")
        ]).groupby(level=0).mean().round(2).sort_values("goles_favor", ascending=False)
        st.dataframe(team_stats, use_container_width=True)

    with tab2:
        teams = sorted(set(df.HomeTeam).union(set(df.AwayTeam)))
        left, right = st.columns(2)
        home = left.selectbox("Equipo local", teams, index=0, key="pred_home")
        away_options = [t for t in teams if t != home]
        default_away_index = min(1, len(away_options) - 1)
        if st.session_state.get("pred_away") not in away_options:
            st.session_state["pred_away"] = away_options[default_away_index]
        away = right.selectbox("Equipo visitante", away_options, key="pred_away")

        notice = st.session_state.pop("calendar_notice", None)
        if notice:
            st.info(notice)

        if st.button("Calcular predicción", type="primary"):
            result = predict_match(df, home, away)
            st.subheader(f"{home} vs {away}")
            cols = st.columns(3)
            cols[0].metric("Victoria local", f"{result['home_win']:.1%}")
            cols[1].metric("Empate", f"{result['draw']:.1%}")
            cols[2].metric("Victoria visitante", f"{result['away_win']:.1%}")
            st.markdown("#### Mercados de goles")
            a, b, c = st.columns(3)
            a.metric("Más de 1,5 goles", f"{result['over15']:.1%}")
            b.metric("Más de 2,5 goles", f"{result['over25']:.1%}")
            c.metric("Ambos marcan", f"{result['btts']:.1%}")
            st.markdown("#### Goles esperados por el modelo")
            st.write(f"{home}: **{result['lambda_home']:.2f}** · {away}: **{result['lambda_away']:.2f}**")
            st.caption("Estimación basada en promedios históricos local/visitante y distribución de Poisson. No incorpora alineaciones, lesiones ni cuotas.")
            if "corners_home" in result:
                st.markdown("#### Estadísticas adicionales")
                x, y = st.columns(2)
                x.metric("Córners esperados", f"{result['corners_home'] + result['corners_away']:.1f}")
                y.metric("Tarjetas amarillas esperadas", f"{result['cards_home'] + result['cards_away']:.1f}")
            st.warning("No interpretes estas probabilidades como garantía ni como recomendación de apuesta. Comprueba primero el rendimiento histórico del modelo.")
        st.info("Elige dos equipos y pulsa «Calcular predicción». Para resultados fiables necesitas suficientes partidos recientes y comparables.")

    with tab3:
        st.subheader("Importar tus estadísticas")
        st.write("Columnas mínimas: Date, HomeTeam, AwayTeam, FTHG, FTAG. Columnas opcionales: HC, AC, HY, AY.")
        st.download_button("Descargar plantilla CSV", df.head(0).to_csv(index=False).encode("utf-8"), "plantilla_partidos.csv", "text/csv")
        st.write("Para empezar con datos históricos gratuitos, consulta football-data.co.uk. Descarga el CSV de una liga/temporada y súbelo en la barra lateral.")
        st.link_button("Abrir football-data.co.uk", "https://www.football-data.co.uk/")
        st.dataframe(df.head(10), use_container_width=True)

    with tab4:
        st.markdown("""
        **Qué hace esta primera versión**
        - Lee resultados históricos en CSV.
        - Calcula promedios de goles como local y visitante.
        - Estima probabilidades 1X2 y mercados básicos de goles mediante Poisson.
        - Si el CSV incluye córners y tarjetas, muestra promedios descriptivos.

        **Qué todavía no hace**
        - No consulta cuotas en tiempo real.
        - No incorpora lesiones, alineaciones ni xG.
        - No es todavía un modelo entrenado de Machine Learning.
        - Las predicciones dependen de los resultados históricos que cargues.

        **Calendario**
        - Consulta los partidos programados mediante API Ninjas.
        - Los horarios se muestran en hora peninsular española.
        - El calendario no modifica por sí mismo las probabilidades del modelo.
        """)

    with tab5:
        st.subheader("Calendario de partidos")
        st.write("Consulta los encuentros programados por competición y fecha. Los horarios se muestran en hora peninsular española.")

        api_key = get_api_key()
        if not api_key:
            st.error("No encuentro API_NINJAS_KEY en los Secrets de Streamlit. Revisa Manage app → Settings → Secrets.")
        else:
            today = date.today()
            days_to_saturday = (5 - today.weekday()) % 7
            if today.weekday() == 6:
                days_to_saturday = 6
            default_start = today + timedelta(days=days_to_saturday)
            default_end = default_start + timedelta(days=1)

            selected_dates = st.date_input(
                "Selecciona el día o el intervalo de fechas",
                value=(default_start, default_end),
                help="Para consultar el fin de semana, deja seleccionado sábado y domingo."
            )
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
            else:
                start_date = selected_dates
                end_date = selected_dates

            if (end_date - start_date).days > 6:
                st.warning("Para evitar consumir demasiadas consultas, selecciona un máximo de 7 días.")
            else:
                competition_names = list(COMPETITIONS.keys())
                chosen_competitions = st.multiselect(
                    "Competiciones",
                    competition_names,
                    default=competition_names,
                    help="Puedes seleccionar una o varias competiciones."
                )

                if st.button("🔎 Consultar calendario", type="primary"):
                    fixtures = []
                    errors_api = []
                    total_days = (end_date - start_date).days + 1
                    with st.spinner("Consultando los calendarios..."):
                        for competition_name in chosen_competitions:
                            league_code = COMPETITIONS[competition_name]
                            for day_offset in range(total_days):
                                match_day = start_date + timedelta(days=day_offset)
                                try:
                                    day_matches = get_fixtures(
                                        league_code, match_day.isoformat(), api_key
                                    )
                                    for match in day_matches:
                                        home_team = (match.get("home_team") or {}).get("name", "Local")
                                        away_team = (match.get("away_team") or {}).get("name", "Visitante")
                                        start_time = match.get("start_time")
                                        try:
                                            date_label = datetime.fromisoformat(
                                                start_time.replace("Z", "+00:00")
                                            ).astimezone(MADRID_TZ).strftime("%d/%m/%Y") if start_time else match_day.strftime("%d/%m/%Y")
                                        except (ValueError, TypeError):
                                            date_label = match_day.strftime("%d/%m/%Y")

                                        home_score = match.get("home_score")
                                        away_score = match.get("away_score")
                                        score = (
                                            f"{home_score} - {away_score}"
                                            if home_score is not None and away_score is not None
                                            else "—"
                                        )
                                        fixtures.append({
                                            "Fecha": date_label,
                                            "Hora (España)": madrid_time(start_time),
                                            "Competición": competition_name,
                                            "Local": home_team,
                                            "Visitante": away_team,
                                            "Estado": spanish_status(match.get("status")),
                                            "Resultado": score,
                                            "_game_id": match.get("game_id", ""),
                                        })
                                except requests.HTTPError as exc:
                                    detail = exc.response.text[:250] if exc.response is not None else str(exc)
                                    errors_api.append(f"{competition_name} ({match_day:%d/%m}): {detail}")
                                except (requests.RequestException, ValueError) as exc:
                                    errors_api.append(f"{competition_name} ({match_day:%d/%m}): {exc}")

                    if fixtures:
                        fixtures = sorted(fixtures, key=lambda item: (item["Fecha"], item["Hora (España)"], item["Competición"]))
                        st.session_state["calendar_fixtures"] = fixtures
                    else:
                        st.session_state["calendar_fixtures"] = []
                    st.session_state["calendar_errors"] = errors_api

                fixtures = st.session_state.get("calendar_fixtures", [])
                errors_api = st.session_state.get("calendar_errors", [])
                if errors_api:
                    with st.expander(f"Incidencias en {len(errors_api)} consultas"):
                        for error in errors_api:
                            st.warning(error)

                if fixtures:
                    visible_columns = ["Fecha", "Hora (España)", "Competición", "Local", "Visitante", "Estado", "Resultado"]
                    st.success(f"Se han encontrado {len(fixtures)} partidos.")
                    st.dataframe(
                        pd.DataFrame(fixtures)[visible_columns],
                        use_container_width=True,
                        hide_index=True
                    )

                    match_labels = [
                        f"{item['Fecha']} · {item['Hora (España)']} · {item['Competición']} · {item['Local']} vs {item['Visitante']}"
                        for item in fixtures
                    ]
                    selected_label = st.selectbox("Selecciona un partido para llevarlo a Predicción", match_labels)
                    if st.button("Preparar este partido en Predicción"):
                        selected_index = match_labels.index(selected_label)
                        selected_match = fixtures[selected_index]
                        st.session_state["pending_fixture"] = (
                            selected_match["Local"], selected_match["Visitante"]
                        )
                        st.success("Partido seleccionado. Abre la pestaña «Predicción» para continuar.")
                        st.rerun()
                elif "calendar_fixtures" in st.session_state:
                    st.info("No se han encontrado partidos para esos filtros y fechas.")
                else:
                    st.info("Selecciona las fechas y competiciones y pulsa «Consultar calendario».")

                st.caption("La disponibilidad y el detalle de los partidos dependen de API Ninjas y de tu cuota gratuita. Los resultados de partidos en directo pueden estar limitados en el plan gratuito.")
else:
    st.info("Carga un archivo CSV para ver el análisis. El calendario se muestra cuando hay datos cargados o se usa el modo de demostración.")
