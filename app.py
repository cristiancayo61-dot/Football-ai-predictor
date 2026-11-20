import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO

from predictor import predict_match, compare_odds, match_team_name, poisson_over

st.set_page_config(page_title="Football AI Predictor", page_icon="⚽", layout="wide")

API_URL = "https://api.api-ninjas.com/v1/soccermatches"
MADRID_TZ = ZoneInfo("Europe/Madrid")

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
                    headers={"User-Agent": "Football-AI-Predictor-personal-use/1.0"},
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                raw = pd.read_csv(BytesIO(response.content), encoding="latin-1")
                raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
                if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(raw.columns):
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
    shortcuts = [
        ("Hoy", today),
        ("Mañana", today + timedelta(days=1)),
        ("Sábado", today + timedelta(days=(5 - today.weekday()) % 7)),
        ("Domingo", today + timedelta(days=(6 - today.weekday()) % 7)),
    ]
    for col, (label, value) in zip(cols, shortcuts):
        if col.button(label, use_container_width=True):
            st.session_state.calendar_date = value
            st.rerun()

    chosen_date = st.date_input(
        "Fecha", value=st.session_state.calendar_date,
        key="calendar_date", format="DD/MM/YYYY",
    )
    active = [name for name, cfg in COMPETITIONS.items() if cfg["fd"] is not None]
    selected_competitions = st.multiselect(
        "Competiciones", active, default=active, key="calendar_competitions",
    )
    api_key = get_api_key()
    if not api_key:
        st.error("No encuentro API_NINJAS_KEY en los Secrets de Streamlit.")
        return

    fixtures, errors = [], []
    with st.spinner("Consultando el calendario..."):
        for comp in selected_competitions:
            try:
                payload = get_fixtures(
                    COMPETITIONS[comp]["api"], chosen_date.isoformat(), api_key,
                )
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    start = item.get("start_time") or item.get("date")
                    dt = parse_api_datetime(start)
                    home = get_team_name(item.get("home_team"), "Local")
                    away = get_team_name(item.get("away_team"), "Visitante")
                    hs, aws = item.get("home_score"), item.get("away_score")
                    fixtures.append({
                        "Hora": dt.strftime("%H:%M") if dt else "Pendiente",
                        "Competición": comp,
                        "Local": home,
                        "Visitante": away,
                        "Estado": item.get("status", "Sin estado"),
                        "Resultado": f"{hs} - {aws}" if hs is not None and aws is not None else "—",
                    })
            except Exception as exc:
                errors.append(f"{comp}: {exc}")

    if errors:
        with st.expander(f"Incidencias ({len(errors)})"):
            for error in errors:
                st.warning(error)

    if fixtures:
        fixtures.sort(key=lambda x: (x["Hora"], x["Competición"], x["Local"]))
        st.session_state["day_fixtures"] = fixtures
        st.success(f"{len(fixtures)} partidos encontrados.")
        st.dataframe(pd.DataFrame(fixtures), use_container_width=True, hide_index=True)
        labels = [
            f'{x["Hora"]} · {x["Competición"]} · {x["Local"]} vs {x["Visitante"]}'
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


def _bet_log_metrics(log):
    realised = pending = settled_stakes = 0.0
    wins = losses = voids = 0
    for item in log:
        try:
            stake = max(0.0, float(item.get("Stake", 0) or 0))
            odds = float(item.get("Cuota Bet365", 0) or 0)
        except (TypeError, ValueError):
            stake, odds = 0.0, 0.0
        status = item.get("Resultado", "Pendiente")
        if status == "Pendiente":
            pending += stake
        elif status == "Ganada":
            realised += stake * (odds - 1)
            settled_stakes += stake
            wins += 1
        elif status == "Perdida":
            realised -= stake
            settled_stakes += stake
            losses += 1
        elif status == "Anulada":
            voids += 1
    return {
        "realised": realised, "pending": pending,
        "settled_stakes": settled_stakes, "wins": wins,
        "losses": losses, "voids": voids,
        "roi": realised / settled_stakes if settled_stakes else 0.0,
    }


def render_odds(result, fixture_name="Partido seleccionado"):
    st.markdown("#### Comparador de cuotas Bet365")
    st.caption("Introduce manualmente la cuota decimal de Bet365. Las probabilidades son estimaciones, no garantías.")

    log = st.session_state.setdefault("bet_log", [])
    stats = _bet_log_metrics(log)
    if "bankroll_initial" not in st.session_state:
        st.session_state["bankroll_initial"] = 100.0
    st.number_input(
        "Banca inicial (€)", min_value=1.0, max_value=1000000.0,
        step=10.0, key="bankroll_initial",
        help="Importe de referencia. No está conectado con tu cuenta de apuestas.",
    )
    bankroll = float(st.session_state["bankroll_initial"])
    available = max(0.0, bankroll + stats["realised"] - stats["pending"])
    st.metric("Saldo disponible estimado", f"{available:.2f} €")

    markets = {
        "Victoria local": result.get("home_win"),
        "Empate": result.get("draw"),
        "Victoria visitante": result.get("away_win"),
        "Doble oportunidad 1X": result.get("double_chance_1x"),
        "Doble oportunidad X2": result.get("double_chance_x2"),
        "Doble oportunidad 12": result.get("double_chance_12"),
        "Más de 1,5 goles": result.get("over15"),
        "Más de 2,5 goles": result.get("over25"),
        "Más de 3,5 goles": result.get("over35"),
        "Ambos equipos marcan - Sí": result.get("btts"),
    }
    auxiliary = result.get("auxiliary", {})
    market_lines = {
        "corners": ("córners totales", [7.5, 8.5, 9.5, 10.5]),
        "fouls": ("faltas totales", [19.5, 21.5, 23.5, 25.5]),
        "yellow_cards": ("tarjetas amarillas totales", [2.5, 3.5, 4.5, 5.5]),
        "saves": ("paradas totales estimadas", [3.5, 4.5, 5.5, 6.5]),
        "shots_on_target": ("tiros a puerta totales", [7.5, 8.5, 9.5, 10.5]),
    }
    for key, (label, lines) in market_lines.items():
        item = auxiliary.get(key)
        if item and item.get("total") is not None:
            for line in lines:
                p_over = poisson_over(item["total"], line)
                if p_over is not None:
                    line_text = str(line).replace(".", ",")
                    markets[f"Más de {line_text} {label}"] = p_over
                    markets[f"Menos de {line_text} {label}"] = 1.0 - p_over

    market = st.selectbox("Mercado", list(markets.keys()), key="odds_market")
    probability = markets[market]
    odd = st.number_input(
        "Cuota Bet365", min_value=1.01, max_value=1000.0,
        value=2.00, step=0.05, key="odds_decimal",
    )
    if probability is None or pd.isna(probability):
        st.warning("No hay una probabilidad válida para este mercado.")
        return

    probability = max(0.0, min(1.0, float(probability)))
    comp = compare_odds(probability, odd)
    edge = probability - comp["implied_probability"]
    a, b, c = st.columns(3)
    a.metric("Probabilidad estimada", f"{probability:.1%}")
    b.metric("Probabilidad implícita", f"{comp['implied_probability']:.1%}")
    c.metric("Valor esperado teórico", f"{comp['expected_value']:+.1%}")
    st.write(f"Cuota justa estimada: **{comp['fair_odds']:.2f}**")
    st.write(f"Ventaja estimada: **{edge:+.1%}**")

    home_n = int(result.get("home_matches") or 0)
    away_n = int(result.get("away_matches") or 0)
    min_sample = min(home_n, away_n)
    st.caption(f"Muestra: {home_n} partidos del local en casa y {away_n} del visitante fuera.")

    b = odd - 1.0
    kelly = max(0.0, ((probability * odd) - 1.0) / b) if b > 0 else 0.0
    suggested = min(available * kelly * 0.25, available * 0.01)
    eligible = (
        0 < probability < 1 and comp["expected_value"] > 0
        and edge >= 0.03 and min_sample >= 8 and available > 0
    )
    if eligible and suggested > 0:
        st.success(f"Candidato a revisar. Stake orientativo: **{suggested:.2f} €**.")
        st.caption("Kelly fraccional al 25%, limitado al 1% del saldo disponible. No garantiza rentabilidad.")
    else:
        reasons = []
        if comp["expected_value"] <= 0:
            reasons.append("valor esperado no positivo")
        if edge < 0.03:
            reasons.append("ventaja inferior al 3%")
        if min_sample < 8:
            reasons.append("muestra inferior a 8 partidos")
        if available <= 0:
            reasons.append("saldo disponible insuficiente")
        st.warning("No supera los filtros prudentes: " + "; ".join(reasons) + ". Stake sugerido: 0 €.")

    if available > 0:
        stake = st.number_input(
            "Importe real que quieres registrar (€)",
            min_value=0.0, max_value=float(available),
            value=float(min(suggested, available)), step=0.50,
            key="actual_bet_stake",
        )
    else:
        stake = 0.0
        st.info("No queda saldo disponible para registrar una apuesta.")
    if stake > 0 and not eligible:
        st.warning("El importe indicado no supera los filtros del modelo. Revisa el riesgo.")

    if st.button("Registrar apuesta en el seguimiento", type="primary", disabled=stake <= 0):
        st.session_state["bet_log"].append({
            "Fecha registro": datetime.now(MADRID_TZ).strftime("%Y-%m-%d %H:%M"),
            "Partido": fixture_name,
            "Mercado": market,
            "Probabilidad modelo": round(probability, 6),
            "Cuota Bet365": float(odd),
            "Cuota justa": round(float(comp["fair_odds"]), 4),
            "Valor esperado": round(float(comp["expected_value"]), 6),
            "Ventaja": round(float(edge), 6),
            "Muestra local casa": home_n,
            "Muestra visitante fuera": away_n,
            "Stake": round(float(stake), 2),
            "Resultado": "Pendiente",
            "Beneficio neto": 0.0,
        })
        st.success("Apuesta registrada. Descarga el CSV para conservar una copia.")
        st.rerun()


def render_bet_tracker():
    st.divider()
    st.subheader("Seguimiento de apuestas")
    st.caption("El historial se conserva durante la sesión. Descarga el CSV y vuelve a importarlo para conservarlo entre sesiones.")
    log = st.session_state.setdefault("bet_log", [])
    if log:
        legacy = 0
        for item in log:
            if "Stake" not in item or item.get("Stake") in ("", None):
                item["Stake"] = 1.0
                legacy += 1
            item.setdefault("Beneficio neto", 0.0)
            item.setdefault("Resultado", "Pendiente")
        if legacy:
            st.warning(f"{legacy} apuestas antiguas no tenían importe. Se les asigna 1 € como unidad aproximada.")
        stats = _bet_log_metrics(log)
        initial = float(st.session_state.get("bankroll_initial", 100.0))
        available = max(0.0, initial + stats["realised"] - stats["pending"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Banca inicial", f"{initial:.2f} €")
        c2.metric("Saldo disponible", f"{available:.2f} €")
        c3.metric("Beneficio liquidado", f"{stats['realised']:+.2f} €")
        c4.metric("ROI liquidado", f"{stats['roi']:+.1%}")

        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar historial CSV",
            data=df_log.to_csv(index=False).encode("utf-8-sig"),
            file_name="football_ai_historial_apuestas.csv",
            mime="text/csv",
        )
        pending = [i for i, item in enumerate(log) if item.get("Resultado") == "Pendiente"]
        if pending:
            idx = st.selectbox(
                "Apuesta que quieres liquidar", pending,
                format_func=lambda i: (
                    f'{log[i].get("Partido", "Partido")} · {log[i].get("Mercado", "Mercado")} · '
                    f'cuota {log[i].get("Cuota Bet365", "")} · {float(log[i].get("Stake", 0) or 0):.2f} €'
                ),
            )
            status = st.selectbox("Resultado", ["Ganada", "Perdida", "Anulada"])
            if st.button("Guardar resultado de la apuesta"):
                item = log[idx]
                amount = float(item.get("Stake", 0) or 0)
                odds = float(item.get("Cuota Bet365", 0) or 0)
                item["Resultado"] = status
                if status == "Ganada":
                    item["Beneficio neto"] = round(amount * (odds - 1), 2)
                elif status == "Perdida":
                    item["Beneficio neto"] = round(-amount, 2)
                else:
                    item["Beneficio neto"] = 0.0
                st.session_state["bet_log"] = log
                st.rerun()

        total = stats["wins"] + stats["losses"] + stats["voids"]
        st.write(
            f"Liquidadas: **{total}** · Ganadas: **{stats['wins']}** · "
            f"Perdidas: **{stats['losses']}** · Anuladas: **{stats['voids']}** · "
            f"Pendiente: **{stats['pending']:.2f} €**"
        )
        st.caption("Ganada = stake × (cuota − 1); perdida = −stake; anulada = 0. ROI = beneficio neto / stakes liquidados no anulados.")
    else:
        st.info("Todavía no has registrado apuestas.")

    uploaded = st.file_uploader("Importar un historial CSV anterior", type=["csv"], key="bet_log_upload")
    if uploaded is not None and st.button("Importar historial"):
        try:
            imported = pd.read_csv(uploaded).fillna("")
            required = {"Partido", "Mercado", "Cuota Bet365", "Resultado"}
            if not required.issubset(imported.columns):
                st.error("El CSV no contiene las columnas necesarias.")
            else:
                rows = imported.to_dict("records")
                for item in rows:
                    item.setdefault("Stake", 1.0)
                    item.setdefault("Beneficio neto", 0.0)
                    item.setdefault("Resultado", "Pendiente")
                st.session_state["bet_log"] = rows
                st.success(f"Se han importado {len(rows)} registros.")
                st.rerun()
        except Exception as exc:
            st.error(f"No se pudo importar el CSV: {exc}")


def render_day_radar(raw):
    fixtures = st.session_state.get("day_fixtures", [])
    if not fixtures:
        return
    st.subheader("Radar de la jornada")
    st.caption("Probabilidades estimadas por partido. La probabilidad más alta no equivale necesariamente a valor esperado positivo.")
    rows = []
    for fixture in fixtures:
        comp = fixture["Competición"]
        league_df = raw[raw["League"] == comp].copy()
        home = match_team_name(fixture["Local"], league_df)
        away = match_team_name(fixture["Visitante"], league_df)
        match_name = f'{fixture["Local"]} vs {fixture["Visitante"]}'
        if not home or not away or home == away:
            rows.append({
                "Hora": fixture["Hora"], "Competición": comp, "Partido": match_name,
                "Mercado": "—", "Selección": "Pendiente de revisar nombres",
                "Probabilidad": None, "Cuota justa": None,
            })
            continue
        try:
            pred = predict_match(league_df, home, away)
            options = [
                ("1X2", "Victoria local", pred["home_win"]),
                ("1X2", "Empate", pred["draw"]),
                ("1X2", "Victoria visitante", pred["away_win"]),
                ("Goles", "Más de 1,5", pred["over15"]),
                ("Goles", "Más de 2,5", pred["over25"]),
                ("Goles", "Más de 3,5", pred["over35"]),
                ("Ambos marcan", "Sí", pred["btts"]),
            ]
            for market, selection, p in options:
                rows.append({
                    "Hora": fixture["Hora"], "Competición": comp,
                    "Partido": f"{home} vs {away}", "Mercado": market,
                    "Selección": selection, "Probabilidad": p,
                    "Cuota justa": 1 / p if p > 0 else None,
                })
        except Exception:
            rows.append({
                "Hora": fixture["Hora"], "Competición": comp, "Partido": match_name,
                "Mercado": "—", "Selección": "No se pudo calcular",
                "Probabilidad": None, "Cuota justa": None,
            })
    radar = pd.DataFrame(rows).sort_values("Probabilidad", ascending=False, na_position="last")
    if not radar.empty:
        radar["Probabilidad"] = radar["Probabilidad"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        radar["Cuota justa"] = radar["Cuota justa"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    st.dataframe(radar, use_container_width=True, hide_index=True)


def main():
    show_calendar()
    st.divider()
    st.header("Análisis y apuestas")
    with st.sidebar:
        st.header("Datos históricos")
        st.caption("Resultados históricos descargados de Football-Data.co.uk.")
        if st.button("Actualizar histórico ahora"):
            get_all_history.clear()
            st.rerun()

    selected = st.session_state.get("selected_fixture")
    raw, download_errors = get_all_history()
    if raw.empty:
        st.error("No se pudo cargar el histórico. Comprueba la conexión o inténtalo más tarde.")
        st.stop()

    render_day_radar(raw)
    if not selected:
        st.info("Selecciona primero un partido del calendario para ver su análisis automático.")
        st.caption(f"Histórico cargado: {len(raw):,} resultados en {raw['League'].nunique()} competiciones.")
        with st.expander("Estado de las fuentes de datos"):
            for error in download_errors[:30]:
                st.write(error)
        st.stop()

    st.info(f"Partido seleccionado: **{selected['Local']} vs {selected['Visitante']}** · {selected['Competición']}")
    fd_code = COMPETITIONS[selected["Competición"]]["fd"]
    if not fd_code:
        st.warning("Este torneo está en el calendario, pero todavía no tiene histórico incorporado.")
        st.stop()

    league_df = raw[raw["League"] == selected["Competición"]].copy()
    home = match_team_name(selected["Local"], league_df)
    away = match_team_name(selected["Visitante"], league_df)
    if not home or not away:
        st.warning("No he encontrado una coincidencia segura para los nombres de los equipos.")
        teams = sorted(set(league_df["HomeTeam"]).union(set(league_df["AwayTeam"])))
        st.selectbox("Equipo local", teams, key="manual_home")
        st.selectbox("Equipo visitante", teams, key="manual_away")
        if st.button("Analizar equipos seleccionados"):
            home, away = st.session_state.manual_home, st.session_state.manual_away
        else:
            st.stop()
    if home == away:
        st.error("Los equipos local y visitante no pueden ser el mismo.")
        st.stop()

    result = predict_match(league_df, home, away)
    st.session_state.last_prediction = {"home": home, "away": away, "result": result}
    st.subheader(f"{home} vs {away}")

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

    st.write(f"Goles esperados: **{home} {result['lambda_home']:.2f}** · **{away} {result['lambda_away']:.2f}**")
    st.markdown("#### Otros mercados estadísticos")
    aux = result.get("auxiliary", {})
    any_aux = False
    for key, title in [
        ("corners", "Córners"), ("fouls", "Faltas cometidas"),
        ("yellow_cards", "Tarjetas amarillas"),
        ("shots_on_target", "Tiros a puerta"), ("saves", "Paradas estimadas"),
    ]:
        item = aux.get(key)
        if item and item.get("total") is not None:
            any_aux = True
            st.write(
                f"**{title}:** {home} {item['home']:.1f} · {away} {item['away']:.1f} · "
                f"Total esperado {item['total']:.1f}"
            )
        else:
            st.caption(f"{title}: no hay suficientes estadísticas históricas para estimarlo.")
    if not any_aux:
        st.info("No hay estadísticas auxiliares suficientes para esta competición.")

    st.markdown("#### Marcadores exactos más probables")
    scores = pd.DataFrame(result["top_scores"], columns=["Marcador", "Probabilidad"])
    scores["Probabilidad"] = scores["Probabilidad"].map(lambda p: f"{p:.1%}")
    st.dataframe(scores, use_container_width=True, hide_index=True)
    st.caption(
        f"Muestra utilizada: {result.get('home_matches', 0)} partidos del local en casa y "
        f"{result.get('away_matches', 0)} del visitante fuera."
    )

    render_odds(result, f"{home} vs {away}")
    st.warning(
        "Modelo Poisson básico, no validado como sistema rentable. Los mercados auxiliares "
        "son aproximaciones basadas en promedios de equipos, no datos en directo. "
        "No incorpora alineaciones, lesiones, árbitro ni cuotas en tiempo real."
    )
    st.info(
        "Los mercados individuales de jugadores quedan pendientes de una fuente fiable "
        "de estadísticas por futbolista. La aplicación no inventará esas cifras."
    )
    render_bet_tracker()

    with st.expander("Sobre el histórico y sus limitaciones"):
        st.write("Se consultan temporadas 2022/23 a 2026/27 cuando los archivos están disponibles.")
        st.write("Fuente: Football-Data.co.uk. Los datos históricos pueden contener estadísticas incompletas.")
        for error in download_errors[:30]:
            st.write(error)


if __name__ == "__main__":
    main()
