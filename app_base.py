import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO

from predictor import predict_match, match_team_name, poisson_over
import team_form
from odds_api import get_match_odds, odds_for_market
from intelligent_report import render_intelligent_report

st.set_page_config(page_title="Football AI Predictor", page_icon="⚽", layout="wide")

API_URL = "https://api.api-ninjas.com/v1/soccermatches"
TZ = ZoneInfo("Europe/Madrid")
LEAGUES = {
    "LaLiga (España)": ("laliga", "SP1"),
    "Segunda División (España)": ("laliga2", "SP2"),
    "Premier League (Inglaterra)": ("epl", "E0"),
    "Bundesliga (Alemania)": ("bundesliga", "D1"),
    "Serie A (Italia)": ("seriea", "I1"),
    "Ligue 1 (Francia)": ("ligue1", "F1"),
    "Eredivisie (Países Bajos)": ("eredivisie", "N1"),
    "Champions League": ("ucl", None), "Europa League": ("uel", None),
}
SEASONS = ["2223", "2324", "2425", "2526", "2627"]
RISKS = {
    "Equilibrado": {"normal": .02, "medium": .025, "high": .03, "max": .03, "combo": .015, "exposure": .05},
    "Agresivo": {"normal": .03, "medium": .04, "high": .05, "max": .05, "combo": .02, "exposure": .08},
}
NAV = ["Inicio", "Partidos", "Análisis IA", "Análisis por equipo", "Oportunidades", "Mi banca", "Resultados y estadísticas"]
AUTO_MARKETS = ["Victoria local", "Empate", "Victoria visitante",
                "Más de 1,5 goles", "Más de 2,5 goles", "Más de 3,5 goles"]

def today():
    return datetime.now(TZ).date()

def api_key():
    try:
        return st.secrets.get("API_NINJAS_KEY", "")
    except Exception:
        return ""

def team_name(v, fallback):
    if isinstance(v, dict):
        return str(v.get("name") or fallback)
    return v.strip() if isinstance(v, str) and v.strip() else fallback

def parse_dt(v):
    if not v:
        return None
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=ZoneInfo("UTC"))
        return d.astimezone(TZ)
    except (ValueError, TypeError):
        return None

@st.cache_data(ttl=900, show_spinner=False)
def fixtures_for(league, day, key):
    r = requests.get(API_URL, params={"league": league, "date": day},
                     headers={"X-Api-Key": key}, timeout=25)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("Formato inesperado de API Ninjas")
    return data

@st.cache_data(ttl=21600, show_spinner=False)
def historical():
    frames, errors = [], []
    for label, (_, league_code) in LEAGUES.items():
        if not league_code:
            continue
        for season in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
            try:
                response = requests.get(url, timeout=20, headers={"User-Agent": "Football-AI-Predictor/1.0"})
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                frame = pd.read_csv(BytesIO(response.content), encoding="latin-1")
                frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
                if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(frame.columns):
                    errors.append(f"{label} {season}: columnas básicas ausentes")
                    continue
                frame["League"], frame["Season"] = label, season
                frames.append(frame)
            except Exception as exc:
                errors.append(f"{label} {season}: {type(exc).__name__}")
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), errors

def stats_log():
    log = st.session_state.setdefault("bet_log", [])
    profit = pending = settled = 0.0
    for item in log:
        stake = float(item.get("Stake", 0) or 0)
        odds = float(item.get("Cuota Bet365", 0) or 0)
        status = item.get("Resultado", "Pendiente")
        if status == "Pendiente":
            pending += stake
        elif status == "Ganada":
            profit += stake * (odds - 1)
            settled += stake
        elif status == "Perdida":
            profit -= stake
            settled += stake
    return profit, pending, settled

def bankroll():
    profit, pending, _ = stats_log()
    return max(0.0, float(st.session_state.get("bankroll_initial", 100.0)) + profit - pending)

def current_profile():
    return RISKS.get(st.session_state.get("risk_profile", "Agresivo"), RISKS["Agresivo"])

def markets(pred):
    result = {
        "Victoria local": pred.get("home_win"), "Empate": pred.get("draw"),
        "Victoria visitante": pred.get("away_win"),
        "Doble oportunidad 1X": pred.get("double_chance_1x"),
        "Doble oportunidad X2": pred.get("double_chance_x2"),
        "Doble oportunidad 12": pred.get("double_chance_12"),
        "Más de 1,5 goles": pred.get("over15"), "Más de 2,5 goles": pred.get("over25"),
        "Más de 3,5 goles": pred.get("over35"), "Ambos equipos marcan - Sí": pred.get("btts"),
    }
    definitions = {
        "corners": ("córners totales", [7.5, 8.5, 9.5, 10.5]),
        "fouls": ("faltas totales", [19.5, 21.5, 23.5, 25.5]),
        "yellow_cards": ("tarjetas amarillas totales", [2.5, 3.5, 4.5, 5.5]),
        "saves": ("paradas estimadas", [3.5, 4.5, 5.5, 6.5]),
        "shots_on_target": ("tiros a puerta totales", [7.5, 8.5, 9.5, 10.5]),
    }
    for key, (label, lines) in definitions.items():
        item = pred.get("auxiliary", {}).get(key)
        if item and item.get("total") is not None:
            for line in lines:
                probability = poisson_over(item["total"], line)
                if probability is not None:
                    shown_line = str(line).replace(".", ",")
                    result[f"Más de {shown_line} {label}"] = probability
                    result[f"Menos de {shown_line} {label}"] = 1 - probability
    return result

def stake_advice(p, odds, edge, cash, profile, combo=False):
    if not (0 < p < 1 and odds > 1 and cash > 0):
        return 0.0, "Probabilidad o saldo insuficiente"
    if p * odds - 1 <= 0 or edge < .01:
        return 0.0, "Sin ventaja/valor esperado suficiente"
    if combo:
        cap, tier = profile["combo"], "combinada"
    elif p >= .68 and edge >= .04:
        tier, cap = "alta", profile["high"]
    elif p >= .57 and edge >= .025:
        tier, cap = "media", profile["medium"]
    elif p >= .50 and edge >= .01:
        tier, cap = "normal", profile["normal"]
    else:
        return 0.0, "Confianza insuficiente"
    kelly = max(0.0, (p * odds - 1) / (odds - 1))
    amount = cash * min(cap, kelly * .50)
    _, pending, _ = stats_log()
    room = max(0.0, float(st.session_state.get("bankroll_initial", 100.0)) * profile["exposure"] - pending)
    return min(amount, room), tier

def show_calendar():
    st.subheader("Calendario de partidos")
    day = today()

    # Al cambiar el día, el calendario vuelve automáticamente a la fecha actual.
    if st.session_state.get("calendar_anchor_date") != day:
        st.session_state.calendar_anchor_date = day
        st.session_state.calendar_date = day
    elif "calendar_date" not in st.session_state:
        st.session_state.calendar_date = day

    weekday_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    quick_days = [
        (("Hoy" if offset == 0 else weekday_names[(day + timedelta(days=offset)).weekday()]),
         day + timedelta(days=offset))
        for offset in range(5)
    ]

    selected_date = st.session_state.get("calendar_date", day)
    columns = st.columns(5)
    for col, (label, date_value) in zip(columns, quick_days):
        button_type = "primary" if selected_date == date_value else "secondary"
        button_label = f"{label}\n{date_value:%d/%m}"
        if col.button(button_label, key=f"calendar_day_{date_value.isoformat()}",
                      type=button_type, use_container_width=True):
            st.session_state.calendar_date = date_value
            st.rerun()

    chosen = st.date_input("O elegir otra fecha", key="calendar_date", format="DD/MM/YYYY")
    competitions = [name for name, value in LEAGUES.items() if value[1]]
    selected = st.multiselect("Competiciones", competitions, default=competitions, key="calendar_competitions")
    key = api_key()
    if not key:
        st.error("No encuentro API_NINJAS_KEY en los Secrets de Streamlit.")
        return
    matches, errors = [], []
    with st.spinner("Consultando calendario de API Ninjas..."):
        for competition in selected:
            try:
                for item in fixtures_for(LEAGUES[competition][0], chosen.isoformat(), key):
                    if not isinstance(item, dict):
                        continue
                    dt = parse_dt(item.get("start_time") or item.get("date"))
                    home_score, away_score = item.get("home_score"), item.get("away_score")
                    matches.append({
                        "Hora": dt.strftime("%H:%M") if dt else "Pendiente",
                        "Competición": competition,
                        "Local": team_name(item.get("home_team"), "Local"),
                        "Visitante": team_name(item.get("away_team"), "Visitante"),
                        "Estado": item.get("status", "Sin estado"),
                        "Resultado": f"{home_score} - {away_score}" if home_score is not None and away_score is not None else "—",
                    })
            except Exception as exc:
                errors.append(f"{competition}: {exc}")
    if errors:
        with st.expander(f"Incidencias de calendario ({len(errors)})"):
            for error in errors:
                st.warning(error)
    st.session_state.day_fixtures = matches
    if not matches:
        st.info(f"No hay partidos para el {chosen:%d/%m/%Y} en las competiciones elegidas.")
        return
    matches.sort(key=lambda x: (x["Hora"], x["Competición"], x["Local"]))
    st.success(f"{len(matches)} partidos encontrados.")
    st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
    labels = [f'{x["Hora"]} · {x["Competición"]} · {x["Local"]} vs {x["Visitante"]}' for x in matches]
    selected_label = st.selectbox("Selecciona un partido", labels, key="fixture_choice")
    if st.button("Abrir ficha de análisis", type="primary"):
        st.session_state.selected_fixture = matches[labels.index(selected_label)]
        st.success("Partido guardado. Abre la sección «Análisis IA» en el menú lateral.")

def selected_teams(raw, fixture):
    if raw is None or raw.empty or "League" not in raw.columns:
        return None, None, None
    competition = fixture.get("Competición", "")
    if competition not in LEAGUES or not LEAGUES[competition][1]:
        return None, None, None
    league = raw[raw.League == competition].copy()
    home = match_team_name(fixture.get("Local", ""), league)
    away = match_team_name(fixture.get("Visitante", ""), league)
    if not home or not away or home == away:
        return league, None, None
    return league, home, away

def _form_panel(team, all_results):
    recent = team_form.last_five(team, all_results)
    st.markdown(f"**{team}**")
    if recent.empty:
        st.info("No se encontraron coincidencias seguras para este equipo.")
        return recent
    summary = team_form.form_summary(recent)
    a, b, c, d = st.columns(4)
    a.metric("Partidos", summary["played"])
    b.metric("Puntos", f'{summary["points"]}/{summary["played"]*3}')
    c.metric("Victorias", summary["wins"])
    d.metric("Goles a favor", f'{summary["goals_for"]:.1f}')
    st.dataframe(recent, use_container_width=True, hide_index=True)
    stat_rows = []
    for stat in ["Córners", "Tiros a puerta", "Faltas", "Amarillas", "Rojas"]:
        average, count = team_form.stat_average(recent, stat)
        stat_rows.append({"Estadística": stat,
                          "Media": f"{average:.2f}" if average is not None else "No disponible",
                          "Cobertura": f"{count}/{len(recent)} partidos"})
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    return recent

def show_recent_form(home, away):
    st.markdown("### Forma reciente · últimos cinco partidos oficiales")
    st.caption("Se consultan competiciones oficiales disponibles en open-matchday. Los datos estadísticos no publicados se muestran como no disponibles.")
    try:
        all_results = team_form.load_official_results()
        if all_results.empty:
            st.warning("La fuente de resultados recientes no devolvió partidos.")
            return
        left, right = st.columns(2)
        with left:
            _form_panel(home, all_results)
        with right:
            _form_panel(away, all_results)
    except Exception as exc:
        st.warning(f"No se pudo cargar la forma reciente: {type(exc).__name__}: {exc}")

def render_analysis(raw, profile, cash):
    fixture = st.session_state.get("selected_fixture")
    if not fixture:
        st.info("Primero selecciona un partido en la sección Partidos.")
        return
    st.title("Análisis IA")
    st.write(f'**{fixture.get("Local")} vs {fixture.get("Visitante")}** · {fixture.get("Competición")} · {fixture.get("Hora")}')
    if raw is None or raw.empty:
        st.warning("No se ha podido cargar el histórico de Football-Data.co.uk.")
        return
    league, home, away = selected_teams(raw, fixture)
    if league is None:
        st.warning("Este torneo está en el calendario, pero no tiene histórico de liga integrado en el modelo actual.")
        return
    if not home or not away:
        st.warning("No he encontrado una coincidencia segura de nombres en el histórico.")
        teams = sorted(set(league.HomeTeam).union(set(league.AwayTeam)))
        left, right = st.columns(2)
        left.selectbox("Equipo local", teams, key="manual_home")
        right.selectbox("Equipo visitante", teams, key="manual_away")
        if st.button("Analizar equipos elegidos"):
            home, away = st.session_state.manual_home, st.session_state.manual_away
        else:
            return
    if home == away:
        st.error("Los equipos no pueden ser el mismo.")
        return
    try:
        pred = predict_match(league, home, away)
    except Exception as exc:
        st.error(f"No se pudo calcular el modelo: {exc}")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Informe inteligente", "Pronóstico", "Goles y mercados", "Cuotas y valor", "Riesgo y stake", "Forma reciente"])
    with tab1:
        render_intelligent_report(fixture, pred, home, away, league, profile, cash)
    with tab2:
        st.subheader("Probabilidades 1X2")
        a, b, c = st.columns(3)
        a.metric("Victoria local", f'{pred["home_win"]:.1%}')
        b.metric("Empate", f'{pred["draw"]:.1%}')
        c.metric("Victoria visitante", f'{pred["away_win"]:.1%}')
        a, b, c = st.columns(3)
        a.metric("Doble oportunidad 1X", f'{pred["double_chance_1x"]:.1%}')
        b.metric("Doble oportunidad X2", f'{pred["double_chance_x2"]:.1%}')
        c.metric("Doble oportunidad 12", f'{pred["double_chance_12"]:.1%}')
        st.write(f'Goles esperados: **{home} {pred["lambda_home"]:.2f}** · **{away} {pred["lambda_away"]:.2f}**')
        st.caption(f'Muestra histórica: {pred.get("home_matches", 0)} partidos del local en casa y {pred.get("away_matches", 0)} del visitante fuera.')
        scores = pd.DataFrame(pred["top_scores"], columns=["Marcador", "Probabilidad"])
        scores["Probabilidad"] = scores["Probabilidad"].map(lambda value: f"{value:.1%}")
        st.markdown("#### Marcadores exactos más probables")
        st.dataframe(scores, use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("Goles y mercados complementarios")
        a, b, c, d = st.columns(4)
        a.metric("Más de 1,5 goles", f'{pred["over15"]:.1%}')
        b.metric("Más de 2,5 goles", f'{pred["over25"]:.1%}')
        c.metric("Más de 3,5 goles", f'{pred["over35"]:.1%}')
        d.metric("Ambos marcan", f'{pred["btts"]:.1%}')
        aux_rows = []
        for key, label in [("corners", "Córners"), ("fouls", "Faltas"), ("yellow_cards", "Tarjetas amarillas"),
                           ("shots_on_target", "Tiros a puerta"), ("saves", "Paradas estimadas")]:
            item = pred.get("auxiliary", {}).get(key)
            if item and item.get("total") is not None:
                aux_rows.append({"Mercado": label, "Local": round(item["home"], 2),
                                 "Visitante": round(item["away"], 2), "Total": round(item["total"], 2)})
            else:
                aux_rows.append({"Mercado": label, "Local": "No disponible",
                                 "Visitante": "No disponible", "Total": "No disponible"})
        st.dataframe(pd.DataFrame(aux_rows), use_container_width=True, hide_index=True)
        st.caption("Las paradas son una estimación indirecta por equipo, no una estadística oficial de portero.")
    with tab4:
        st.subheader("Comparador de cuotas")
        render_odds(pred, f"{home} vs {away}", profile, cash)
    with tab5:
        st.subheader("Control de riesgo")
        sample = min(int(pred.get("home_matches", 0)), int(pred.get("away_matches", 0)))
        st.metric("Muestra mínima de equipos", sample)
        st.write(f'Perfil activo: **{st.session_state.get("risk_profile", "Agresivo")}**')
        st.write(f'Banca disponible: **{cash:.2f} €**')
        st.caption("El stake solo se calcula si hay valor esperado positivo, ventaja mínima y saldo/exposición disponibles. La muestra automática mínima es de ocho partidos.")
    with tab6:
        show_recent_form(home, away)
        st.info("La forma reciente se muestra como contexto. No modifica todavía las probabilidades ni el stake: primero debe validarse su ponderación mediante backtesting.")
    st.warning("Modelo Poisson básico, no validado como sistema rentable. No incorpora alineaciones, lesiones, árbitro ni información en directo.")

def _automatic_market_odds(fixture, market):
    if market not in AUTO_MARKETS:
        return []
    result = get_match_odds(fixture.get("Competición", ""),
                            fixture.get("Local", ""), fixture.get("Visitante", ""))
    return odds_for_market(result.get("event"), market,
                           fixture.get("Local", ""), fixture.get("Visitante", ""))

def radar(raw, show_filters=True):
    fixtures = st.session_state.get("day_fixtures", [])
    if raw is None or raw.empty or "League" not in raw.columns:
        st.info("El histórico no está disponible para calcular oportunidades.")
        return pd.DataFrame()
    if show_filters:
        st.markdown("#### Filtro de cuotas")
        left, right = st.columns(2)
        min_odds = left.number_input("Cuota mínima", min_value=1.15, max_value=10.0,
                                     value=1.20, step=.05, key="opportunity_min_odds")
        max_odds = right.number_input("Cuota máxima", min_value=1.20, max_value=20.0,
                                      value=2.50, step=.05, key="opportunity_max_odds")
        if max_odds <= min_odds:
            st.error("La cuota máxima debe ser superior a la mínima.")
            return pd.DataFrame()
    else:
        min_odds, max_odds = 1.20, 2.50

    rows, skipped_no_odds, skipped_sample = [], 0, 0
    for fixture in fixtures:
        competition = fixture["Competición"]
        if competition not in LEAGUES or not LEAGUES[competition][1]:
            continue
        league = raw[raw.League == competition].copy()
        home = match_team_name(fixture["Local"], league)
        away = match_team_name(fixture["Visitante"], league)
        if not home or not away or home == away:
            continue
        try:
            pred = predict_match(league, home, away)
        except Exception:
            continue
        sample = min(int(pred.get("home_matches", 0)), int(pred.get("away_matches", 0)))
        if sample < 8:
            skipped_sample += 1
            continue
        choices = markets(pred)
        for market in AUTO_MARKETS:
            probability = choices.get(market)
            if probability is None or pd.isna(probability) or not 0 < float(probability) < 1:
                continue
            odds_rows = _automatic_market_odds(fixture, market)
            if not odds_rows:
                skipped_no_odds += 1
                continue
            best = max(odds_rows, key=lambda item: float(item["Cuota"]))
            odd = float(best["Cuota"])
            if not min_odds <= odd <= max_odds:
                continue
            edge = float(probability) - 1 / odd
            ev = float(probability) * odd - 1
            if ev <= 0 or edge < .01:
                continue
            if odd <= 1.60:
                tier = "Moderado"
            elif odd <= 2.00:
                tier = "Medio"
            else:
                tier = "Alto"
            rows.append({
                "Hora": fixture["Hora"], "Competición": competition,
                "Partido": f"{home} vs {away}", "Selección": market,
                "Probabilidad": float(probability), "Cuota real": odd,
                "Cuota justa": 1 / float(probability), "EV": ev,
                "Ventaja": edge, "Casa": best.get("Casa", "No indicada"),
                "Riesgo orientativo": tier, "Muestra": sample,
            })
    if not rows:
        st.info("No hay selecciones que cumplan simultáneamente el rango de cuotas, cuota automática disponible, EV positivo, ventaja mínima del 1% y muestra de al menos ocho partidos.")
        st.caption(f"Partidos descartados por muestra insuficiente: {skipped_sample}. Mercados sin cuota automática disponible: {skipped_no_odds}.")
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(["EV", "Ventaja", "Muestra"], ascending=False)
    view = frame.copy()
    for col in ["Probabilidad", "EV", "Ventaja"]:
        view[col] = view[col].map(lambda value: f"{value:+.1%}" if col != "Probabilidad" else f"{value:.1%}")
    for col in ["Cuota real", "Cuota justa"]:
        view[col] = view[col].map(lambda value: f"{value:.2f}")
    st.success(f"{len(frame)} oportunidades cumplen los filtros.")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("Solo se incluyen cuotas reales devueltas por The Odds API para 1X2 y totales de goles. Las etiquetas de riesgo son orientativas, no garantías.")
    st.caption(f"Mercados sin cuota automática disponible descartados: {skipped_no_odds}.")
    return frame

def render_combo(raw, profile, cash):
    st.subheader("Constructor de combinadas")
    if raw is None or raw.empty or "League" not in raw.columns:
        st.info("El histórico no está disponible para construir combinadas.")
        return
    st.caption("Una selección por partido, de 2 a 15 selecciones. La probabilidad conjunta supone independencia entre partidos.")
    rows = []
    for fixture in st.session_state.get("day_fixtures", []):
        competition = fixture["Competición"]
        if competition not in LEAGUES or not LEAGUES[competition][1]:
            continue
        league = raw[raw.League == competition].copy()
        home = match_team_name(fixture["Local"], league)
        away = match_team_name(fixture["Visitante"], league)
        if not home or not away or home == away:
            continue
        try:
            pred = predict_match(league, home, away)
        except Exception:
            continue
        sample = min(int(pred.get("home_matches", 0)), int(pred.get("away_matches", 0)))
        if sample < 8:
            continue
        for label, probability in markets(pred).items():
            if probability is not None and not pd.isna(probability) and .54 <= float(probability) <= .88:
                rows.append({"Hora": fixture["Hora"], "Partido": f"{home} vs {away}",
                             "Mercado": label, "Probabilidad": float(probability),
                             "Cuota justa": 1/float(probability), "Muestra": sample,
                             "rank": float(probability)*.75 + min(sample, 20)/20*.25})
    if not rows:
        st.info("No hay suficientes selecciones con muestra mínima de ocho partidos.")
        return
    frame = pd.DataFrame(rows).sort_values(["rank", "Muestra"], ascending=False).drop_duplicates("Partido")
    maximum = min(15, len(frame))
    count = st.slider("Número de selecciones", 2, max(2, maximum), min(4, max(2, maximum)))
    st.dataframe(frame[["Hora", "Partido", "Mercado", "Probabilidad", "Cuota justa", "Muestra"]],
                 use_container_width=True, hide_index=True)
    picks = frame.head(count)
    probability_combo = float(picks.Probabilidad.prod())
    fair = float(picks["Cuota justa"].prod())
    st.markdown("**Combinada propuesta**")
    st.dataframe(picks[["Partido", "Mercado", "Probabilidad", "Cuota justa"]],
                 use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    a.metric("Probabilidad conjunta aproximada", f"{probability_combo:.3%}")
    b.metric("Cuota justa estimada", f"{fair:.2f}")
    actual = st.number_input("Cuota total real", 1.01, 100000.0,
                             float(max(1.01, round(fair, 2))), .05, key="combo_odds")
    edge = probability_combo - 1/actual
    amount, _ = stake_advice(probability_combo, actual, edge, cash, profile, True)
    a, b, c = st.columns(3)
    a.metric("Probabilidad implícita", f"{1/actual:.3%}")
    b.metric("Ventaja estimada", f"{edge:+.2%}")
    c.metric("Stake orientativo", f"{amount:.2f} €")
    st.caption("No se registran combinadas en el historial para evitar liquidarlas incorrectamente como apuestas simples.")

def render_tracker():
    log = st.session_state.setdefault("bet_log", [])
    if log:
        profit, pending, _ = stats_log()
        initial = float(st.session_state.get("bankroll_initial", 100))
        a, b, c = st.columns(3)
        a.metric("Banca inicial", f"{initial:.2f} €")
        b.metric("Saldo disponible", f"{bankroll():.2f} €")
        c.metric("Beneficio liquidado", f"{profit:+.2f} €")
        frame = pd.DataFrame(log)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button("Descargar historial CSV", frame.to_csv(index=False).encode("utf-8-sig"),
                           "football_ai_historial_apuestas.csv", "text/csv")
        pending_ids = [i for i, item in enumerate(log) if item.get("Resultado") == "Pendiente"]
        if pending_ids:
            idx = st.selectbox("Apuesta pendiente", pending_ids,
                format_func=lambda i: f'{log[i].get("Partido")} · {log[i].get("Mercado")} · {log[i].get("Stake")} €')
            status = st.selectbox("Resultado", ["Ganada", "Perdida", "Anulada"])
            if st.button("Guardar resultado"):
                item = log[idx]
                stake, odds = float(item.get("Stake", 0)), float(item.get("Cuota Bet365", 0))
                item["Resultado"] = status
                item["Beneficio neto"] = round(stake*(odds-1), 2) if status == "Ganada" else (-stake if status == "Perdida" else 0)
                st.rerun()
    else:
        st.info("Todavía no has registrado apuestas.")
    uploaded = st.file_uploader("Importar historial CSV anterior", type=["csv"])
    if uploaded is not None and st.button("Importar historial"):
        try:
            frame = pd.read_csv(uploaded).fillna("")
            if not {"Partido", "Mercado", "Cuota Bet365", "Resultado"}.issubset(frame.columns):
                st.error("El CSV no contiene las columnas necesarias.")
            else:
                st.session_state.bet_log = frame.to_dict("records")
                st.success("Historial importado.")
                st.rerun()
        except Exception as exc:
            st.error(f"No se pudo importar: {exc}")

def page_home(raw, errors, profile, cash):
    st.title("⚽ Football AI Predictor")
    st.caption("Calendario de partidos · Horario peninsular español")
    show_calendar()

def page_team():
    st.title("Análisis por equipo")
    st.caption("Últimos cinco partidos oficiales disponibles entre las competiciones indexadas. Puede haber menos de cinco si el equipo aún no los ha disputado.")
    fixture = st.session_state.get("selected_fixture", {})
    try:
        with st.spinner("Cargando resultados oficiales..."):
            frame = team_form.load_official_results()
        if frame.empty:
            st.warning("La fuente de resultados no está disponible ahora mismo.")
            return
        names = sorted(set(frame["home"].dropna().astype(str)) | set(frame["away"].dropna().astype(str)))
        default_name = fixture.get("Local") if fixture.get("Local") in names else names[0]
        team = st.selectbox("Equipo", names, index=names.index(default_name), key="team_form_select")
        recent = team_form.last_five(team, frame)
        if recent.empty:
            st.info("No se encontraron coincidencias seguras para este equipo.")
            return
        summary = team_form.form_summary(recent)
        a, b, c, d = st.columns(4)
        a.metric("Partidos", summary["played"])
        b.metric("Puntos", f'{summary["points"]}/{summary["played"]*3}')
        c.metric("Victorias", summary["wins"])
        d.metric("Media goles a favor", f'{summary["goals_for"]:.2f}')
        tab1, tab2 = st.tabs(["Últimos partidos", "Estadísticas y comparación"])
        with tab1:
            st.dataframe(recent, use_container_width=True, hide_index=True)
        with tab2:
            rows = []
            for stat in ["Córners", "Tiros a puerta", "Faltas", "Amarillas", "Rojas"]:
                average, count = team_form.stat_average(recent, stat)
                rows.append({"Estadística": stat, "Media": f"{average:.2f}" if average is not None else "No disponible",
                             "Cobertura": f"{count}/{len(recent)} partidos"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            other = fixture.get("Visitante") if team == fixture.get("Local") else fixture.get("Local")
            if other and other in names:
                other_recent = team_form.last_five(other, frame)
                if not other_recent.empty:
                    other_summary = team_form.form_summary(other_recent)
                    st.markdown(f"**Comparación: {team} vs {other}**")
                    x, y = st.columns(2)
                    x.metric(f"{team} · puntos", f'{summary["points"]}/{summary["played"]*3}')
                    y.metric(f"{other} · puntos", f'{other_summary["points"]}/{other_summary["played"]*3}')
                    st.dataframe(other_recent, use_container_width=True, hide_index=True)
        st.caption("Fuente: open-matchday. Actualización diaria, no en directo. Los campos vacíos indican que la estadística no fue publicada.")
    except Exception as exc:
        st.error(f"No se pudo cargar el análisis de equipo: {type(exc).__name__}: {exc}")

def page_bank():
    st.title("Mi banca")
    st.caption("Control de saldo, perfil de riesgo y exposición.")
    profiles = list(RISKS)
    current = st.session_state.get("risk_profile", "Agresivo")
    if current not in profiles:
        current = "Agresivo"
    st.selectbox("Perfil de riesgo", profiles, index=profiles.index(current),
                 key="risk_profile")
    st.number_input("Banca inicial (€)", min_value=1.0, max_value=1000000.0,
                    value=float(st.session_state.get("bankroll_initial", 100.0)), step=10.0, key="bankroll_initial")
    profile = current_profile()
    profit, pending, settled = stats_log()
    cash = bankroll()
    a, b, c, d = st.columns(4)
    a.metric("Saldo disponible", f"{cash:.2f} €")
    b.metric("Beneficio liquidado", f"{profit:+.2f} €")
    c.metric("Exposición pendiente", f"{pending:.2f} €")
    d.metric("Límite de exposición", f'{float(st.session_state.bankroll_initial)*profile["exposure"]:.2f} €')
    st.write(f'Tope por apuesta simple: **{profile["max"]:.0%}** · Tope combinadas: **{profile["combo"]:.1%}** · Exposición máxima: **{profile["exposure"]:.0%}**')
    st.markdown("### Seguimiento de apuestas")
    render_tracker()

def page_results():
    st.title("Resultados y estadísticas")
    log = st.session_state.get("bet_log", [])
    wins = sum(1 for item in log if item.get("Resultado") == "Ganada")
    losses = sum(1 for item in log if item.get("Resultado") == "Perdida")
    decided = wins + losses
    pending_count = sum(1 for item in log if item.get("Resultado", "Pendiente") == "Pendiente")
    a, b, c, d = st.columns(4)
    a.metric("Apuestas registradas", len(log))
    b.metric("Apuestas decididas", decided)
    c.metric("Acierto", f"{wins/decided:.1%}" if decided else "—")
    d.metric("Pendientes", pending_count)
    st.caption("El control de saldo, beneficios, exposición, importación y liquidación económica de apuestas está centralizado en «Mi banca».")
    st.markdown("### Estado del modelo")
    st.info("Modelo Poisson básico. Todavía no existe validación prospectiva suficiente para afirmar que sea rentable. La forma reciente se muestra como contexto hasta calibrar su peso mediante backtesting.")

def main():
    st.session_state.setdefault("risk_profile", "Agresivo")
    st.session_state.setdefault("bankroll_initial", 100.0)
    st.session_state.setdefault("bet_log", [])
    st.session_state.setdefault("navigation", "Inicio")
    with st.sidebar:
        st.title("⚽ Football AI")
        page = st.radio("Navegación", NAV, key="navigation")
        st.divider()
        st.caption("Calendario: API Ninjas")
        st.caption("Cuotas: The Odds API")
        st.caption("Histórico: Football-Data.co.uk")
        st.caption("Forma reciente: open-matchday")
        if st.button("Actualizar datos históricos"):
            historical.clear()
            team_form.load_official_results.clear()
            fixtures_for.clear()
            st.rerun()
    profile = current_profile()
    cash = bankroll()
    raw, errors = historical()
    if page == "Inicio":
        page_home(raw, errors, profile, cash)
    elif page == "Partidos":
        st.title("Partidos")
        show_calendar()
    elif page == "Análisis IA":
        render_analysis(raw, profile, cash)
    elif page == "Análisis por equipo":
        page_team()
    elif page == "Oportunidades":
        st.title("Oportunidades")
        st.caption("Selecciones con cuotas reales, valor esperado positivo y muestra histórica suficiente.")
        radar(raw, show_filters=True)
        st.divider()
        render_combo(raw, profile, cash)
    elif page == "Mi banca":
        page_bank()
    elif page == "Resultados y estadísticas":
        page_results()
    if not raw.empty:
        with st.expander("Información y cobertura del histórico"):
            st.write(f"Football-Data.co.uk: {len(raw):,} resultados en {raw.League.nunique()} competiciones de liga.")
            for error in errors[:25]:
                st.write(error)

def render_odds(pred, name, profile, cash):
    """Fallback manual; app.py lo sustituye por el comparador automático."""
    st.caption("Introduce una cuota decimal para comparar con el modelo.")
    choices = markets(pred)
    market = st.selectbox("Mercado", list(choices), key="odds_market")
    probability = choices[market]
    if probability is None or pd.isna(probability):
        st.warning("No hay probabilidad válida para este mercado.")
        return
    odd = st.number_input("Cuota decimal", 1.01, 1000.0, 2.0, .05, key="odds_decimal")
    edge = float(probability) - 1/float(odd)
    a, b, c = st.columns(3)
    a.metric("Probabilidad", f"{probability:.1%}")
    b.metric("Cuota justa", f"{1/float(probability):.2f}")
    c.metric("Valor esperado", f"{float(probability)*float(odd)-1:+.1%}")
