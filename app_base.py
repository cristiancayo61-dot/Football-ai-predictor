import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO

from predictor import predict_match, match_team_name, poisson_over
import team_form

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
    "Conservador": {"normal": .01, "medium": .015, "high": .02, "max": .02, "combo": .005, "exposure": .03},
    "Equilibrado": {"normal": .02, "medium": .025, "high": .03, "max": .03, "combo": .015, "exposure": .05},
    "Agresivo": {"normal": .03, "medium": .04, "high": .05, "max": .05, "combo": .02, "exposure": .08},
}
NAV = ["Inicio", "Partidos", "Análisis IA", "Análisis por equipo", "Oportunidades", "Mi banca", "Resultados y estadísticas"]

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
    for label, (_, code) in LEAGUES.items():
        if not code:
            continue
        for season in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                r = requests.get(url, timeout=20, headers={"User-Agent": "Football-AI-Predictor/1.0"})
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                df = pd.read_csv(BytesIO(r.content), encoding="latin-1")
                df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
                if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(df.columns):
                    errors.append(f"{label} {season}: columnas básicas ausentes")
                    continue
                df["League"], df["Season"] = label, season
                frames.append(df)
            except Exception as e:
                errors.append(f"{label} {season}: {type(e).__name__}")
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), errors

def stats_log():
    log = st.session_state.setdefault("bet_log", [])
    profit = pending = settled = 0.0
    for x in log:
        stake = float(x.get("Stake", 0) or 0)
        odds = float(x.get("Cuota Bet365", 0) or 0)
        status = x.get("Resultado", "Pendiente")
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
    name = st.session_state.get("risk_profile", "Equilibrado")
    return RISKS.get(name, RISKS["Equilibrado"])

def markets(pred):
    result = {
        "Victoria local": pred.get("home_win"), "Empate": pred.get("draw"),
        "Victoria visitante": pred.get("away_win"), "Doble oportunidad 1X": pred.get("double_chance_1x"),
        "Doble oportunidad X2": pred.get("double_chance_x2"), "Doble oportunidad 12": pred.get("double_chance_12"),
        "Más de 1,5 goles": pred.get("over15"), "Más de 2,5 goles": pred.get("over25"),
        "Más de 3,5 goles": pred.get("over35"), "Ambos equipos marcan - Sí": pred.get("btts"),
    }
    defs = {
        "corners": ("córners totales", [7.5, 8.5, 9.5, 10.5]),
        "fouls": ("faltas totales", [19.5, 21.5, 23.5, 25.5]),
        "yellow_cards": ("tarjetas amarillas totales", [2.5, 3.5, 4.5, 5.5]),
        "saves": ("paradas estimadas", [3.5, 4.5, 5.5, 6.5]),
        "shots_on_target": ("tiros a puerta totales", [7.5, 8.5, 9.5, 10.5]),
    }
    for k, (label, lines) in defs.items():
        item = pred.get("auxiliary", {}).get(k)
        if item and item.get("total") is not None:
            for line in lines:
                p = poisson_over(item["total"], line)
                if p is not None:
                    l = str(line).replace(".", ",")
                    result[f"Más de {l} {label}"] = p
                    result[f"Menos de {l} {label}"] = 1 - p
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
    d = today()
    if "calendar_date" not in st.session_state:
        st.session_state.calendar_date = d
    cols = st.columns(4)
    quick = [("Hoy", d), ("Mañana", d + timedelta(days=1)),
             ("Sábado", d + timedelta(days=(5-d.weekday()) % 7)),
             ("Domingo", d + timedelta(days=(6-d.weekday()) % 7))]
    for col, (label, day) in zip(cols, quick):
        if col.button(label, use_container_width=True):
            st.session_state.calendar_date = day
            st.rerun()
    chosen = st.date_input("Fecha", key="calendar_date", format="DD/MM/YYYY")
    comps = [x for x, y in LEAGUES.items() if y[1]]
    selected = st.multiselect("Competiciones", comps, default=comps, key="calendar_competitions")
    key = api_key()
    if not key:
        st.error("No encuentro API_NINJAS_KEY en los Secrets de Streamlit.")
        return
    matches, errors = [], []
    with st.spinner("Consultando calendario de API Ninjas..."):
        for comp in selected:
            try:
                for item in fixtures_for(LEAGUES[comp][0], chosen.isoformat(), key):
                    if not isinstance(item, dict):
                        continue
                    dt = parse_dt(item.get("start_time") or item.get("date"))
                    hs, aws = item.get("home_score"), item.get("away_score")
                    matches.append({
                        "Hora": dt.strftime("%H:%M") if dt else "Pendiente",
                        "Competición": comp,
                        "Local": team_name(item.get("home_team"), "Local"),
                        "Visitante": team_name(item.get("away_team"), "Visitante"),
                        "Estado": item.get("status", "Sin estado"),
                        "Resultado": f"{hs} - {aws}" if hs is not None and aws is not None else "—",
                    })
            except Exception as e:
                errors.append(f"{comp}: {e}")
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
    chosen_label = st.selectbox("Selecciona un partido", labels, key="fixture_choice")
    if st.button("Abrir ficha de análisis", type="primary"):
        st.session_state.selected_fixture = matches[labels.index(chosen_label)]
        st.success("Partido guardado. Abre ahora la sección «Análisis IA» en el menú lateral.")

def selected_teams(raw, fixture):
    if raw is None or raw.empty or "League" not in raw.columns:
        return None, None, None
    comp = fixture.get("Competición", "")
    if comp not in LEAGUES or not LEAGUES[comp][1]:
        return None, None, None
    league = raw[raw.League == comp].copy()
    home = match_team_name(fixture.get("Local", ""), league)
    away = match_team_name(fixture.get("Visitante", ""), league)
    if not home or not away or home == away:
        return league, None, None
    return league, home, away

def show_recent_form(home, away):
    st.markdown("### Forma reciente: últimos cinco partidos oficiales")
    st.caption("Incluye liga, copas y competiciones europeas dentro de las 75 competiciones disponibles. La fuente cubre la temporada 2026/27; si un equipo aún no acumula cinco encuentros, se mostrarán los disponibles.")
    try:
        all_results = team_form.load_official_results()
        if all_results.empty:
            st.warning("La fuente de resultados recientes no devolvió partidos.")
            return
        left, right = st.columns(2)
        for col, team in [(left, home), (right, away)]:
            with col:
                st.markdown(f"**{team}**")
                recent = team_form.last_five(team, all_results)
                if recent.empty:
                    st.info("No se han encontrado coincidencias seguras para este equipo.")
                    continue
                summary = team_form.form_summary(recent)
                a, b, c = st.columns(3)
                a.metric("Puntos", f'{summary["points"]}/{summary["played"]*3}')
                b.metric("Victorias", summary["wins"])
                c.metric("Goles", f'{summary["goals_for"]:.1f} a favor')
                st.dataframe(recent, use_container_width=True, hide_index=True)
                stats = []
                for stat in ["Córners", "Tiros a puerta", "Faltas", "Amarillas"]:
                    avg, n = team_form.stat_average(recent, stat)
                    stats.append({"Estadística": stat, "Media": f"{avg:.1f}" if avg is not None else "No disponible",
                                  "Partidos con dato": f"{n}/{len(recent)}"})
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)
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
        st.warning("No se ha podido cargar el histórico de Football-Data.co.uk. El análisis estadístico no está disponible ahora mismo.")
        return
    league, home, away = selected_teams(raw, fixture)
    if league is None:
        st.warning("Este torneo está en el calendario, pero no tiene histórico de liga integrado en el modelo actual.")
        return
    if not home or not away:
        st.warning("No he encontrado una coincidencia segura de nombres en el histórico.")
        teams = sorted(set(league.HomeTeam).union(set(league.AwayTeam)))
        c1, c2 = st.columns(2)
        c1.selectbox("Equipo local", teams, key="manual_home")
        c2.selectbox("Equipo visitante", teams, key="manual_away")
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
    st.subheader("Probabilidades principales")
    a, b, c = st.columns(3)
    a.metric("Victoria local", f'{pred["home_win"]:.1%}')
    b.metric("Empate", f'{pred["draw"]:.1%}')
    c.metric("Victoria visitante", f'{pred["away_win"]:.1%}')
    a, b, c, d = st.columns(4)
    a.metric("Más de 1,5 goles", f'{pred["over15"]:.1%}')
    b.metric("Más de 2,5 goles", f'{pred["over25"]:.1%}')
    c.metric("Más de 3,5 goles", f'{pred["over35"]:.1%}')
    d.metric("Ambos marcan", f'{pred["btts"]:.1%}')
    a, b, c = st.columns(3)
    a.metric("Doble oportunidad 1X", f'{pred["double_chance_1x"]:.1%}')
    b.metric("Doble oportunidad X2", f'{pred["double_chance_x2"]:.1%}')
    c.metric("Doble oportunidad 12", f'{pred["double_chance_12"]:.1%}')
    st.write(f'Goles esperados: **{home} {pred["lambda_home"]:.2f}** · **{away} {pred["lambda_away"]:.2f}**')
    st.markdown("### Mercados estadísticos complementarios")
    for k, label in [("corners", "Córners"), ("fouls", "Faltas"), ("yellow_cards", "Tarjetas amarillas"),
                     ("shots_on_target", "Tiros a puerta"), ("saves", "Paradas estimadas")]:
        item = pred.get("auxiliary", {}).get(k)
        if item and item.get("total") is not None:
            st.write(f'**{label}:** {home} {item["home"]:.1f} · {away} {item["away"]:.1f} · Total {item["total"]:.1f}')
        else:
            st.caption(f"{label}: datos insuficientes.")
    scores = pd.DataFrame(pred["top_scores"], columns=["Marcador", "Probabilidad"])
    scores["Probabilidad"] = scores["Probabilidad"].map(lambda x: f"{x:.1%}")
    st.markdown("### Marcadores exactos más probables")
    st.dataframe(scores, use_container_width=True, hide_index=True)
    st.caption(f'Muestra histórica: {pred.get("home_matches",0)} partidos del local en casa y {pred.get("away_matches",0)} del visitante fuera.')
    show_recent_form(home, away)
    st.info("La forma de los últimos cinco partidos se incorpora como contexto de análisis. No modifica todavía las probabilidades ni el stake: primero debe validarse su ponderación con pruebas históricas para evitar introducir un sesgo no medido.")
    st.markdown("### Comparador de cuotas")
    render_odds(pred, f"{home} vs {away}", profile, cash)
    st.warning("Modelo Poisson básico, no validado como sistema rentable. Los mercados auxiliares son estimaciones históricas de equipos, no datos en directo. No incorpora alineaciones, lesiones, árbitro ni cuotas en tiempo real.")

def radar(raw):
    fixtures = st.session_state.get("day_fixtures", [])
    if raw is None or raw.empty or "League" not in raw.columns:
        st.info("El histórico no está disponible para calcular el radar.")
        return pd.DataFrame()
    rows = []
    for f in fixtures:
        comp = f["Competición"]
        if comp not in LEAGUES or not LEAGUES[comp][1]:
            continue
        league = raw[raw.League == comp].copy()
        h, a = match_team_name(f["Local"], league), match_team_name(f["Visitante"], league)
        if not h or not a or h == a:
            continue
        try:
            pred = predict_match(league, h, a)
        except Exception:
            continue
        for label, p in [("Victoria local", pred["home_win"]), ("Empate", pred["draw"]),
                         ("Victoria visitante", pred["away_win"]), ("Doble oportunidad 1X", pred["double_chance_1x"]),
                         ("Doble oportunidad X2", pred["double_chance_x2"]), ("Más de 1,5 goles", pred["over15"]),
                         ("Más de 2,5 goles", pred["over25"]), ("Ambos marcan", pred["btts"])]:
            rows.append({"Hora": f["Hora"], "Competición": comp, "Partido": f'{h} vs {a}',
                         "Selección": label, "Probabilidad": p, "Cuota justa": 1/p if p > 0 else None})
    if not rows:
        st.info("Carga primero los partidos del día para generar oportunidades.")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Probabilidad", ascending=False)
    view = df.copy()
    view["Probabilidad"] = view["Probabilidad"].map(lambda x: f"{x:.1%}")
    view["Cuota justa"] = view["Cuota justa"].map(lambda x: f"{x:.2f}")
    st.dataframe(view, use_container_width=True, hide_index=True)
    return df

def render_combo(raw, profile, cash):
    st.subheader("Constructor de combinadas")
    if raw is None or raw.empty or "League" not in raw.columns:
        st.info("El histórico no está disponible para construir combinadas.")
        return
    st.caption("Una selección por partido, de 2 a 15 selecciones. La probabilidad conjunta supone independencia entre partidos.")
    rows = []
    for f in st.session_state.get("day_fixtures", []):
        comp = f["Competición"]
        if comp not in LEAGUES or not LEAGUES[comp][1]:
            continue
        league = raw[raw.League == comp].copy()
        h, a = match_team_name(f["Local"], league), match_team_name(f["Visitante"], league)
        if not h or not a or h == a:
            continue
        try:
            pred = predict_match(league, h, a)
        except Exception:
            continue
        n = min(int(pred.get("home_matches", 0)), int(pred.get("away_matches", 0)))
        if n < 8:
            continue
        for label, p in markets(pred).items():
            if p is not None and not pd.isna(p) and .54 <= float(p) <= .88:
                rows.append({"Hora": f["Hora"], "Partido": f"{h} vs {a}", "Mercado": label,
                             "Probabilidad": float(p), "Cuota justa": 1/float(p), "Muestra": n,
                             "rank": float(p)*.75 + min(n, 20)/20*.25})
    if not rows:
        st.info("No hay suficientes selecciones con muestra mínima de ocho partidos.")
        return
    df = pd.DataFrame(rows).sort_values(["rank", "Muestra"], ascending=False).drop_duplicates("Partido")
    maximum = min(15, len(df))
    count = st.slider("Número de selecciones", 2, max(2, maximum), min(4, max(2, maximum)))
    st.dataframe(df[["Hora", "Partido", "Mercado", "Probabilidad", "Cuota justa", "Muestra"]],
                 use_container_width=True, hide_index=True)
    picks = df.head(count)
    pcombo = float(picks.Probabilidad.prod())
    fair = float(picks["Cuota justa"].prod())
    st.markdown("**Combinada propuesta**")
    st.dataframe(picks[["Partido", "Mercado", "Probabilidad", "Cuota justa"]], use_container_width=True, hide_index=True)
    a, b = st.columns(2)
    a.metric("Probabilidad conjunta aproximada", f"{pcombo:.3%}")
    b.metric("Cuota justa estimada", f"{fair:.2f}")
    actual = st.number_input("Cuota total real", 1.01, 100000.0, float(max(1.01, round(fair, 2))), .05, key="combo_odds")
    edge = pcombo - 1/actual
    amount, _ = stake_advice(pcombo, actual, edge, cash, profile, True)
    a, b, c = st.columns(3)
    a.metric("Probabilidad implícita", f"{1/actual:.3%}")
    b.metric("Ventaja estimada", f"{edge:+.2%}")
    c.metric("Stake orientativo", f"{amount:.2f} €")
    st.caption("No se registran combinadas en el historial para evitar liquidarlas incorrectamente como apuestas simples.")

def render_tracker():
    log = st.session_state.setdefault("bet_log", [])
    if log:
        profit, pending, settled = stats_log()
        initial = float(st.session_state.get("bankroll_initial", 100))
        c1, c2, c3 = st.columns(3)
        c1.metric("Banca inicial", f"{initial:.2f} €")
        c2.metric("Saldo disponible", f"{bankroll():.2f} €")
        c3.metric("Beneficio liquidado", f"{profit:+.2f} €")
        df = pd.DataFrame(log)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Descargar historial CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "football_ai_historial_apuestas.csv", "text/csv")
        pending_ids = [i for i, x in enumerate(log) if x.get("Resultado") == "Pendiente"]
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
            df = pd.read_csv(uploaded).fillna("")
            if not {"Partido", "Mercado", "Cuota Bet365", "Resultado"}.issubset(df.columns):
                st.error("El CSV no contiene las columnas necesarias.")
            else:
                st.session_state.bet_log = df.to_dict("records")
                st.success("Historial importado.")
                st.rerun()
        except Exception as exc:
            st.error(f"No se pudo importar: {exc}")

def page_home(raw, errors, profile, cash):
    st.title("⚽ Football AI Predictor")
    st.caption("Panel principal · Horario peninsular español")
    c1, c2, c3 = st.columns(3)
    c1.metric("Banca disponible", f"{cash:.2f} €")
    c2.metric("Perfil de riesgo", st.session_state.get("risk_profile", "Equilibrado"))
    c3.metric("Partidos cargados", len(st.session_state.get("day_fixtures", [])))
    selected = st.session_state.get("selected_fixture")
    if selected:
        st.markdown("### Partido seleccionado")
        st.write(f'**{selected.get("Local")} vs {selected.get("Visitante")}** · {selected.get("Competición")} · {selected.get("Hora")}')
        st.caption("Para abrir la ficha completa, entra en «Análisis IA» desde el menú lateral.")
    else:
        st.info("Empieza consultando los partidos en la sección Partidos.")
    st.markdown("### Resumen de actividad")
    log = st.session_state.get("bet_log", [])
    profit, pending, settled = stats_log()
    a, b, c = st.columns(3)
    a.metric("Apuestas registradas", len(log))
    b.metric("Exposición pendiente", f"{pending:.2f} €")
    a2 = float(st.session_state.get("bankroll_initial", 100))
    c.metric("Rentabilidad liquidada", f"{(profit/settled):+.1%}" if settled else "—")
    if st.session_state.get("day_fixtures"):
        st.markdown("### Oportunidades destacadas")
        with st.expander("Ver radar de probabilidades de la jornada"):
            radar(raw)
    profit, pending, _ = stats_log()
    exposure_limit = float(st.session_state.get("bankroll_initial", 100.0)) * profile["exposure"]
    if pending >= exposure_limit and pending > 0:
        st.warning("La exposición pendiente ha alcanzado el límite del perfil de riesgo.")
    if cash <= 0:
        st.warning("La banca disponible está agotada. No se debe recomendar nuevo stake.")
    if errors:
        st.caption(f"Incidencias al cargar histórico: {len(errors)}. Consulta el detalle en Resultados y estadísticas.")

def page_team():
    st.title("Análisis por equipo")
    st.caption("Últimos cinco partidos oficiales disponibles entre las 75 competiciones indexadas. Datos de la temporada 2026/27; al inicio de curso puede haber menos de cinco.")
    fixture = st.session_state.get("selected_fixture", {})
    try:
        with st.spinner("Descargando resultados oficiales de las competiciones..."):
            df = team_form.load_official_results()
        if df.empty:
            st.warning("La fuente de resultados no está disponible ahora mismo.")
            return
        names = sorted(set(df["home"].dropna().astype(str)) | set(df["away"].dropna().astype(str)))
        default_name = fixture.get("Local") if fixture.get("Local") in names else names[0]
        team = st.selectbox("Equipo", names, index=names.index(default_name), key="team_form_select")
        recent = team_form.last_five(team, df)
        if recent.empty:
            st.info("No se encontraron cinco partidos coincidentes. Comprueba el nombre del equipo.")
            return
        summary = team_form.form_summary(recent)
        a, b, c, d = st.columns(4)
        a.metric("Partidos", summary["played"])
        b.metric("Puntos", f'{summary["points"]}/{summary["played"]*3}')
        c.metric("Victorias", summary["wins"])
        d.metric("Media goles a favor", f'{summary["goals_for"]:.2f}')
        st.dataframe(recent, use_container_width=True, hide_index=True)
        st.markdown("### Medias estadísticas y cobertura")
        rows = []
        for stat in ["Córners", "Tiros a puerta", "Faltas", "Amarillas", "Rojas"]:
            avg, count = team_form.stat_average(recent, stat)
            rows.append({"Estadística": stat, "Media": f"{avg:.2f}" if avg is not None else "No disponible",
                         "Cobertura": f"{count}/{len(recent)} partidos"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Fuente: open-matchday. Datos diarios, no en directo. Los campos vacíos indican que la estadística no fue publicada.")
        if fixture.get("Local") and fixture.get("Visitante"):
            st.markdown("### Comparación del partido seleccionado")
            other = fixture.get("Visitante") if team == fixture.get("Local") else fixture.get("Local")
            other_matches = team_form.last_five(other, df)
            if not other_matches.empty:
                st.write(f"Comparación con **{other}**")
                x, y = st.columns(2)
                x.metric(f"{team} · puntos", f'{summary["points"]}/{summary["played"]*3}')
                os = team_form.form_summary(other_matches)
                y.metric(f"{other} · puntos", f'{os["points"]}/{os["played"]*3}')
                st.dataframe(other_matches, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"No se pudo cargar el análisis de equipo: {type(exc).__name__}: {exc}")

def page_bank(profile):
    st.title("Mi banca")
    st.caption("Control de saldo, perfil de riesgo y exposición.")
    st.selectbox("Perfil de riesgo", list(RISKS), index=list(RISKS).index(st.session_state.get("risk_profile", "Equilibrado")),
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
    profit, pending, settled = stats_log()
    initial = float(st.session_state.get("bankroll_initial", 100.0))
    wins = sum(1 for x in log if x.get("Resultado") == "Ganada")
    losses = sum(1 for x in log if x.get("Resultado") == "Perdida")
    decided = wins + losses
    a, b, c, d = st.columns(4)
    a.metric("Apuestas decididas", decided)
    b.metric("Acierto", f"{wins/decided:.1%}" if decided else "—")
    c.metric("Beneficio neto", f"{profit:+.2f} €")
    d.metric("ROI", f"{profit/settled:+.1%}" if settled else "—")
    st.caption("El ROI se calcula sobre el importe apostado en apuestas liquidadas. Las anuladas no se incluyen.")
    render_tracker()
    st.markdown("### Evolución de banca")
    if log:
        rows = [{"Fecha": x.get("Fecha registro", ""), "Resultado": x.get("Resultado", ""),
                 "Beneficio": float(x.get("Beneficio neto", 0) or 0)} for x in log]
        df = pd.DataFrame(rows)
        df["Beneficio acumulado"] = df["Beneficio"].cumsum()
        st.line_chart(df.set_index("Fecha")["Beneficio acumulado"])
    st.markdown("### Estado del modelo")
    st.info("Modelo Poisson básico. Todavía no existe una validación prospectiva suficiente para afirmar que sea rentable. Las señales de forma reciente se muestran como contexto hasta calibrar su peso mediante backtesting.")

def main():
    st.session_state.setdefault("risk_profile", "Equilibrado")
    st.session_state.setdefault("bankroll_initial", 100.0)
    st.session_state.setdefault("bet_log", [])
    st.session_state.setdefault("navigation", "Inicio")
    with st.sidebar:
        st.title("⚽ Football AI")
        page = st.radio("Navegación", NAV, key="navigation")
        st.divider()
        st.caption("Datos de calendario: API Ninjas")
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
        st.caption("Radar estadístico y constructor de combinadas.")
        st.markdown("### Radar de la jornada")
        radar(raw)
        st.divider()
        render_combo(raw, profile, cash)
    elif page == "Mi banca":
        page_bank(profile)
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
    p = choices[market]
    if p is None or pd.isna(p):
        st.warning("No hay probabilidad válida para este mercado.")
        return
    odd = st.number_input("Cuota decimal", 1.01, 1000.0, 2.0, .05, key="odds_decimal")
    edge = float(p) - 1/float(odd)
    a, b, c = st.columns(3)
    a.metric("Probabilidad", f"{p:.1%}")
    b.metric("Cuota justa", f"{1/float(p):.2f}")
    c.metric("Valor esperado", f"{float(p)*float(odd)-1:+.1%}")
