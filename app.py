import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO
from predictor import predict_match, compare_odds, match_team_name, poisson_over

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

def today():
    return datetime.now(TZ).date()

def api_key():
    try: return st.secrets.get("API_NINJAS_KEY", "")
    except Exception: return ""

def team_name(v, fallback):
    if isinstance(v, dict): return str(v.get("name") or fallback)
    return v.strip() if isinstance(v, str) and v.strip() else fallback

def parse_dt(v):
    if not v: return None
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if d.tzinfo is None: d = d.replace(tzinfo=ZoneInfo("UTC"))
        return d.astimezone(TZ)
    except (ValueError, TypeError): return None

@st.cache_data(ttl=900, show_spinner=False)
def fixtures_for(league, day, key):
    r = requests.get(API_URL, params={"league": league, "date": day},
                     headers={"X-Api-Key": key}, timeout=25)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list): raise ValueError("Formato inesperado de API Ninjas")
    return data

@st.cache_data(ttl=21600, show_spinner=False)
def historical():
    frames, errors = [], []
    for label, (api, code) in LEAGUES.items():
        if not code: continue
        for season in SEASONS:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                r = requests.get(url, timeout=20, headers={"User-Agent":"Football-AI-Predictor/1.0"})
                if r.status_code == 404: continue
                r.raise_for_status()
                df = pd.read_csv(BytesIO(r.content), encoding="latin-1")
                df.columns = [str(c).replace("\ufeff","").strip() for c in df.columns]
                if not {"HomeTeam","AwayTeam","FTHG","FTAG"}.issubset(df.columns):
                    errors.append(f"{label} {season}: columnas básicas ausentes"); continue
                df["League"], df["Season"] = label, season
                frames.append(df)
            except Exception as e: errors.append(f"{label} {season}: {type(e).__name__}")
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), errors

def stats_log():
    log = st.session_state.setdefault("bet_log", [])
    profit = pending = settled = 0.0
    for x in log:
        stake, odds = float(x.get("Stake",0) or 0), float(x.get("Cuota Bet365",0) or 0)
        status = x.get("Resultado","Pendiente")
        if status == "Pendiente": pending += stake
        elif status == "Ganada": profit += stake*(odds-1); settled += stake
        elif status == "Perdida": profit -= stake; settled += stake
    return profit, pending, settled

def bankroll():
    profit, pending, _ = stats_log()
    return max(0.0, float(st.session_state.get("bankroll_initial",100.0))+profit-pending)

def risk_setup():
    st.sidebar.header("Gestión de riesgo")
    st.sidebar.selectbox("Perfil de riesgo", list(RISKS), index=1, key="risk_profile")
    st.sidebar.number_input("Banca inicial (€)", 1.0, 1000000.0,
                            value=float(st.session_state.get("bankroll_initial",100.0)),
                            step=10.0, key="bankroll_initial")
    profile = RISKS[st.session_state["risk_profile"]]
    available = bankroll()
    st.sidebar.metric("Saldo disponible estimado", f"{available:.2f} €")
    st.sidebar.caption(f"Tope por apuesta: {profile['max']:.0%} · Exposición total orientativa: {profile['exposure']:.0%}")
    return profile, available

def show_calendar():
    st.title("⚽ Football AI Predictor")
    st.caption("Calendario y análisis estadístico · Horario peninsular español")
    st.subheader("Calendario")
    d = today()
    if "calendar_date" not in st.session_state: st.session_state.calendar_date = d
    cols = st.columns(4)
    for col, (label, day) in zip(cols, [("Hoy",d),("Mañana",d+timedelta(days=1)),
        ("Sábado",d+timedelta(days=(5-d.weekday())%7)),("Domingo",d+timedelta(days=(6-d.weekday())%7))]):
        if col.button(label, use_container_width=True):
            st.session_state.calendar_date = day; st.rerun()
    chosen = st.date_input("Fecha", key="calendar_date", format="DD/MM/YYYY")
    comps = [x for x,y in LEAGUES.items() if y[1]]
    selected = st.multiselect("Competiciones", comps, default=comps, key="calendar_competitions")
    key = api_key()
    if not key:
        st.error("No encuentro API_NINJAS_KEY en los Secrets de Streamlit."); return
    matches, errors = [], []
    with st.spinner("Consultando calendario..."):
        for comp in selected:
            try:
                for item in fixtures_for(LEAGUES[comp][0], chosen.isoformat(), key):
                    if not isinstance(item,dict): continue
                    dt = parse_dt(item.get("start_time") or item.get("date"))
                    hs, aws = item.get("home_score"), item.get("away_score")
                    matches.append({"Hora":dt.strftime("%H:%M") if dt else "Pendiente",
                        "Competición":comp,"Local":team_name(item.get("home_team"),"Local"),
                        "Visitante":team_name(item.get("away_team"),"Visitante"),
                        "Estado":item.get("status","Sin estado"),
                        "Resultado":f"{hs} - {aws}" if hs is not None and aws is not None else "—"})
            except Exception as e: errors.append(f"{comp}: {e}")
    if errors:
        with st.expander(f"Incidencias ({len(errors)})"):
            for e in errors: st.warning(e)
    st.session_state.day_fixtures = matches
    if not matches:
        st.info(f"No hay partidos para el {chosen:%d/%m/%Y} en las competiciones elegidas."); return
    matches.sort(key=lambda x:(x["Hora"],x["Competición"],x["Local"]))
    st.success(f"{len(matches)} partidos encontrados.")
    st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
    labels = [f'{x["Hora"]} · {x["Competición"]} · {x["Local"]} vs {x["Visitante"]}' for x in matches]
    chosen_label = st.selectbox("Elige el partido que quieres analizar", labels, key="fixture_choice")
    if st.button("Analizar este partido", type="primary"):
        st.session_state.selected_fixture = matches[labels.index(chosen_label)]
        st.rerun()

def markets(pred):
    result = {
        "Victoria local":pred.get("home_win"), "Empate":pred.get("draw"),
        "Victoria visitante":pred.get("away_win"), "Doble oportunidad 1X":pred.get("double_chance_1x"),
        "Doble oportunidad X2":pred.get("double_chance_x2"), "Doble oportunidad 12":pred.get("double_chance_12"),
        "Más de 1,5 goles":pred.get("over15"), "Más de 2,5 goles":pred.get("over25"),
        "Más de 3,5 goles":pred.get("over35"), "Ambos equipos marcan - Sí":pred.get("btts")}
    defs = {"corners":("córners totales",[7.5,8.5,9.5,10.5]),
        "fouls":("faltas totales",[19.5,21.5,23.5,25.5]),
        "yellow_cards":("tarjetas amarillas totales",[2.5,3.5,4.5,5.5]),
        "saves":("paradas estimadas",[3.5,4.5,5.5,6.5]),
        "shots_on_target":("tiros a puerta totales",[7.5,8.5,9.5,10.5])}
    for k,(label,lines) in defs.items():
        item = pred.get("auxiliary",{}).get(k)
        if item and item.get("total") is not None:
            for line in lines:
                p = poisson_over(item["total"],line)
                if p is not None:
                    l = str(line).replace(".",",")
                    result[f"Más de {l} {label}"] = p
                    result[f"Menos de {l} {label}"] = 1-p
    return result

def stake_advice(p, odds, edge, cash, profile, combo=False):
    if not (0 < p < 1 and odds > 1 and cash > 0): return 0.0,"Probabilidad o saldo insuficiente"
    if p*odds-1 <= 0 or edge < .01: return 0.0,"Sin ventaja/valor esperado suficiente"
    if combo: cap, tier = profile["combo"], "combinada"
    else:
        if p >= .68 and edge >= .04: tier, cap = "alta", profile["high"]
        elif p >= .57 and edge >= .025: tier, cap = "media", profile["medium"]
        elif p >= .50 and edge >= .01: tier, cap = "normal", profile["normal"]
        else: return 0.0,"Confianza insuficiente"
    kelly = max(0.0,(p*odds-1)/(odds-1))
    amount = cash * min(cap, kelly*.50)
    # No permitir que las apuestas pendientes superen el límite global del perfil.
    _, pending_exposure, _ = stats_log()
    initial_bank = float(st.session_state.get("bankroll_initial", 100.0))
    exposure_room = max(0.0, initial_bank * profile["exposure"] - pending_exposure)
    amount = min(amount, exposure_room)
    return amount,tier

def render_odds(pred, name, profile, cash):
    st.markdown("#### Comparador de cuotas Bet365")
    st.caption("Introduce la cuota decimal manualmente; el modelo no consulta cuotas en directo.")
    choices = markets(pred)
    market = st.selectbox("Mercado",list(choices),key="odds_market")
    p = choices[market]
    if p is None or pd.isna(p): st.warning("No hay probabilidad válida para este mercado."); return
    odd = st.number_input("Cuota Bet365",1.01,1000.0,2.00,.05,key="odds_decimal")
    fair = 1/p if p>0 else float("inf")
    edge = p-1/odd
    a,b,c = st.columns(3)
    a.metric("Probabilidad estimada",f"{p:.1%}")
    b.metric("Probabilidad implícita",f"{1/odd:.1%}")
    c.metric("Valor esperado",f"{p*odd-1:+.1%}")
    st.write(f"Cuota justa: **{fair:.2f}** · Ventaja: **{edge:+.1%}**")
    n = min(int(pred.get("home_matches",0)),int(pred.get("away_matches",0)))
    st.caption(f"Muestra mínima de equipos: {n} partidos.")
    suggested,tier = stake_advice(float(p),float(odd),edge,cash,profile)
    if n < 8: suggested,tier = 0.0,"Muestra inferior a 8 partidos"
    if suggested: st.success(f"Confianza {tier}. Stake orientativo: **{suggested:.2f} €**.")
    else: st.warning(f"No se recomienda stake: {tier}.")
    if cash <= 0: st.info("No queda saldo disponible."); return
    stake = st.number_input("Importe que quieres registrar (€)",0.0,float(cash),
                            float(min(suggested,cash)),.5,key="actual_bet_stake")
    _, pending_exposure, _ = stats_log()
    exposure_limit = float(st.session_state.get("bankroll_initial",100.0)) * profile["exposure"]
    if stake > cash*profile["max"]+.001:
        st.error(f"Supera el límite de {profile['max']:.0%} por apuesta.")
    elif pending_exposure + stake > exposure_limit + .001:
        st.error(f"Supera el límite de exposición total ({profile['exposure']:.0%} de la banca).")
    elif stake>0 and st.button("Registrar apuesta",type="primary"):
        st.session_state.bet_log.append({"Fecha registro":datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
            "Partido":name,"Mercado":market,"Probabilidad modelo":round(float(p),6),
            "Cuota Bet365":float(odd),"Cuota justa":round(fair,4),
            "Valor esperado":round(float(p*odd-1),6),"Ventaja":round(float(edge),6),
            "Stake":round(float(stake),2),"Resultado":"Pendiente","Beneficio neto":0.0})
        st.success("Apuesta registrada."); st.rerun()

def combo_candidates(raw):
    rows = []
    for f in st.session_state.get("day_fixtures",[]):
        comp = f["Competición"]
        if not LEAGUES[comp][1]: continue
        league = raw[raw.League==comp].copy()
        h, a = match_team_name(f["Local"],league), match_team_name(f["Visitante"],league)
        if not h or not a or h==a: continue
        try: pred = predict_match(league,h,a)
        except Exception: continue
        n = min(int(pred.get("home_matches",0)),int(pred.get("away_matches",0)))
        if n < 8: continue
        for label,p in markets(pred).items():
            if p is not None and not pd.isna(p) and .54 <= float(p) <= .88:
                rows.append({"Hora":f["Hora"],"Partido":f"{h} vs {a}","Mercado":label,
                    "Probabilidad":float(p),"Cuota justa":1/float(p),"Muestra":n,
                    "score":float(p)*.75+min(n,20)/20*.25})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["score","Muestra"],ascending=False)
    return df.drop_duplicates("Partido").reset_index(drop=True)

def render_combo(raw, profile, cash):
    st.divider(); st.header("Constructor de combinadas")
    st.caption("Máximo una selección por partido. Elige entre 2 y 15 partidos; solo se muestran selecciones con muestra histórica suficiente.")
    df = combo_candidates(raw)
    if df.empty:
        st.info("No hay suficientes selecciones con al menos 8 partidos de muestra y probabilidad entre 54% y 88%."); return
    maximum = min(15,len(df))
    count = st.slider("Número de selecciones",2,max(2,maximum),min(4,max(2,maximum)))
    view = df.copy()
    view["Probabilidad"] = view["Probabilidad"].map(lambda x:f"{x:.1%}")
    view["Cuota justa"] = view["Cuota justa"].map(lambda x:f"{x:.2f}")
    st.dataframe(view[["Hora","Partido","Mercado","Probabilidad","Cuota justa","Muestra"]],
                 use_container_width=True,hide_index=True)
    picks = df.head(count)
    pcombo = float(picks.Probabilidad.prod())
    faircombo = float(picks["Cuota justa"].prod())
    st.markdown("#### Combinada propuesta")
    st.dataframe(picks[["Partido","Mercado","Probabilidad","Cuota justa"]],
                 use_container_width=True,hide_index=True)
    x,y = st.columns(2)
    x.metric("Probabilidad conjunta aproximada",f"{pcombo:.3%}")
    y.metric("Cuota justa estimada",f"{faircombo:.2f}")
    st.caption("Producto de probabilidades individuales: supone independencia entre partidos y es solo una aproximación.")
    actual = st.number_input("Cuota total real de Bet365",1.01,100000.0,
                              float(max(1.01,round(faircombo,2))),.05,key="combo_odds")
    edge = pcombo-1/actual
    ev = pcombo*actual-1
    a,b,c = st.columns(3)
    a.metric("Probabilidad implícita",f"{1/actual:.3%}")
    b.metric("Ventaja estimada",f"{edge:+.2%}")
    c.metric("Valor esperado",f"{ev:+.2%}")
    amount,tier = stake_advice(pcombo,actual,edge,cash,profile,True)
    if amount: st.success(f"Stake orientativo ({st.session_state.risk_profile}): **{amount:.2f} €**.")
    else: st.warning("No se recomienda apostar esta combinada con la cuota introducida.")
    st.caption(f"Tope de stake para combinadas: {profile['combo']:.1%} de la banca disponible.")
    st.info("La cuota debe introducirse manualmente. El historial todavía no liquida combinadas; no se registran como apuestas simples.")

def radar(raw):
    fs = st.session_state.get("day_fixtures",[])
    if not fs: return
    st.subheader("Radar de la jornada")
    rows=[]
    for f in fs:
        comp=f["Competición"]; league=raw[raw.League==comp].copy()
        h,a=match_team_name(f["Local"],league),match_team_name(f["Visitante"],league)
        if not h or not a or h==a: continue
        try: pred=predict_match(league,h,a)
        except Exception: continue
        for label,p in [("Victoria local",pred["home_win"]),("Empate",pred["draw"]),
            ("Victoria visitante",pred["away_win"]),("Doble oportunidad 1X",pred["double_chance_1x"]),
            ("Doble oportunidad X2",pred["double_chance_x2"]),("Más de 1,5 goles",pred["over15"]),
            ("Más de 2,5 goles",pred["over25"]),("Ambos marcan",pred["btts"])]:
            rows.append({"Hora":f["Hora"],"Competición":comp,"Partido":f"{h} vs {a}",
                         "Selección":label,"Probabilidad":p,"Cuota justa":1/p if p>0 else None})
    if rows:
        df=pd.DataFrame(rows).sort_values("Probabilidad",ascending=False)
        df["Probabilidad"]=df["Probabilidad"].map(lambda x:f"{x:.1%}")
        df["Cuota justa"]=df["Cuota justa"].map(lambda x:f"{x:.2f}")
        st.dataframe(df,use_container_width=True,hide_index=True)

def tracker():
    st.divider(); st.subheader("Seguimiento de apuestas")
    log=st.session_state.setdefault("bet_log",[])
    if log:
        profit,pending,settled=stats_log()
        initial=float(st.session_state.get("bankroll_initial",100))
        c1,c2,c3=st.columns(3)
        c1.metric("Banca inicial",f"{initial:.2f} €")
        c2.metric("Saldo disponible",f"{bankroll():.2f} €")
        c3.metric("Beneficio liquidado",f"{profit:+.2f} €")
        df=pd.DataFrame(log)
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Descargar historial CSV",df.to_csv(index=False).encode("utf-8-sig"),
                           "football_ai_historial_apuestas.csv","text/csv")
        pending_ids=[i for i,x in enumerate(log) if x.get("Resultado")=="Pendiente"]
        if pending_ids:
            idx=st.selectbox("Apuesta que quieres liquidar",pending_ids,
                format_func=lambda i:f'{log[i].get("Partido")} · {log[i].get("Mercado")} · {log[i].get("Stake")} €')
            status=st.selectbox("Resultado",["Ganada","Perdida","Anulada"])
            if st.button("Guardar resultado"):
                item=log[idx]; s=float(item.get("Stake",0)); o=float(item.get("Cuota Bet365",0))
                item["Resultado"]=status
                item["Beneficio neto"]=round(s*(o-1),2) if status=="Ganada" else (-s if status=="Perdida" else 0)
                st.rerun()
    else: st.info("Todavía no has registrado apuestas.")
    uploaded=st.file_uploader("Importar historial CSV anterior",type=["csv"])
    if uploaded is not None and st.button("Importar historial"):
        try:
            df=pd.read_csv(uploaded).fillna("")
            if not {"Partido","Mercado","Cuota Bet365","Resultado"}.issubset(df.columns):
                st.error("El CSV no contiene las columnas necesarias.")
            else:
                st.session_state.bet_log=df.to_dict("records"); st.success("Historial importado."); st.rerun()
        except Exception as e: st.error(f"No se pudo importar: {e}")

def main():
    show_calendar()
    with st.sidebar:
        st.header("Datos históricos")
        st.caption("Resultados de Football-Data.co.uk.")
        if st.button("Actualizar histórico ahora"): historical.clear(); st.rerun()
    profile,cash=risk_setup()
    raw,errors=historical()
    if raw.empty:
        st.error("No se pudo cargar el histórico. Comprueba la conexión e inténtalo más tarde."); st.stop()
    radar(raw)
    render_combo(raw,profile,cash)
    selected=st.session_state.get("selected_fixture")
    if not selected:
        st.info("Selecciona un partido del calendario para ver su análisis individual.")
        st.caption(f"Histórico cargado: {len(raw):,} resultados en {raw.League.nunique()} competiciones.")
        tracker(); return
    st.divider(); st.header("Análisis del partido seleccionado")
    st.info(f'{selected["Local"]} vs {selected["Visitante"]} · {selected["Competición"]}')
    comp=selected["Competición"]
    if not LEAGUES[comp][1]:
        st.warning("Este torneo está en el calendario, pero no tiene histórico incorporado."); tracker(); return
    league=raw[raw.League==comp].copy()
    home,away=match_team_name(selected["Local"],league),match_team_name(selected["Visitante"],league)
    if not home or not away:
        st.warning("No he encontrado coincidencia segura de nombres.")
        teams=sorted(set(league.HomeTeam).union(set(league.AwayTeam)))
        st.selectbox("Equipo local",teams,key="manual_home")
        st.selectbox("Equipo visitante",teams,key="manual_away")
        if st.button("Analizar equipos seleccionados"):
            home,away=st.session_state.manual_home,st.session_state.manual_away
        else: tracker(); return
    if home==away: st.error("Los equipos no pueden ser el mismo."); tracker(); return
    pred=predict_match(league,home,away)
    st.subheader(f"{home} vs {away}")
    a,b,c=st.columns(3)
    a.metric("Victoria local",f'{pred["home_win"]:.1%}')
    b.metric("Empate",f'{pred["draw"]:.1%}')
    c.metric("Victoria visitante",f'{pred["away_win"]:.1%}')
    a,b,c,d=st.columns(4)
    a.metric("Más de 1,5",f'{pred["over15"]:.1%}')
    b.metric("Más de 2,5",f'{pred["over25"]:.1%}')
    c.metric("Más de 3,5",f'{pred["over35"]:.1%}')
    d.metric("Ambos marcan",f'{pred["btts"]:.1%}')
    a,b,c=st.columns(3)
    a.metric("Doble oportunidad 1X",f'{pred["double_chance_1x"]:.1%}')
    b.metric("Doble oportunidad X2",f'{pred["double_chance_x2"]:.1%}')
    c.metric("Doble oportunidad 12",f'{pred["double_chance_12"]:.1%}')
    st.write(f'Goles esperados: **{home} {pred["lambda_home"]:.2f}** · **{away} {pred["lambda_away"]:.2f}**')
    st.markdown("#### Otros mercados estadísticos")
    for k,label in [("corners","Córners"),("fouls","Faltas"),("yellow_cards","Tarjetas amarillas"),
                    ("shots_on_target","Tiros a puerta"),("saves","Paradas estimadas")]:
        item=pred.get("auxiliary",{}).get(k)
        if item and item.get("total") is not None:
            st.write(f'**{label}:** {home} {item["home"]:.1f} · {away} {item["away"]:.1f} · Total {item["total"]:.1f}')
        else: st.caption(f"{label}: datos insuficientes.")
    scores=pd.DataFrame(pred["top_scores"],columns=["Marcador","Probabilidad"])
    scores["Probabilidad"]=scores["Probabilidad"].map(lambda x:f"{x:.1%}")
    st.markdown("#### Marcadores exactos más probables")
    st.dataframe(scores,use_container_width=True,hide_index=True)
    st.caption(f'Muestra: {pred.get("home_matches",0)} partidos del local en casa y {pred.get("away_matches",0)} del visitante fuera.')
    render_odds(pred,f"{home} vs {away}",profile,cash)
    st.warning("Modelo Poisson básico, no validado como sistema rentable. Los mercados auxiliares son estimaciones de promedios de equipos, no datos en directo. No incorpora alineaciones, lesiones, árbitro ni cuotas en tiempo real.")
    st.info("No se inventan estadísticas individuales de jugadores; falta una fuente fiable por futbolista.")
    tracker()
    with st.expander("Sobre los datos"):
        st.write("Temporadas 2022/23 a 2026/27 cuando están disponibles. Fuente: Football-Data.co.uk.")
        for error in errors[:30]: st.write(error)

if __name__ == "__main__":
    main()
