import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

import app_base as base
from odds_api import get_match_odds, odds_for_market

TZ = ZoneInfo("Europe/Madrid")


def _comparison(p, odd, cash, profile, market, name, bookmaker, updated, sample):
    edge = float(p) - 1 / float(odd)
    fair = 1 / float(p) if float(p) > 0 else float("inf")
    a, b, c = st.columns(3)
    a.metric("Probabilidad estimada", f"{p:.1%}")
    b.metric("Probabilidad implícita", f"{1 / odd:.1%}")
    c.metric("Valor esperado", f"{p * odd - 1:+.1%}")
    st.write(f"Cuota justa: **{fair:.2f}** · Ventaja: **{edge:+.1%}**")
    st.caption(f"Casa: {bookmaker} · Actualización: {updated}")
    st.caption(f"Muestra mínima de equipos: {sample} partidos.")
    suggested, tier = base.stake_advice(float(p), float(odd), edge, cash, profile)
    if sample < 8:
        suggested, tier = 0.0, "Muestra inferior a 8 partidos"
    if suggested:
        st.success(f"Confianza {tier}. Stake orientativo: **{suggested:.2f} €**.")
    else:
        st.warning(f"No se recomienda stake: {tier}.")
    if cash <= 0:
        st.info("No queda saldo disponible.")
        return
    stake = st.number_input("Importe que quieres registrar (€)", 0.0, float(cash),
                            float(min(suggested, cash)), .5, key="actual_bet_stake")
    _, pending, _ = base.stats_log()
    limit = float(st.session_state.get("bankroll_initial", 100.0)) * profile["exposure"]
    if stake > cash * profile["max"] + .001:
        st.error(f"Supera el límite de {profile['max']:.0%} por apuesta.")
    elif pending + stake > limit + .001:
        st.error(f"Supera el límite de exposición total ({profile['exposure']:.0%}).")
    elif stake > 0 and st.button("Registrar apuesta", type="primary", key="register_auto_odds_bet"):
        base.st.session_state.bet_log.append({
            "Fecha registro": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
            "Partido": name, "Mercado": market, "Probabilidad modelo": round(float(p), 6),
            "Cuota Bet365": float(odd), "Casa de apuestas": bookmaker,
            "Cuota justa": round(fair, 4), "Valor esperado": round(float(p * odd - 1), 6),
            "Ventaja": round(float(edge), 6), "Stake": round(float(stake), 2),
            "Resultado": "Pendiente", "Beneficio neto": 0.0
        })
        st.success("Apuesta registrada.")
        st.rerun()


def render_odds(pred, name, profile, cash):
    st.markdown("#### Comparador de cuotas automáticas")
    st.caption("Cuotas europeas consultadas a través de The Odds API. La disponibilidad depende del partido y del mercado.")
    choices = base.markets(pred)
    market = st.selectbox("Mercado", list(choices), key="odds_market")
    p = choices[market]
    if p is None or pd.isna(p):
        st.warning("No hay probabilidad válida para este mercado.")
        return
    fixture = st.session_state.get("selected_fixture", {})
    result = get_match_odds(fixture.get("Competición", ""),
                            fixture.get("Local", ""), fixture.get("Visitante", ""))
    rows = odds_for_market(result.get("event"), market,
                           fixture.get("Local", ""), fixture.get("Visitante", ""))
    sample = min(int(pred.get("home_matches", 0)), int(pred.get("away_matches", 0)))
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        names = ["Mejor cuota disponible"] + sorted(set(x["Casa"] for x in rows))
        selected_book = st.selectbox("Utilizar cuota de", names, key="odds_bookmaker")
        if selected_book == "Mejor cuota disponible":
            chosen = max(rows, key=lambda x: x["Cuota"])
        else:
            chosen = max([x for x in rows if x["Casa"] == selected_book],
                         key=lambda x: x["Cuota"])
        st.success(f"Cuota seleccionada: **{chosen['Cuota']:.2f}** · {chosen['Casa']}")
        _comparison(float(p), float(chosen["Cuota"]), cash, profile, market, name,
                    chosen["Casa"], chosen.get("Actualizada", "No indicada"), sample)
    else:
        if result.get("error"):
            st.warning(f"No se pudieron obtener cuotas automáticas: {result['error']}")
        else:
            st.info("No hay cuota automática para este mercado o partido. Puedes introducirla manualmente.")
        odd = st.number_input("Cuota decimal manual", 1.01, 1000.0, 2.00, .05,
                              key="odds_decimal")
        _comparison(float(p), float(odd), cash, profile, market, name,
                    "Manual", "Introducida por el usuario", sample)


base.render_odds = render_odds
base.main()
