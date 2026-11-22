import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

import app_base as base
from odds_api import get_match_odds, odds_for_market
import ml_predictor

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


_original_analysis = base.render_analysis
_original_results = base.page_results


def _ml_comparison_panel(raw, profile, cash):
    fixture = st.session_state.get("selected_fixture")
    if not fixture or raw is None or raw.empty:
        return
    league_df, home, away = base.selected_teams(raw, fixture)
    if league_df is None or not home or not away:
        return
    st.divider()
    st.subheader("Segundo modelo IA · Machine Learning")
    st.caption("Modelo independiente del Poisson. Aprende de resultados históricos y utiliza únicamente estadísticas anteriores a cada partido.")
    try:
        with st.spinner("Entrenando y validando el modelo con el histórico..."):
            ml = ml_predictor.predict_match(raw, fixture.get("Competición"), home, away)
    except Exception as exc:
        st.warning(f"El segundo modelo todavía no puede generar una predicción: {exc}")
        return

    validation = ml["validation"]
    a, b, c = st.columns(3)
    a.metric("Partidos de entrenamiento", f'{ml["training_matches"]:,}')
    b.metric("Acierto validación temporal", f'{validation["accuracy"]:.1%}')
    c.metric("Log-loss validación", f'{validation["log_loss"]:.3f}')
    st.caption(
        f"Validación cronológica: {validation['validation_matches']} partidos posteriores "
        f"al corte {validation['cutoff']}. El acierto y el log-loss son métricas históricas, "
        "no una garantía de rendimiento futuro."
    )

    poisson = base.predict_match(league_df, home, away)
    labels = [
        ("Victoria local", "home_win"),
        ("Empate", "draw"),
        ("Victoria visitante", "away_win"),
    ]
    rows = []
    for label, key in labels:
        p1, p2 = float(poisson[key]), float(ml[key])
        rows.append({
            "Mercado": label,
            "Poisson": f"{p1:.1%}",
            "Machine Learning": f"{p2:.1%}",
            "Diferencia ML - Poisson": f"{p2-p1:+.1%}",
            "Cuota justa ML": f"{1/p2:.2f}" if p2 > 0 else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Comparar ambos modelos con la cuota real")
    market_map = {
        "Victoria local": ("home_win", "Victoria local"),
        "Empate": ("draw", "Empate"),
        "Victoria visitante": ("away_win", "Victoria visitante"),
    }
    market = st.selectbox("Mercado 1X2 para comparar", list(market_map), key="ml_compare_market")
    key, odds_label = market_map[market]
    odds_result = get_match_odds(fixture.get("Competición", ""), fixture.get("Local", ""), fixture.get("Visitante", ""))
    odds_rows = odds_for_market(odds_result.get("event"), odds_label, fixture.get("Local", ""), fixture.get("Visitante", ""))
    if odds_rows:
        best = max(odds_rows, key=lambda x: float(x["Cuota"]))
        odd = float(best["Cuota"])
        st.write(f"Mejor cuota disponible: **{odd:.2f}** · {best.get('Casa', 'Casa desconocida')}")
        st.caption(f"Actualización de cuota: {best.get('Actualizada', 'No indicada')}")
    else:
        if odds_result.get("error"):
            st.info(f"No se pudieron consultar cuotas automáticas: {odds_result['error']}")
        else:
            st.info("No hay cuota automática para este mercado.")
        odd = st.number_input("Introduce una cuota decimal para comparar", min_value=1.01, max_value=1000.0,
                              value=2.00, step=0.05, key="ml_manual_odds")

    p_poisson, p_ml = float(poisson[key]), float(ml[key])
    comp = pd.DataFrame([
        {"Modelo": "Poisson", "Probabilidad": f"{p_poisson:.1%}",
         "Cuota justa": f"{1/p_poisson:.2f}" if p_poisson else "—",
         "Valor esperado": f"{p_poisson*odd-1:+.1%}"},
        {"Modelo": "Machine Learning", "Probabilidad": f"{p_ml:.1%}",
         "Cuota justa": f"{1/p_ml:.2f}" if p_ml else "—",
         "Valor esperado": f"{p_ml*odd-1:+.1%}"},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.caption("El segundo modelo se muestra para comparar. Su probabilidad no modifica el stake automático ni la gestión de banca.")

    # Registro prospectivo de predicciones para comprobarlas posteriormente.
    log = st.session_state.setdefault("ml_prediction_log", [])
    prediction_id = "|".join([
        str(fixture.get("Competición", "")), str(fixture.get("Hora", "")),
        str(fixture.get("Local", "")), str(fixture.get("Visitante", "")),
    ])
    if not any(item.get("ID") == prediction_id for item in log):
        log.append({
            "ID": prediction_id,
            "Fecha registro": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
            "Competición": fixture.get("Competición", ""),
            "Partido": f"{home} vs {away}",
            "Poisson local": poisson["home_win"], "Poisson empate": poisson["draw"],
            "Poisson visitante": poisson["away_win"],
            "ML local": ml["home_win"], "ML empate": ml["draw"],
            "ML visitante": ml["away_win"],
            "Cuota comparada": odd, "Mercado comparado": market,
            "Validación accuracy": validation["accuracy"],
            "Validación log-loss": validation["log_loss"],
            "Resultado real": "Pendiente",
        })
    st.caption("La predicción queda guardada durante esta sesión. Descarga el CSV para conservar el registro y poder analizarlo más adelante.")


def _render_analysis_with_ml(raw, profile, cash):
    _original_analysis(raw, profile, cash)
    _ml_comparison_panel(raw, profile, cash)


def _render_results_with_ml():
    _original_results()
    log = st.session_state.get("ml_prediction_log", [])
    st.markdown("### Seguimiento de predicciones de los modelos")
    if not log:
        st.info("Todavía no hay predicciones ML registradas en esta sesión.")
        return
    frame = pd.DataFrame(log)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar predicciones IA en CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="predicciones_modelos_ia.csv",
        mime="text/csv",
    )


base.render_odds = render_odds
base.render_analysis = _render_analysis_with_ml
base.page_results = _render_results_with_ml
base.main()
