import pandas as pd
import streamlit as st

import team_form
from odds_api import get_match_odds, odds_for_market


MARKETS_TO_COMPARE = [
    "Victoria local", "Empate", "Victoria visitante",
    "Más de 1,5 goles", "Más de 2,5 goles", "Más de 3,5 goles",
]


def _pct(value):
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "No disponible"


def _recent_team(team):
    try:
        all_results = team_form.load_official_results()
        if all_results is None or all_results.empty:
            return pd.DataFrame(), None
        matches = team_form.last_five(team, all_results)
        if matches.empty:
            return matches, None
        return matches, team_form.form_summary(matches)
    except Exception:
        return pd.DataFrame(), None


def _head_to_head(league, home, away, limit=5):
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if league is None or league.empty or not required.issubset(league.columns):
        return pd.DataFrame()
    matches = league[
        ((league["HomeTeam"] == home) & (league["AwayTeam"] == away)) |
        ((league["HomeTeam"] == away) & (league["AwayTeam"] == home))
    ].copy()
    if matches.empty:
        return matches
    if "Date" in matches.columns:
        matches["_date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
        matches = matches.sort_values("_date", ascending=False, na_position="last")
    cols = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"] if c in matches.columns]
    return matches[cols].head(limit)


def _form_text(team, summary):
    if not summary or not summary.get("played"):
        return f"No hay suficientes resultados recientes verificables para {team}."
    played = summary["played"]
    return (
        f"{team}: {summary['wins']} victorias, {summary['draws']} empates y "
        f"{summary['losses']} derrotas en sus últimos {played} partidos oficiales "
        f"disponibles. Promedia {summary['goals_for']:.2f} goles a favor y "
        f"{summary['goals_against']:.2f} en contra por encuentro."
    )


def _best_market_rows(fixture, pred, home, away):
    probabilities = {
        "Victoria local": pred.get("home_win"),
        "Empate": pred.get("draw"),
        "Victoria visitante": pred.get("away_win"),
        "Más de 1,5 goles": pred.get("over15"),
        "Más de 2,5 goles": pred.get("over25"),
        "Más de 3,5 goles": pred.get("over35"),
    }
    try:
        result = get_match_odds(
            fixture.get("Competición", ""),
            fixture.get("Local", home),
            fixture.get("Visitante", away),
        )
    except Exception as exc:
        return [], f"No se pudieron consultar las cuotas: {type(exc).__name__}."
    if not result.get("event"):
        return [], result.get("error") or "No hay cuotas disponibles para este encuentro."
    rows = []
    for market in MARKETS_TO_COMPARE:
        p = probabilities.get(market)
        if p is None or pd.isna(p) or not 0 < float(p) < 1:
            continue
        try:
            prices = odds_for_market(
                result.get("event"), market,
                fixture.get("Local", home), fixture.get("Visitante", away)
            )
        except Exception:
            prices = []
        if not prices:
            continue
        best = max(prices, key=lambda item: float(item["Cuota"]))
        odd = float(best["Cuota"])
        rows.append({
            "Mercado": market, "Prob. modelo": float(p), "Mejor cuota": odd,
            "Cuota justa": 1 / float(p), "Valor esperado": float(p) * odd - 1,
            "Ventaja": float(p) - 1 / odd, "Casa": best.get("Casa", "No indicada"),
            "Actualizada": best.get("Actualizada", "No indicada"),
        })
    return rows, None


def _recommendation_block(pred, odds_rows):
    st.markdown("#### Selecciones por perfil de riesgo")
    safe_candidates = [
        ("Doble oportunidad 1X", pred.get("double_chance_1x")),
        ("Doble oportunidad X2", pred.get("double_chance_x2")),
        ("Más de 1,5 goles", pred.get("over15")),
    ]
    safe_candidates = [
        (name, float(p)) for name, p in safe_candidates
        if p is not None and not pd.isna(p) and 0 < float(p) < 1
    ]
    if safe_candidates:
        name, probability = max(safe_candidates, key=lambda item: item[1])
        st.write(
            f"**Conservadora (probabilidad):** {name} — {probability:.1%} según el modelo. "
            "No implica que tenga valor a la cuota disponible."
        )
    else:
        st.write("**Conservadora:** no hay una probabilidad válida suficiente para seleccionar.")

    value_rows = [row for row in odds_rows if row["Valor esperado"] > 0 and row["Ventaja"] >= .01]
    if value_rows:
        best = max(value_rows, key=lambda row: row["Valor esperado"])
        st.write(
            f"**Intermedia (valor):** {best['Mercado']} — cuota {best['Mejor cuota']:.2f}, "
            f"probabilidad {best['Prob. modelo']:.1%}, EV {best['Valor esperado']:+.1%}."
        )
    else:
        st.write("**Intermedia:** no se detecta ninguna selección con valor esperado positivo y ventaja mínima del 1%.")

    ambitious = [row for row in value_rows if row["Mejor cuota"] >= 2.00]
    if ambitious:
        best = max(ambitious, key=lambda row: row["Valor esperado"])
        st.write(
            f"**Ambiciosa (mayor varianza):** {best['Mercado']} — cuota {best['Mejor cuota']:.2f}, "
            f"probabilidad {best['Prob. modelo']:.1%}, EV {best['Valor esperado']:+.1%}."
        )
    else:
        st.write("**Ambiciosa:** no hay una selección de cuota igual o superior a 2.00 que supere los filtros de valor.")


def render_intelligent_report(fixture, pred, home, away, league, profile, cash):
    """Informe narrativo basado en el predictor y en fuentes disponibles."""
    st.subheader(f"{home} vs {away}")
    st.caption(
        f"{fixture.get('Competición', 'Competición no indicada')} · "
        f"{fixture.get('Hora', 'Hora no disponible')} · Informe previo basado en datos disponibles"
    )

    probs = [
        (home, float(pred.get("home_win") or 0)),
        ("Empate", float(pred.get("draw") or 0)),
        (away, float(pred.get("away_win") or 0)),
    ]
    favorite, favorite_p = max(probs, key=lambda item: item[1])
    if favorite_p > 0:
        if favorite == "Empate":
            opening = f"El modelo considera el empate como el resultado individual más probable ({favorite_p:.1%})."
        else:
            opening = f"El modelo se inclina por {favorite} como resultado más probable ({favorite_p:.1%})."
    else:
        opening = "El modelo no ha devuelto probabilidades 1X2 válidas para este encuentro."

    home_recent, home_summary = _recent_team(home)
    away_recent, away_summary = _recent_team(away)

    st.markdown("### Análisis de los equipos")
    st.write(_form_text(home, home_summary))
    st.write(_form_text(away, away_summary))
    if home_summary and away_summary:
        points_diff = home_summary["points"] - away_summary["points"]
        if points_diff > 0:
            st.write(f"En la muestra reciente disponible, {home} ha sumado más puntos que {away}.")
        elif points_diff < 0:
            st.write(f"En la muestra reciente disponible, {away} ha sumado más puntos que {home}.")
        else:
            st.write("Ambos equipos presentan el mismo número de puntos en la muestra reciente disponible.")
    else:
        st.caption("La forma reciente no está completa para ambos equipos; no se extrae una conclusión comparativa.")

    with st.expander("Ver últimos partidos oficiales"):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{home}**")
            if home_recent.empty:
                st.info("Sin resultados recientes disponibles.")
            else:
                st.dataframe(home_recent, use_container_width=True, hide_index=True)
        with right:
            st.markdown(f"**{away}**")
            if away_recent.empty:
                st.info("Sin resultados recientes disponibles.")
            else:
                st.dataframe(away_recent, use_container_width=True, hide_index=True)

    st.markdown("### Enfrentamientos directos")
    h2h = _head_to_head(league, home, away)
    if h2h.empty:
        st.caption("No se han encontrado enfrentamientos directos en el histórico de liga integrado.")
    else:
        st.dataframe(h2h, use_container_width=True, hide_index=True)
        st.caption("Histórico de liga de Football-Data.co.uk; no equivale a un registro completo de todas las copas.")

    st.markdown("### Mi pronóstico")
    st.write(opening)
    a, b, c = st.columns(3)
    a.metric(f"Victoria {home}", _pct(pred.get("home_win")))
    b.metric("Empate", _pct(pred.get("draw")))
    c.metric(f"Victoria {away}", _pct(pred.get("away_win")))
    d, e, f = st.columns(3)
    d.metric("Más de 1,5 goles", _pct(pred.get("over15")))
    e.metric("Más de 2,5 goles", _pct(pred.get("over25")))
    f.metric("Ambos marcan", _pct(pred.get("btts")))

    lh, la = pred.get("lambda_home"), pred.get("lambda_away")
    if lh is not None and la is not None:
        st.write(f"Goles esperados por el modelo: **{home} {float(lh):.2f}** · **{away} {float(la):.2f}**.")
    scores = pred.get("top_scores") or []
    if scores:
        score, score_p = scores[0]
        st.write(f"**Marcador exacto más probable:** {score} ({float(score_p):.1%} según el modelo).")

    st.markdown("### Cuotas y valor de mercado")
    with st.spinner("Contrastando probabilidades con las cuotas disponibles..."):
        odds_rows, odds_error = _best_market_rows(fixture, pred, home, away)
    if odds_rows:
        display = pd.DataFrame(odds_rows).copy()
        for col in ["Prob. modelo", "Valor esperado", "Ventaja"]:
            display[col] = display[col].map(lambda value: f"{value:.1%}")
        for col in ["Mejor cuota", "Cuota justa"]:
            display[col] = display[col].map(lambda value: f"{value:.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Cuotas devueltas por The Odds API en el momento de la consulta; pueden cambiar.")
    else:
        st.info(odds_error or "No hay cuotas comparables para este partido.")
    _recommendation_block(pred, odds_rows)

    sample = min(int(pred.get("home_matches", 0)), int(pred.get("away_matches", 0)))
    if sample < 8:
        confidence = "Insuficiente"
        explanation = "La muestra histórica mínima de uno de los equipos es inferior a ocho partidos."
    elif sample < 15:
        confidence = "Limitada"
        explanation = "La muestra permite una estimación inicial, pero sigue siendo reducida."
    else:
        confidence = "Moderada"
        explanation = "Hay una muestra histórica mayor, pero la confianza no está calibrada mediante validación prospectiva."
    st.markdown("### Confianza y control de riesgo")
    a, b = st.columns(2)
    a.metric("Confianza de la muestra", confidence)
    b.metric("Partidos históricos mínimos", sample)
    st.caption(explanation)
    st.write(
        f"Perfil de riesgo configurado: **{st.session_state.get('risk_profile', 'Equilibrado')}** · "
        f"Banca disponible: **{cash:.2f} €**."
    )
    st.warning(
        "No hay una fuente integrada y fiable para lesiones, sanciones, árbitro ni alineaciones confirmadas. "
        "Esos factores no se inventan. El informe es una ayuda estadística, no una garantía ni una instrucción de apuesta."
    )
