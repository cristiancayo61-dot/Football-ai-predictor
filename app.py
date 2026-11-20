import streamlit as st
import pandas as pd
from predictor import prepare_data, predict_match, demo_data

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

    st.success(f"Datos cargados: {len(df)} partidos · {df['HomeTeam'].nunique()} equipos")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "🔮 Predicción", "📁 Datos", "ℹ️ Cómo funciona"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Partidos", len(df))
        c2.metric("Equipos", df["HomeTeam"].nunique())
        c3.metric("Goles por partido", round((df.FTHG + df.FTAG).mean(), 2))
        st.subheader("Últimos resultados del archivo")
        st.dataframe(df.tail(15).sort_index(ascending=False), use_container_width=True)
        st.subheader("Goles por equipo")
        team_stats = pd.concat([
            df.groupby("HomeTeam").agg(goles_favor=("FTHG","mean"), goles_contra=("FTAG","mean")).rename_axis("Equipo"),
            df.groupby("AwayTeam").agg(goles_favor=("FTAG","mean"), goles_contra=("FTHG","mean")).rename_axis("Equipo")
        ]).groupby(level=0).mean().round(2).sort_values("goles_favor", ascending=False)
        st.dataframe(team_stats, use_container_width=True)

    with tab2:
        teams = sorted(set(df.HomeTeam).union(set(df.AwayTeam)))
        left, right = st.columns(2)
        home = left.selectbox("Equipo local", teams, index=0)
        away_options = [t for t in teams if t != home]
        away = right.selectbox("Equipo visitante", away_options, index=min(1, len(away_options)-1))
        if st.button("Calcular predicción", type="primary"):
            result = predict_match(df, home, away)
            st.subheader(f"{home} vs {away}")
            cols = st.columns(3)
            cols[0].metric("Victoria local", f"{result['home_win']:.1%}")
            cols[1].metric("Empate", f"{result['draw']:.1%}")
            cols[2].metric("Victoria visitante", f"{result['away_win']:.1%}")
            st.markdown("#### Mercados de goles")
            a,b,c = st.columns(3)
            a.metric("Más de 1,5 goles", f"{result['over15']:.1%}")
            b.metric("Más de 2,5 goles", f"{result['over25']:.1%}")
            c.metric("Ambos marcan", f"{result['btts']:.1%}")
            st.markdown("#### Goles esperados por el modelo")
            st.write(f"{home}: **{result['lambda_home']:.2f}** · {away}: **{result['lambda_away']:.2f}**")
            st.caption("Estimación basada en promedios históricos local/visitante y distribución de Poisson. No incorpora alineaciones, lesiones ni cuotas.")
            if "corners_home" in result:
                st.markdown("#### Estadísticas adicionales")
                x,y = st.columns(2)
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
        - No descarga datos automáticamente.
        - No consulta cuotas en tiempo real.
        - No tiene lesiones, alineaciones ni xG.
        - No es todavía un modelo entrenado de Machine Learning.

        **Siguiente evolución:** validación temporal, modelo Elo/XGBoost, almacenamiento persistente e importación automatizada.
        """)
else:
    st.info("Carga un archivo CSV para ver el análisis.")
