# Football AI Predictor

Aplicación Streamlit para consultar partidos y generar estimaciones estadísticas de fútbol.

## Fuente de datos

La aplicación utiliza API-Football v3 (`https://v3.football.api-sports.io`) para calendario e histórico reciente. Las competiciones y los equipos se relacionan mediante sus identificadores oficiales, no mediante equivalencias de nombres.

Competiciones configuradas:
- LaLiga (ID 140)
- Segunda División (ID 141)
- Premier League (ID 39)
- Bundesliga (ID 78)
- Serie A (ID 135)
- Ligue 1 (ID 61)
- Eredivisie (ID 88)

Champions League y Europa League se incorporarán en una fase posterior, una vez verificada la cobertura y el consumo de consultas.

## Configuración en Streamlit

En **App → Settings → Secrets**, configura:

```toml
API_FOOTBALL_KEY = "TU_CLAVE_DE_API_FOOTBALL"
```

No publiques ni compartas la clave. No es necesario conservar `API_NINJAS_KEY`.

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Alcance de esta versión

- Calendario por fecha para las siete competiciones configuradas.
- Selección de partido con fixture ID y team IDs de API-Football.
- Histórico reciente de la temporada 2026, limitado a los últimos 120 días y a la primera página de la competición del partido seleccionado para controlar el consumo del plan gratuito.
- Estimaciones Poisson para 1X2, doble oportunidad, goles, ambos marcan y marcadores exactos.
- Comparador manual de cuotas Bet365 y seguimiento exportable/importable en CSV.

Las estadísticas de córners, faltas, tarjetas, tiros a puerta, paradas y mercados individuales de jugadores están desactivadas hasta verificar datos reales suficientes. El modelo no garantiza rentabilidad. Comprueba cobertura y condiciones de API-Football antes de ampliar el uso o comercializar la aplicación.
