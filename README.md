# Football AI Predictor — actualización

## Archivos
- `app.py`: calendario y pantalla de análisis.
- `predictor.py`: modelo Poisson y comparación de cuotas.
- `requirements.txt`: dependencias de Streamlit Cloud.
- `data/README.txt`: nota sobre datos históricos.

## Instalación
1. Descarga y descomprime este ZIP.
2. En GitHub abre el repositorio `football-ai-predictor`.
3. Sustituye `app.py`, `predictor.py` y `requirements.txt` por los de este paquete.
4. Conserva el archivo `README.md` del proyecto o sustitúyelo por este.
5. No borres `API_NINJAS_KEY` de Streamlit Secrets.
6. Espera a que Streamlit Cloud reinicie la aplicación.

## Datos históricos
La aplicación descarga automáticamente, con caché de 6 horas, los CSV de Football-Data.co.uk para las temporadas 2022/23 a 2026/27, si están disponibles:
- SP1 LaLiga
- SP2 Segunda
- E0 Premier League
- D1 Bundesliga
- I1 Serie A
- F1 Ligue 1
- N1 Eredivisie

No es necesario subir un CSV manualmente. En la barra lateral hay un botón para invalidar la caché y volver a consultar los archivos.

## Importante
- El calendario usa API Ninjas y necesita el secreto `API_NINJAS_KEY` ya configurado.
- Champions League y Europa League aparecen en el calendario, pero el análisis histórico de esos torneos queda pendiente para una fase posterior.
- El modelo es Poisson básico, no un sistema de IA entrenado ni una estrategia rentable validada. No incorpora lesiones, alineaciones, xG ni cuotas en directo.
- Los CSV de Football-Data.co.uk están destinados a uso personal. Consulta sus condiciones antes de redistribuirlos o dar acceso comercial a la aplicación.
- Las cuotas de Bet365 se introducen manualmente; no se realiza scraping.
