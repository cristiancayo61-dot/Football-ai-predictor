# Football AI Predictor FREE

Primera versión educativa y gratuita de un panel de análisis futbolístico.

## Ejecutar en Streamlit Community Cloud
1. Crea una cuenta en GitHub.
2. Crea un repositorio público llamado `football-ai-predictor`.
3. Sube todos los archivos de esta carpeta, manteniendo `app.py`, `predictor.py` y `requirements.txt` en la raíz.
4. Entra en https://share.streamlit.io/ e inicia sesión con GitHub.
5. Selecciona el repositorio, rama `main` y archivo principal `app.py`.
6. Pulsa Deploy.

## Datos
La app incluye datos sintéticos para probar la interfaz. No son partidos reales ni deben utilizarse para apostar.
Para análisis real, descarga un CSV de resultados de https://www.football-data.co.uk/ y súbelo en la barra lateral.

Columnas mínimas: HomeTeam, AwayTeam, FTHG, FTAG.
Columnas opcionales: HC, AC, HY, AY.

## Limitaciones
- No descarga datos automáticamente.
- No ofrece cuotas en directo.
- No incluye lesiones, alineaciones ni xG.
- El modelo Poisson es una línea base estadística, no un sistema validado ni una garantía de acierto.
- Streamlit Community Cloud puede reiniciar la aplicación; esta versión no guarda historial permanente.

## Privacidad y seguridad
No introduzcas contraseñas ni claves API en archivos públicos.
