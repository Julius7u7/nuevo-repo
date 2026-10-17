# Análisis Box-Jenkins del EMBIG Colombia

El script `box_jenkins_embi.py` carga el Excel `base de datos/Datos_EMBI.xlsx` desde la
raíz del proyecto, convierte la serie diaria a trimestral y ejecuta el análisis
ARIMA. No depende de la carpeta desde la que se inicia Python.

Desde la raíz del proyecto, ejecútalo con:

```powershell
python code\box_jenkins_embi.py
```

El script genera las gráficas en `code/resultados_embi/`. Necesita `pandas`,
`openpyxl`, `matplotlib`, `numpy`, `scipy` y `statsmodels`; pueden instalarse con:

```powershell
python -m pip install pandas openpyxl matplotlib numpy scipy statsmodels
```
