# %%
from itertools import product
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ARCHIVO = BASE_DIR / "database" / "Datos_EMBI.xlsx"
SALIDA = SCRIPT_DIR / "resultados_embi"
DIA_CORTE = 1
ORDEN = (1, 1, 1)

if not ARCHIVO.is_file():
    raise FileNotFoundError(
        f"No se encontró el archivo de datos: {ARCHIVO}. "
        "Verifica que database/Datos_EMBI.xlsx esté dentro del proyecto."
    )

SALIDA.mkdir(parents=True, exist_ok=True)

# Cargar EMBI diario y obtener el último dato disponible hasta cada corte trimestral.
raw = pd.read_excel(ARCHIVO, sheet_name="Diarias", header=1)
if len(raw.columns) < 2:
    raise ValueError("La hoja 'Diarias' debe tener al menos una columna de fecha y otra de spread.")

col_fecha, col_spread = raw.columns[:2]
raw[col_fecha] = pd.to_datetime(raw[col_fecha], errors="coerce")
spread = pd.to_numeric(raw[col_spread], errors="coerce")
validas = raw[col_fecha].notna() & spread.notna()
if not validas.any():
    raise ValueError("No se encontraron filas diarias con fecha y spread válidos.")
if not validas.all():
    print(f"Filas omitidas por fecha o spread inválido/vacío: {(~validas).sum()}")

diaria = pd.Series(
    spread.loc[validas].to_numpy(),
    index=pd.DatetimeIndex(raw.loc[validas, col_fecha]),
    name="spread",
).sort_index()

cortes, valores = [], []
for anio in range(diaria.index.min().year, diaria.index.max().year + 1):
    for mes in (3, 6, 9, 12):
        corte = pd.Timestamp(anio, mes, DIA_CORTE)
        if diaria.index.min() <= corte <= diaria.index.max():
            valores_hasta_corte = diaria.loc[:corte]
            if not valores_hasta_corte.empty:
                cortes.append(corte)
                valores.append(valores_hasta_corte.iloc[-1])

if not cortes:
    raise ValueError("No se pudieron construir observaciones trimestrales desde los datos diarios.")

fechas = pd.DatetimeIndex(cortes)
y = pd.Series(valores, index=fechas, name="spread")
if (y <= 0).any():
    raise ValueError("El spread debe ser positivo para calcular su logaritmo.")

ly = np.log(y)
dly = ly.diff().dropna()
dy = y.diff().dropna()
print(f"Diaria: {diaria.index.min():%d/%m/%Y} a {diaria.index.max():%d/%m/%Y}")
print(
    f"Trimestral (corte día {DIA_CORTE}): {len(y)} obs., "
    f"{fechas[0]:%d/%m/%Y} a {fechas[-1]:%d/%m/%Y}"
)
print(f"Fecha convertida a {raw[col_fecha].dtype}; spread original: {raw[col_spread].dtype}")


# %%
# Transformaciones y comprobación de varianza.
fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
ax[0].plot(fechas, y, color="#1F4E78", lw=1.8)
ax[0].set_title("ANTES: Spread EMBIG Colombia (pbs), trimestral")
ax[0].set_ylabel("pbs")
ax[1].plot(fechas, ly, color="#2E7D32", lw=1.8)
ax[1].set_title("Transformación 1: ln(spread)")
ax[1].set_ylabel("ln")
ax[2].plot(fechas[1:], dly, color="#C0392B", lw=1.5)
ax[2].axhline(0, color="gray", lw=0.8)
ax[2].set_title("DESPUÉS: Δln(spread)")
ax[2].set_ylabel("Δln")
for axis in ax:
    axis.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(SALIDA / "fig1_transformacion.png", dpi=150)


# %%
def arch_ar1(serie, rezagos):
    valores_serie = np.asarray(serie)
    residuos = sm.OLS(
        valores_serie[1:],
        sm.add_constant(valores_serie[:-1]),
    ).fit().resid
    return het_arch(residuos, nlags=rezagos)[1]


print("\n--- Varianza constante, antes y después ---")
for nombre, serie in [("Nivel", y), ("Dif. nivel", dy), ("ln", ly), ("Dif. ln", dly)]:
    serie = serie.reset_index(drop=True)
    mitad = len(serie) // 2
    print(
        f"{nombre:8s} var1={serie[:mitad].var():.4f} var2={serie[mitad:].var():.4f} "
        f"Levene p={stats.levene(serie[:mitad], serie[mitad:]).pvalue:.4f} | "
        f"ARCH(1) p={arch_ar1(serie, 1):.4f} ARCH(4) p={arch_ar1(serie, 4):.4f}"
    )

# %%
# Identificación: pruebas de raíz unitaria y FAC/FACP.
print("\n=== ETAPA 1: IDENTIFICACIÓN ===")


def imprimir_adf(serie, regresion, nombre):
    resultado = adfuller(serie, regression=regresion, autolag="AIC")
    print(
        f"ADF  {nombre:22s} est={resultado[0]:.3f} "
        f"p={resultado[1]:.4f} rezagos={resultado[2]} "
        f"crit5%={resultado[4]['5%']:.3f}"
    )


def imprimir_kpss(serie, regresion, nombre):
    rezagos = int(4 * (len(serie) / 100) ** 0.25)
    resultado = kpss(serie, regression=regresion, nlags=max(1, rezagos))
    print(
        f"KPSS {nombre:22s} est={resultado[0]:.3f} "
        f"p={resultado[1]:.4f} crit5%={resultado[3]['5%']}"
    )


imprimir_adf(ly, "c", "ln (constante)")
imprimir_kpss(ly, "c", "ln (nivel)")
imprimir_adf(ly, "ct", "ln (const.+tend.)")
imprimir_kpss(ly, "ct", "ln (tendencia)")
imprimir_adf(dly, "c", "Dif. ln (constante)")
imprimir_kpss(dly, "c", "Dif. ln (nivel)")

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
plot_acf(ly, lags=24, ax=ax[0, 0], title="FAC – ln(spread)")
plot_pacf(ly, lags=24, ax=ax[1, 0], method="ywm", title="FACP – ln(spread)")
plot_acf(dly, lags=24, ax=ax[0, 1], title="FAC – Δln(spread)")
plot_pacf(dly, lags=24, ax=ax[1, 1], method="ywm", title="FACP – Δln(spread)")
plt.tight_layout()
plt.savefig(SALIDA / "fig2_FAC_FACP.png", dpi=150)

# %%
# Estimación: comparación por AIC y ajuste del modelo elegido.
print("\n=== ETAPA 2: ESTIMACIÓN ===")
resultados_modelos = []
for p, q in product(range(4), range(4)):
    nombre_modelo = f"ARIMA({p},1,{q})"
    try:
        ajuste = ARIMA(ly, order=(p, 1, q)).fit()
    except (ValueError, np.linalg.LinAlgError) as error:
        warnings.warn(f"No se pudo ajustar {nombre_modelo}: {error}", RuntimeWarning)
        continue
    resultados_modelos.append((nombre_modelo, ajuste.aic, ajuste.bic))

if not resultados_modelos:
    raise RuntimeError("No se pudo ajustar ningún modelo ARIMA candidato.")

tabla = pd.DataFrame(resultados_modelos, columns=["Modelo", "AIC", "BIC"]).sort_values("AIC")
print("Comparación (referencia, d=1, sin constante):")
print(tabla.head(6).round(2).to_string(index=False))

modelo = ARIMA(ly, order=ORDEN).fit()
print(f"\nModelo elegido: ARIMA{ORDEN}")
print(modelo.summary().tables[1])
print(f"AIC={modelo.aic:.2f}  BIC={modelo.bic:.2f}")
print("Módulo raíz AR:", np.abs(modelo.arroots).round(3), " raíz MA:", np.abs(modelo.maroots).round(3))

# %%
# Diagnóstico de residuos.
print("\n=== ETAPA 3: DIAGNÓSTICO ===")
res = modelo.resid[1:]
ljung_box = acorr_ljungbox(res, lags=[4, 8, 12], model_df=ORDEN[0] + ORDEN[2])
print("Ljung-Box p (4,8,12):", ljung_box["lb_pvalue"].round(3).tolist())
print("ARCH-LM(4) p:", round(het_arch(res, nlags=4)[1], 3))
jarque_bera = stats.jarque_bera(res)
print(
    f"Jarque-Bera p: {jarque_bera[1]:.2e} | "
    f"asimetría={stats.skew(res):.2f} curtosis={stats.kurtosis(res, fisher=False):.2f}"
)

fig, ax = plt.subplots(2, 1, figsize=(10, 7))
plot_acf(res, lags=24, ax=ax[0], title=f"FAC – residuos ARIMA{ORDEN}")
plot_pacf(res, lags=24, ax=ax[1], method="ywm", title=f"FACP – residuos ARIMA{ORDEN}")
plt.tight_layout()
plt.savefig(SALIDA / "fig3_residuos.png", dpi=150)

# %%
# Pronóstico a dos trimestres e intervalos de confianza.
print("\n=== ETAPA 4: PRONÓSTICO ===")
pronostico = modelo.get_forecast(2)
punto = np.exp(pronostico.predicted_mean.to_numpy())
ic95 = np.exp(pronostico.conf_int(alpha=0.05).to_numpy())
ic90 = np.exp(pronostico.conf_int(alpha=0.10).to_numpy())
futuro = [
    fechas[-1] + pd.DateOffset(months=3),
    fechas[-1] + pd.DateOffset(months=6),
]
tabla_pronostico = pd.DataFrame(
    {
        "Fecha": [f"{fecha:%d/%m/%Y}" for fecha in futuro],
        "Punto": punto,
        "IC95_inf": ic95[:, 0],
        "IC95_sup": ic95[:, 1],
        "IC90_inf": ic90[:, 0],
        "IC90_sup": ic90[:, 1],
    }
)
print(tabla_pronostico.round(1).to_string(index=False))
print(f"Último dato observado ({fechas[-1]:%d/%m/%Y}): {y.iloc[-1]:.0f} pbs")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fechas[-24:], y[-24:], marker="o", ms=3, color="#1F4E78", label="Observado")
ax.plot(
    [fechas[-1], *futuro],
    [y.iloc[-1], *punto],
    "--o",
    color="#C0392B",
    label="Pronóstico",
)
ax.fill_between(futuro, ic95[:, 0], ic95[:, 1], color="#C0392B", alpha=0.15, label="IC 95%")
ax.fill_between(futuro, ic90[:, 0], ic90[:, 1], color="#C0392B", alpha=0.30, label="IC 90%")
for fecha, valor in zip(futuro, punto):
    ax.annotate(
        f"{valor:.1f}",
        xy=(fecha, valor),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#9C2F25",
        fontsize=7,
        fontweight="bold",
        zorder=10,
    )
ax.set_title(f"Pronóstico Spread EMBIG Colombia – ARIMA{ORDEN} sobre ln")
ax.set_ylabel("pbs")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(SALIDA / "fig4_pronostico.png", dpi=150)
plt.show()

# %%
