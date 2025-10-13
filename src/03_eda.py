# -*- coding: utf-8 -*-
# src/03_eda.py
# EDA automático SIN "totales" en poblaciones y con tasas deduplicadas.
# Salidas: reports/tables/*.csv y reports/plots/*.png

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(".")
PROC = BASE / "data" / "processed"
REPORTS = BASE / "reports"
TABLES  = REPORTS / "tables"
PLOTS   = REPORTS / "plots"
for d in [TABLES, PLOTS]:
    d.mkdir(parents=True, exist_ok=True)

FINAL = PROC / "final_merged.csv"
YEAR_MIN, YEAR_MAX = 2021, 2024

def load_final():
    if not FINAL.exists():
        raise FileNotFoundError("No encuentro data/processed/final_merged.csv")
    df = pd.read_csv(FINAL, encoding="utf-8")
    return df

df = load_final()

# Tipados
if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
if "anio" in df.columns:
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")

for c in ["poblacion","tasa_paro","tasa_paro_var_interanual_pct","tasa_paro_cambio_pp"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["tipo_territorio","territorio","sexo","nacionalidad","grupo_edad"]:
    if c in df.columns and df[c].dtype == "object":
        df[c] = df[c].str.strip()

# Rango de años
if "anio" in df.columns:
    df = df[(df["anio"] >= YEAR_MIN) & (df["anio"] <= YEAR_MAX)].copy()

# ============= Claves: quitar "totales" en población, deduplicar tasas =============

# 1) TASAS DEDUP: tasas únicas por (anio, tipo_territorio, territorio)
has_type = "tipo_territorio" in df.columns
rate_cols = ["anio","territorio","tasa_paro"]
if has_type:
    rate_cols.insert(1, "tipo_territorio")

df_rates = pd.DataFrame(columns=rate_cols)
if {"tasa_paro","territorio","anio"}.issubset(df.columns):
    # Dropna y deduplicar
    base = df[rate_cols].dropna(subset=["tasa_paro"]).drop_duplicates()
    df_rates = base.copy()

# 2) POBLACIÓN DETALLADA: excluir filas "totales" (que en tu limpieza quedaron como NaN)
#    - Mantener solo filas con sexo NO NULO
#    - Si existe grupo_edad, también NO NULO
#    - Si existe nacionalidad, también NO NULO
pop_filters = []
if "sexo" in df.columns:
    pop_filters.append(df["sexo"].notna())
if "grupo_edad" in df.columns:
    pop_filters.append(df["grupo_edad"].notna())
if "nacionalidad" in df.columns:
    pop_filters.append(df["nacionalidad"].notna())

if pop_filters:
    mask = np.logical_and.reduce(pop_filters)
    df_pop = df[mask].copy()
else:
    df_pop = df.copy()  # por si faltaran columnas demográficas (no debería)

# ------------------ TABLAS BASE (global) ------------------
resumen = pd.DataFrame({
    "filas_totales_final": [len(df)],
    "filas_detalle_sin_totales": [len(df_pop)],
    "columnas": [df.shape[1]],
    "anio_min": [df["anio"].min() if "anio" in df else np.nan],
    "anio_max": [df["anio"].max() if "anio" in df else np.nan],
    "territorios_distintos": [df["territorio"].nunique(dropna=True) if "territorio" in df else np.nan],
    "tipos_territorio": [df["tipo_territorio"].value_counts().to_dict() if "tipo_territorio" in df else {}]
})
resumen.to_csv(TABLES / "eda_00_resumen.csv", index=False, encoding="utf-8-sig")

nulls = df.isna().mean().sort_values(ascending=False).rename("%_nulos").mul(100).round(2)
nulls.to_csv(TABLES / "eda_01_nulos_por_col.csv", header=True, encoding="utf-8-sig")

nums = df.select_dtypes(include=[np.number])
if not nums.empty:
    desc = nums.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    desc.to_csv(TABLES / "eda_02_descriptivos_numericos.csv", encoding="utf-8-sig")

# ------------------ Helpers de gráfico ------------------
def save_line(x, y, data, title, xlab, ylab, fname, integer_xticks=False):
    if data.empty or x not in data or y not in data:
        return
    plt.figure()
    d2 = data.dropna(subset=[x, y]).sort_values(x)
    plt.plot(d2[x], d2[y])
    if integer_xticks:
        try:
            xs = sorted(pd.to_numeric(d2[x], errors="coerce").dropna().astype(int).unique().tolist())
            plt.xticks(xs, rotation=0)
        except Exception:
            pass
    plt.title(title)
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=120); plt.close()

def save_bar(x, y, data, title, xlab, ylab, fname, rotate=False):
    if data.empty or x not in data or y not in data:
        return
    plt.figure()
    d2 = data.dropna(subset=[x, y])
    plt.bar(d2[x], d2[y])
    plt.title(title)
    plt.xlabel(xlab); plt.ylabel(ylab)
    if rotate:
        plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=120); plt.close()

def save_hist(col, data, bins, title, xlab, fname):
    if data.empty or col not in data:
        return
    plt.figure()
    plt.hist(data[col].dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlab); plt.ylabel("frecuencia")
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=120); plt.close()

# ------------------ EDA por tipo de territorio ------------------
if "tipo_territorio" in df.columns:
    tipos = df["tipo_territorio"].dropna().unique().tolist()
else:
    tipos = [None]

for t in tipos:
    if t is None:
        sub_rates = df_rates.copy()
        sub_pop   = df_pop.copy()
        tipo_slug = "todos"
        tipo_titulo = "Todos los territorios"
    else:
        if has_type and not df_rates.empty:
            sub_rates = df_rates[df_rates["tipo_territorio"] == t].copy()
        else:
            sub_rates = df_rates.copy()

        if has_type:
            sub_pop = df_pop[df_pop["tipo_territorio"] == t].copy()
        else:
            sub_pop = df_pop.copy()

        tipo_slug = "".join(ch if ch.isalnum() else "_" for ch in str(t)).strip("_").lower()
        tipo_titulo = str(t)

    # ===== Evolución 2021–2024: tasa de paro media (USANDO df_rates DEDUP) =====
    evol = pd.DataFrame()
    if not sub_rates.empty and {"anio","tasa_paro"}.issubset(sub_rates.columns):
        evol = (sub_rates.groupby("anio", as_index=False)
                          .agg(tasa_paro=("tasa_paro","mean")))
        evol.to_csv(TABLES / f"{tipo_slug}_evol_tasa_paro_media.csv", index=False, encoding="utf-8-sig")
        save_line("anio","tasa_paro", evol,
                  f"{tipo_titulo} · Tasa de paro media por año (2021–2024)",
                  "Año", "Tasa de paro (%)",
                  f"{tipo_slug}_01_linea_tasa_paro_anual.png", integer_xticks=True)

    # ===== Último año =====
    if "anio" in sub_rates.columns and not sub_rates.empty:
        last_year = int(sub_rates["anio"].max())
    elif "anio" in sub_pop.columns and not sub_pop.empty:
        last_year = int(sub_pop["anio"].max())
    else:
        last_year = None

    # Subconjuntos último año
    rates_last = sub_rates[sub_rates["anio"] == last_year].copy() if last_year is not None else pd.DataFrame()
    pop_last   = sub_pop[sub_pop["anio"] == last_year].copy() if last_year is not None else pd.DataFrame()

    # ===== ¿Varía la tasa entre territorios? (con df_rates) =====
    varia_tasa = False
    if not rates_last.empty and {"territorio","tasa_paro"}.issubset(rates_last.columns):
        tt = (rates_last.groupby("territorio", as_index=False)
                        .agg(tasa_paro=("tasa_paro","mean")))
        varia_tasa = tt["tasa_paro"].nunique(dropna=True) > 1

    # ===== Top 15 =====
    if varia_tasa and not rates_last.empty:
        top_unemp = (rates_last.groupby("territorio", as_index=False)
                                .agg(tasa_paro=("tasa_paro","mean"))
                                .sort_values("tasa_paro", ascending=False)
                                .head(15))
        top_unemp.to_csv(TABLES / f"{tipo_slug}_top15_tasa_paro_{last_year}.csv", index=False, encoding="utf-8-sig")
        save_bar("territorio","tasa_paro", top_unemp,
                 f"{tipo_titulo} · Top 15 por tasa de paro ({last_year})",
                 "Territorio","Tasa de paro (%)",
                 f"{tipo_slug}_02_top15_tasa_paro.png", rotate=True)
    else:
        # Si la tasa no varía, hacemos top por población usando SOLO detalle (sin totales)
        if not pop_last.empty and {"territorio","poblacion"}.issubset(pop_last.columns):
            top_pop = (pop_last.groupby("territorio", as_index=False)
                                .agg(poblacion=("poblacion","sum"))
                                .sort_values("poblacion", ascending=False)
                                .head(15))
            top_pop.to_csv(TABLES / f"{tipo_slug}_top15_poblacion_{last_year}.csv",
                           index=False, encoding="utf-8-sig")
            save_bar("territorio","poblacion", top_pop,
                     f"{tipo_titulo} · Top 15 por población ({last_year})",
                     "Territorio","Población",
                     f"{tipo_slug}_02_top15_poblacion.png", rotate=True)

    # ===== Histograma de tasas (último año) =====
    if not rates_last.empty and "tasa_paro" in rates_last.columns and varia_tasa:
        save_hist("tasa_paro", rates_last, 25,
                  f"{tipo_titulo} · Distribución de tasa de paro ({last_year})",
                  "Tasa de paro (%)",
                  f"{tipo_slug}_03_hist_tasa_paro_{last_year}.png")

    # ===== Población por edad y sexo (último año, sin totales) =====
    if not pop_last.empty and {"grupo_edad","sexo","poblacion"}.issubset(pop_last.columns):
        pes = (pop_last.groupby(["grupo_edad","sexo"], as_index=False)
                        .agg(poblacion=("poblacion","sum"))
                        .sort_values("grupo_edad"))
        pes.to_csv(TABLES / f"{tipo_slug}_poblacion_por_edad_y_sexo_{last_year}.csv",
                   index=False, encoding="utf-8-sig")

        pivot = pes.pivot_table(index="grupo_edad", columns="sexo", values="poblacion", aggfunc="sum").fillna(0)
        pivot = pivot.sort_index()
        plt.figure()
        bottom = None
        for col in pivot.columns:
            vals = pivot[col].values
            if bottom is None:
                plt.bar(pivot.index, vals, label=str(col))
                bottom = vals
            else:
                plt.bar(pivot.index, vals, bottom=bottom, label=str(col))
                bottom = bottom + vals
        plt.title(f"{tipo_titulo} · Población por grupo de edad y sexo ({last_year})")
        plt.xlabel("Grupo de edad"); plt.ylabel("Población")
        plt.xticks(rotation=60, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / f"{tipo_slug}_04_barras_apiladas_poblacion_edad_sexo.png", dpi=120)
        plt.close()

print("[OK] EDA completado (sin 'totales').")
print(f"Tablas en: {TABLES}")
print(f"Gráficos en: {PLOTS}")