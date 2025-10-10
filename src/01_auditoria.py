# -*- coding: utf-8 -*-
# src/01_auditoria.py
# Imprime por consola y además guarda tablas de resumen en reports/tables/

import pandas as pd
from pathlib import Path
import re

# === RUTAS (ajusta nombres si los tuyos son distintos) ===
RAW = Path("data/raw")
REPORTS = Path("reports"); TABLES = REPORTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

PARO_PATH  = RAW / "evolucion-del-porcentaje-de-paro-registrado.csv"
CENSO_CSV  = RAW / "poblacion-censada-por-edad-nacionalidad-y-sexo.-censo-anual.csv"   # si lo tienes en CSV
CENSO_XLSX = RAW / "poblacion-censada-por-edad-nacionalidad-y-sexo.-censo-anual.xlsx"  # si lo tuvieras en Excel
# =========================================================

def leer_csv_flexible(path: Path):
    seps = [",",";","\t","|"]
    encs = ["utf-8","latin-1","cp1252"]
    last = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1]==1 and df.iloc[:,0].astype(str).str.count(sep).mean()>0.5:
                    continue
                return df, sep, enc
            except Exception as e:
                last = e
    raise last

def leer_excel_todas_hojas(path: Path):
    # Requiere: pip install openpyxl
    xls = pd.ExcelFile(path)
    df = pd.concat([pd.read_excel(xls, sh) for sh in xls.sheet_names], ignore_index=True)
    return df, xls.sheet_names

def pista_columna(col: str):
    name = col.lower()
    if re.search(r"año|anio|ano|year|fecha|period", name): return "posible año/fecha"
    if re.search(r"paro|desemple", name): return "posible tasa de paro"
    if re.search(r"poblac|habit|censo|padron", name): return "posible población"
    if re.search(r"provincia|comunidad|ccaa|municipio|territ|pais|país", name): return "posible territorio"
    if re.search(r"sexo|genero|género|hombre|mujer", name): return "posible sexo"
    if re.search(r"edad|grupo", name): return "posible edad"
    if re.search(r"nacionalidad|extranj|esp", name): return "posible nacionalidad"
    return ""

def perfilar(df: pd.DataFrame, nombre: str):
    # resumen general
    resumen = {
        "dataset": nombre,
        "filas": len(df),
        "columnas": df.shape[1],
        "duplicados": int(df.duplicated().sum())
    }
    # perfil por columna
    filas = []
    for c in df.columns:
        s = df[c]
        filas.append({
            "dataset": nombre,
            "columna": c,
            "tipo": str(s.dtype),
            "nulos": int(s.isna().sum()),
            "%_nulos": round(s.isna().mean()*100, 2),
            "unicos": int(s.nunique(dropna=True)),
            "pista": pista_columna(c)
        })
    return pd.DataFrame([resumen]), pd.DataFrame(filas)

def imprimir_quickview(df: pd.DataFrame, titulo: str):
    print(f"\n=== {titulo} ===")
    print("Shape:", df.shape)
    print("\nTipos:\n", df.dtypes)
    print("\nPrimeras filas:\n", df.head(3))
    print("\n% Nulos (Top 15):\n", (df.isna().mean().sort_values(ascending=False).head(15)*100).round(2))

def main():
    # ------- PARO (CSV) -------
    if PARO_PATH.exists():
        paro, sep_p, enc_p = leer_csv_flexible(PARO_PATH)
        imprimir_quickview(paro, f"PARO ({PARO_PATH.name}) | sep='{sep_p}' enc='{enc_p}'")
        ov_p, pr_p = perfilar(paro, "paro")
    else:
        print(f"[AVISO] No encuentro: {PARO_PATH}")
        ov_p = pr_p = pd.DataFrame()

    # ------- CENSO (CSV o XLSX) -------
    censo = None
    if CENSO_CSV.exists():
        censo, sep_c, enc_c = leer_csv_flexible(CENSO_CSV)
        imprimir_quickview(censo, f"CENSO ({CENSO_CSV.name}) | sep='{sep_c}' enc='{enc_c}'")
    elif CENSO_XLSX.exists():
        try:
            censo, hojas = leer_excel_todas_hojas(CENSO_XLSX)
            print(f"\nHojas Excel: {hojas}")
            imprimir_quickview(censo, f"CENSO ({CENSO_XLSX.name}) [todas las hojas unidas]")
        except Exception as e:
            print("[ERROR] Para Excel instala openpyxl: pip install openpyxl")
            raise e
    else:
        print(f"[AVISO] No encuentro: {CENSO_CSV} ni {CENSO_XLSX}")

    if censo is not None:
        ov_c, pr_c = perfilar(censo, "censo")
    else:
        ov_c = pr_c = pd.DataFrame()

    # ------- GUARDAR TABLAS DE AUDITORÍA -------
    resumen = pd.concat([ov_p, ov_c], ignore_index=True)
    perfil  = pd.concat([pr_p, pr_c], ignore_index=True)

    if not resumen.empty:
        resumen.to_csv(TABLES / "00_auditoria_resumen_datasets.csv", index=False, encoding="utf-8-sig")
    if not perfil.empty:
        perfil.to_csv(TABLES / "01_perfilado_columnas.csv", index=False, encoding="utf-8-sig")

    # (Opcional) guardar muestras de 50 filas para revisión rápida
    if PARO_PATH.exists():
        paro.head(50).to_csv(TABLES / "00_sample_paro_50.csv", index=False, encoding="utf-8-sig")
    if censo is not None:
        censo.head(50).to_csv(TABLES / "00_sample_censo_50.csv", index=False, encoding="utf-8-sig")

    print("\n== ARCHIVOS CREADOS EN reports/tables/ ==")
    print("- 00_auditoria_resumen_datasets.csv")
    print("- 01_perfilado_columnas.csv")
    print("- 00_sample_paro_50.csv (opcional)")
    print("- 00_sample_censo_50.csv (opcional)")
    print("\nAuditoría lista.")

if __name__ == "__main__":
    main()