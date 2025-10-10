# -*- coding: utf-8 -*-
# src/02_limpieza.py
# Limpieza de PARO + CENSO, unión y export final (CSV/Parquet).
# Columnas finales en español: anio, fecha, territorio, sexo, nacionalidad, grupo_edad,
# poblacion, tasa_paro, tasa_paro_var_interanual_pct, tasa_paro_cambio_pp

import re
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIG ---
RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

PARO_FILE  = "evolucion-del-porcentaje-de-paro-registrado.csv"
CENSO_FILE = "poblacion-censada-por-edad-nacionalidad-y-sexo.-censo-anual.csv"

OUT_CLEAN_PARO     = PROC / "clean_paro.csv"
OUT_CLEAN_CENSO    = PROC / "clean_censo.csv"
OUT_FINAL_CSV      = PROC / "final_merged.csv"
OUT_FINAL_PARQUET  = PROC / "final_merged.parquet"

# Límite de filas para cumplir proyecto (< 50.000)
MAX_ROWS = 49_999
ROW_LIMIT_STRATEGY = "recent_years"   # "recent_years" | "stratified" | "random"
# ---------------------------------------------------------------

# ---- LECTURA ROBUSTA ----
def leer_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Detecta separador en la 1ª línea y prueba varios encodings.
    Si sale 1 sola columna, reintenta forzando ';', ',', '\\t', '|'.
    """
    candidatos_sep = [",", ";", "\t", "|"]
    candidatos_enc = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]

    last_err = None
    for enc in candidatos_enc:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                head = f.readline()
            counts = {sep: head.count(sep) for sep in candidatos_sep}
            sep_est = max(counts, key=counts.get)
            if counts[sep_est] == 0:
                sep_est = None  # que pandas infiera

            df = pd.read_csv(path, sep=sep_est, encoding=enc, engine="python")

            if df.shape[1] == 1:
                for sep_try in [";", ",", "\t", "|"]:
                    df2 = pd.read_csv(path, sep=sep_try, encoding=enc, engine="python")
                    if df2.shape[1] > 1:
                        return df2
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err

# ---- AYUDAS ----
def snake(nombre: str) -> str:
    nombre = nombre.strip().lower()
    nombre = re.sub(r"[^\w\s]", " ", nombre)
    nombre = re.sub(r"\s+", "_", nombre)
    return nombre

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake(c) for c in df.columns]
    return df

def adivina_col(df, patrones):
    for p in patrones:
        hit = [c for c in df.columns if re.search(p, c)]
        if hit:
            return hit[0]
    return None

def a_numero(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(r"\.", "", regex=True)   # quita miles
    x = x.str.replace(r"\s", "", regex=True)
    x = x.str.replace("%", "", regex=False)
    x = x.str.replace(",", ".", regex=False)   # coma -> punto
    return pd.to_numeric(x, errors="coerce")

def a_anio(s: pd.Series) -> pd.Series:
    y = s.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(y, errors="coerce")

def normalizar_territorio(s: pd.Series) -> pd.Series:
    import unicodedata
    def quitar_acentos(txt):
        txt = unicodedata.normalize("NFKD", str(txt))
        return "".join(c for c in txt if not unicodedata.combining(c))
    return s.astype(str).str.strip().str.lower().map(quitar_acentos)

def normalizar_sexo(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    mapa = {"h":"M","hombre":"M","varon":"M","varón":"M","male":"M","m":"M",
            "mujer":"F","f":"F","female":"F"}
    return x.map(mapa).fillna(x.replace({"1":"M","2":"F"}))

def normalizar_nacionalidad(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    return x.replace({
        r"^espa.*":"espanola",
        r"^nacional.*":"espanola",
        r"^extranj.*":"extranjera",
        r"^no\s?espa.*":"extranjera",
    }, regex=True)

def normalizar_edad(s: pd.Series) -> pd.DataFrame:
    # acepta "25-29", "De 30 a 34", "65 y más", "65+"
    x = s.astype(str).str.strip().str.lower()
    x = (x.str.replace("de ","",regex=False)
           .str.replace("años","",regex=False)
           .str.replace("–","-",regex=False)
           .str.replace("—","-",regex=False)
           .str.replace(r"\s*a\s*","-",regex=True)
           .str.replace(" y mas","+")
           .str.replace(" y más","+"))
    plus = x.str.extract(r"^(\d{1,3})\+$", expand=False)               # ej: "65+"
    rng  = x.str.extract(r"^(\d{1,3})\s*-\s*(\d{1,3})$", expand=True)  # ej: "25-29"

    amin = pd.to_numeric(rng[0], errors="coerce")
    amax = pd.to_numeric(rng[1], errors="coerce")

    plus_min = pd.to_numeric(plus, errors="coerce")
    plus_max = pd.Series(np.where(plus.notna(), 120, np.nan), index=plus.index)

    amin = amin.fillna(plus_min)
    amax = amax.fillna(plus_max)

    single = pd.to_numeric(x, errors="coerce")
    amin = amin.fillna(single)
    amax = amax.fillna(single)

    return pd.DataFrame({"age_min": amin, "age_max": amax, "age_group": s})

# -------- LIMPIEZA PARO --------
def limpiar_paro(df: pd.DataFrame) -> pd.DataFrame:
    df = normalizar_columnas(df)
    col_anio = adivina_col(df, [r"^anio$", r"^ano$", r"^año$", r"^year$", r"^fecha$", r"period"])
    col_geo  = adivina_col(df, [r"provincia", r"comunidad|ccaa|autonom", r"municipio", r"territ", r"pais|país"])
    col_paro = adivina_col(df, [r"^valor$", r"paro", r"porcentaje.*paro", r"tasa.*paro", r"%.*paro", r"desemple"])
    if col_anio is None or col_paro is None:
        print("PARO columnas detectadas:", list(df.columns))
        raise ValueError("PARO: falta columna de año/fecha o de paro (probando con 'Valor').")

    df["year"] = a_anio(df[col_anio])
    df["date"] = pd.to_datetime(df["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    df["territory"] = normalizar_territorio(df[col_geo]) if col_geo else "espana"
    df["unemp_rate"] = a_numero(df[col_paro])

    df = df.dropna(subset=["year","unemp_rate"]).drop_duplicates().reset_index(drop=True)
    df = df.sort_values(["territory","year"])
    df["unemp_rate_yoy"] = df.groupby("territory")["unemp_rate"].pct_change(1) * 100
    df["unemp_rate_pp"] = df.groupby("territory")["unemp_rate"].diff(1)

    return df[["year","date","territory","unemp_rate","unemp_rate_yoy","unemp_rate_pp"]]

# -------- LIMPIEZA CENSO --------
def limpiar_censo(df: pd.DataFrame) -> pd.DataFrame:
    df = normalizar_columnas(df)
    col_anio = adivina_col(df, [r"^anio$", r"^ano$", r"^año$", r"^year$", r"^fecha$", r"period"])
    col_geo  = adivina_col(df, [r"provincia", r"comunidad|ccaa|autonom", r"municipio", r"territ", r"pais|país"])
    col_sex  = adivina_col(df, [r"sexo|genero|género|male|female|hombre|mujer"])
    col_age  = adivina_col(df, [r"edad|grupo_edad|rango"])
    col_nat  = adivina_col(df, [r"nacionalidad|extranj|esp"])
    col_pop  = adivina_col(df, [r"^valor$", r"poblac|habit|censo|padron"])
    if col_anio is None or col_pop is None:
        print("CENSO columnas detectadas:", list(df.columns))
        raise ValueError("CENSO: falta columna de año o población (probando con 'Valor').")

    df["year"] = a_anio(df[col_anio])
    df["date"] = pd.to_datetime(df["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    df["territory"] = normalizar_territorio(df[col_geo]) if col_geo else "espana"
    df["sex"] = normalizar_sexo(df[col_sex]) if col_sex else np.nan
    df["nationality"] = normalizar_nacionalidad(df[col_nat]) if col_nat else np.nan

    if col_age:
        df = pd.concat([df, normalizar_edad(df[col_age])], axis=1)
    else:
        df["age_group"] = np.nan; df["age_min"] = np.nan; df["age_max"] = np.nan

    df["population"] = a_numero(df[col_pop])

    df = df.dropna(subset=["year","population"])
    df = df[df["population"] >= 0].drop_duplicates().reset_index(drop=True)

    group_cols = ["year","territory","sex","nationality","age_group","age_min","age_max"]
    df = df.groupby(group_cols, dropna=False, as_index=False).agg(population=("population","sum"))
    return df

# -------- LIMITADOR DE FILAS (acepta nombres EN/ES) --------
def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def limitar_filas(df: pd.DataFrame, max_rows: int, strategy: str = "recent_years") -> pd.DataFrame:
    col_year = _pick_col(df, ["year", "anio"])
    col_terr = _pick_col(df, ["territory", "territorio"])
    col_sex  = _pick_col(df, ["sex", "sexo"])
    col_nat  = _pick_col(df, ["nationality", "nacionalidad"])
    col_ageg = _pick_col(df, ["age_group", "grupo_edad"])

    if col_year is None:
        return df.sample(n=min(max_rows, len(df)), random_state=42).reset_index(drop=True)

    if len(df) <= max_rows:
        return df

    print(f"[INFO] Limitar filas: {len(df):,} -> {max_rows:,} (estrategia: {strategy})")

    if strategy == "recent_years":
        keep, total = [], 0
        for y in sorted(df[col_year].dropna().unique(), reverse=True):
            part = df[df[col_year] == y]
            if total + len(part) <= max_rows:
                keep.append(part); total += len(part)
            else:
                remain = max_rows - total
                if remain > 0:
                    keep.append(part.sample(n=remain, random_state=42))
                break
        capped = pd.concat(keep, ignore_index=True)
        if col_terr:
            capped = capped.sort_values([col_year, col_terr], na_position="last")
        else:
            capped = capped.sort_values([col_year], na_position="last")
        return capped.reset_index(drop=True)

    elif strategy == "stratified":
        keys = [k for k in [col_year, col_terr, col_sex, col_nat, col_ageg] if k is not None]
        frac = min(1.0, max_rows / len(df))
        sampled = (df.groupby(keys, dropna=False, as_index=False)
                     .apply(lambda g: g.sample(max(1, int(round(len(g)*frac))), random_state=42))
                     .reset_index(drop=True))
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=42)
        return sampled.reset_index(drop=True)

    else:  # "random"
        return df.sample(n=max_rows, random_state=42).reset_index(drop=True)

# -------- UNION --------
def unir_paro_censo(paro: pd.DataFrame, censo: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["year"]
    if paro["territory"].nunique(dropna=True) > 1 or censo["territory"].nunique(dropna=True) > 1:
        join_cols = ["year","territory"]
    merged = censo.merge(paro, on=join_cols, how="left")

    if "unemp_rate" in merged.columns and merged["unemp_rate"].isna().any():
        merged["unemp_rate"] = merged.groupby("year")["unemp_rate"].transform(lambda s: s.fillna(s.mean()))
        merged = merged.sort_values(["territory","year"])
        merged["unemp_rate_pp"] = merged.groupby("territory")["unemp_rate"].diff(1)

    # Orden preliminar (EN)
    primero = ["year","date","territory","sex","nationality","age_group","age_min","age_max",
               "population","unemp_rate","unemp_rate_yoy","unemp_rate_pp"]
    primero = [c for c in primero if c in merged.columns]
    resto = [c for c in merged.columns if c not in primero]
    merged = merged[primero + resto]

    # Renombrar a español
    rename_map = {
        "year": "anio",
        "date": "fecha",
        "territory": "territorio",
        "sex": "sexo",
        "nationality": "nacionalidad",
        "age_group": "grupo_edad",
        "age_min": "edad_min",
        "age_max": "edad_max",
        "population": "poblacion",
        "unemp_rate": "tasa_paro",
        "unemp_rate_yoy": "tasa_paro_var_interanual_pct",
        "unemp_rate_pp": "tasa_paro_cambio_pp"
    }
    merged = merged.rename(columns=rename_map)

    # Normalizaciones simples
    if "sexo" in merged.columns:
        merged["sexo"] = merged["sexo"].replace({"total": np.nan, "todos": np.nan, "ambos": np.nan})
    if "grupo_edad" in merged.columns:
        merged["grupo_edad"] = merged["grupo_edad"].replace({"total": np.nan, "todos": np.nan})

    # Quitar edad_min y edad_max (no aportan en tu caso)
    merged = merged.drop(columns=["edad_min", "edad_max"], errors="ignore")

    # Redondear tasas a 2 decimales
    for col in ["tasa_paro", "tasa_paro_var_interanual_pct", "tasa_paro_cambio_pp"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").round(2)

    return merged

def main():
    paro_raw  = leer_csv_flexible(RAW / PARO_FILE)
    censo_raw = leer_csv_flexible(RAW / CENSO_FILE)

    paro  = limpiar_paro(paro_raw)
    censo = limpiar_censo(censo_raw)

    # Guardar limpios previos a la unión
    paro.to_csv(OUT_CLEAN_PARO, index=False, encoding="utf-8-sig")
    censo.to_csv(OUT_CLEAN_CENSO, index=False, encoding="utf-8-sig")

    final_df = unir_paro_censo(paro, censo)

    # Aplicar límite de filas
    final_df = limitar_filas(final_df, MAX_ROWS, ROW_LIMIT_STRATEGY)

    # Exportar final (español)
    final_df.to_csv(OUT_FINAL_CSV, index=False, encoding="utf-8-sig")
    try:
        final_df.to_parquet(OUT_FINAL_PARQUET, index=False)
        print(f"[OK] Parquet -> {OUT_FINAL_PARQUET}")
    except Exception:
        print("[AVISO] No se pudo exportar Parquet (¿tienes pyarrow instalado?).")

    print(f"[OK] clean_paro: {paro.shape} -> {OUT_CLEAN_PARO}")
    print(f"[OK] clean_censo: {censo.shape} -> {OUT_CLEAN_CENSO}")
    print(f"[OK] final_merged: {final_df.shape} -> {OUT_FINAL_CSV}")
    print(f"Columnas finales: {list(final_df.columns)}")
    print(f"Tope aplicado (< {MAX_ROWS:,} filas): {len(final_df) <= MAX_ROWS}")

if __name__ == "__main__":
    main()




