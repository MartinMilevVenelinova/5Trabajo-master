# -*- coding: utf-8 -*-
# src/02_limpieza.py
# OBJETIVO: dejar un CSV final con TODOS los TERRITORIOS tal y como vienen en el campo "Territorio"
# (Comunidad de Madrid, Municipios, Zonas Estadísticas, etc.), acotado a 2021–2024
# y con un límite equilibrado por año para no pasar de 50.000 filas.

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

PARO_FILE  = RAW / "evolucion-del-porcentaje-de-paro-registrado.csv"
CENSO_FILE = RAW / "poblacion-censada-por-edad-nacionalidad-y-sexo.-censo-anual.csv"

OUT_CLEAN_PARO     = PROC / "clean_paro.csv"
OUT_CLEAN_CENSO    = PROC / "clean_censo.csv"
OUT_FINAL_CSV      = PROC / "final_merged.csv"
OUT_FINAL_PARQUET  = PROC / "final_merged.parquet"

# --- parámetros clave ---
YEAR_MIN, YEAR_MAX = 2021, 2024   # rango de años a mantener (inclusive)
MAX_ROWS = 49_000                 # requisito del proyecto
RANDOM_SEED = 42

# Cupo equilibrado por año (para que 2021–2024 salgan sí o sí)
ENFORCE_PER_YEAR_CAP = True
PER_YEAR_CAP = 12_250   # 12.250 * 4 años ≈ 49.000

# ---------- lectura robusta ----------
def leer_csv(path: Path) -> pd.DataFrame:
    tries = [
        {"sep":";","enc":"latin-1"},
        {"sep":";","enc":"utf-8-sig"},
        {"sep":",","enc":"utf-8-sig"},
        {"sep":",","enc":"latin-1"},
    ]
    last = None
    for t in tries:
        try:
            return pd.read_csv(path, sep=t["sep"], encoding=t["enc"], engine="python")
        except Exception as e:
            last = e
            continue
    raise last

def a_numero(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(r"\.", "", regex=True)  # quita miles
    x = x.str.replace(r"\s", "", regex=True)
    x = x.str.replace("%", "", regex=False)
    x = x.str.replace(",", ".", regex=False)  # coma -> punto
    return pd.to_numeric(x, errors="coerce")

def a_anio(s: pd.Series) -> pd.Series:
    y = s.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(y, errors="coerce")

def normalizar_sexo(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    mapa = {"h":"M","hombre":"M","varon":"M","varón":"M","male":"M","m":"M",
            "mujer":"F","f":"F","female":"F","total":np.nan,"ambos":np.nan,"todos":np.nan}
    return x.map(mapa).fillna(x.replace({"1":"M","2":"F"}))

def normalizar_nacionalidad(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    return x.replace({
        r"^espa.*":"espanola",
        r"^nacional.*":"espanola",
        r"^extranj.*":"extranjera",
        r"^no\s?espa.*":"extranjera",
        r"^total$": np.nan
    }, regex=True)

def normalizar_edad(s: pd.Series) -> pd.DataFrame:
    x = s.astype(str).str.strip().str.lower()
    x = (x.str.replace("de ","",regex=False)
           .str.replace("años","",regex=False)
           .str.replace("–","-",regex=False)
           .str.replace("—","-",regex=False)
           .str.replace(r"\s*a\s*","-",regex=True)
           .str.replace(" y mas","+")
           .str.replace(" y más","+"))
    x = x.replace({"total": np.nan})
    plus = x.str.extract(r"^(\d{1,3})\+$", expand=False)
    rng  = x.str.extract(r"^(\d{1,3})\s*-\s*(\d{1,3})$", expand=True)
    amin = pd.to_numeric(rng[0], errors="coerce")
    amax = pd.to_numeric(rng[1], errors="coerce")
    plus_min = pd.to_numeric(plus, errors="coerce")
    plus_max = pd.Series(np.where(plus.notna(), 120, np.nan), index=plus.index)
    amin = amin.fillna(plus_min); amax = amax.fillna(plus_max)
    single = pd.to_numeric(x, errors="coerce")
    amin = amin.fillna(single); amax = amax.fillna(single)
    return pd.DataFrame({"age_min": amin, "age_max": amax, "age_group": s})

# ---------- PARO (sin filtrar tipo, conservamos TODOS los territorios) ----------
def limpiar_paro(df0: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df0.columns}
    def pick(*cands):
        for c in cands:
            if c in df0.columns: return c
            if c.lower() in cols: return cols[c.lower()]
        return None

    c_anio = pick("Año")
    c_tipo = pick("Tipo territorio")
    c_terr = pick("Territorio")
    c_val  = pick("Valor")  # % de paro

    if not all([c_anio, c_tipo, c_terr, c_val]):
        raise ValueError(f"PARO: columnas esperadas no encontradas. Tengo: {list(df0.columns)}")

    df = df0.copy()
    df["year"] = a_anio(df[c_anio])
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    if df.empty:
        raise ValueError(f"PARO: sin filas en el rango {YEAR_MIN}-{YEAR_MAX}.")

    df["date"]            = pd.to_datetime(df["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    df["territory"]       = df[c_terr].astype(str).str.strip()          # NOMBRE EXACTO de territorio
    df["territory_type"]  = df[c_tipo].astype(str).str.strip()          # Conservamos el tipo (CCAA, Municipios, Zonas…)
    df["unemp_rate"]      = a_numero(df[c_val])

    df = df.dropna(subset=["year","unemp_rate","territory"]).drop_duplicates()
    df = df.sort_values(["territory_type","territory","year"])
    df["unemp_rate_yoy"] = df.groupby(["territory_type","territory"])["unemp_rate"].pct_change(1) * 100
    df["unemp_rate_pp"]  = df.groupby(["territory_type","territory"])["unemp_rate"].diff(1)

    out_cols = ["year","date","territory_type","territory","unemp_rate","unemp_rate_yoy","unemp_rate_pp"]
    return df[out_cols].reset_index(drop=True)

# ---------- CENSO (sin filtrar tipo, conservamos TODOS los territorios) ----------
def limpiar_censo(df0: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df0.columns}
    def pick(*cands):
        for c in cands:
            if c in df0.columns: return c
            if c.lower() in cols: return cols[c.lower()]
        return None

    c_anio = pick("Año")
    c_tipo = pick("Tipo territorio")
    c_terr = pick("Territorio")
    c_val  = pick("Valor")  # población
    c_sex  = pick("Sexo")
    c_nat  = pick("Nacionalidad")
    c_age  = pick("Edad", "Grupo edad", "Grupo de edad", "Rango de edad")

    if not all([c_anio, c_tipo, c_terr, c_val]):
        raise ValueError(f"CENSO: columnas esperadas no encontradas. Tengo: {list(df0.columns)}")

    df = df0.copy()
    df["year"] = a_anio(df[c_anio])
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    if df.empty:
        raise ValueError(f"CENSO: sin filas en el rango {YEAR_MIN}-{YEAR_MAX}.")

    df["date"]           = pd.to_datetime(df["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
    df["territory"]      = df[c_terr].astype(str).str.strip()
    df["territory_type"] = df[c_tipo].astype(str).str.strip()
    df["population"]     = a_numero(df[c_val])

    df["sex"] = normalizar_sexo(df[c_sex]) if c_sex else np.nan
    df["nationality"] = normalizar_nacionalidad(df[c_nat]) if c_nat else np.nan
    if c_age: df = pd.concat([df, normalizar_edad(df[c_age])], axis=1)
    else:
        df["age_group"] = np.nan; df["age_min"] = np.nan; df["age_max"] = np.nan

    df = df.dropna(subset=["year","population","territory"])
    df = df[df["population"] >= 0].drop_duplicates().reset_index(drop=True)

    group_cols = ["year","date","territory_type","territory","sex","nationality","age_group","age_min","age_max"]
    df = df.groupby(group_cols, dropna=False, as_index=False).agg(population=("population","sum"))
    return df

# ---------- limitadores ----------
def limitar_por_cupo_anual(df: pd.DataFrame,
                           year_col: str = "anio",
                           per_year_cap: int = 12250,
                           seed: int = 42) -> pd.DataFrame:
    """
    Máximo 'per_year_cap' filas por año.
    Muestra estratificada por tipo_territorio y territorio dentro de cada año.
    """
    if year_col not in df.columns:
        return df

    out = []
    for y, g in df.groupby(year_col):
        if len(g) <= per_year_cap:
            out.append(g)
            continue

        # estratos: tipo_territorio + territorio (si existen)
        has_type = "tipo_territorio" in g.columns
        has_terr = "territorio" in g.columns

        if has_terr:
            frac = min(1.0, per_year_cap / len(g))
            keys = ["territorio"]
            if has_type:
                keys = ["tipo_territorio","territorio"]

            sampled = (g.groupby(keys, group_keys=False, dropna=False)
                         .apply(lambda s: s.sample(max(1, int(round(len(s)*frac))),
                                                   random_state=seed)))
            # ajuste fino
            if len(sampled) > per_year_cap:
                sampled = sampled.sample(n=per_year_cap, random_state=seed)
            elif len(sampled) < per_year_cap:
                rest = g.drop(sampled.index, errors="ignore")
                need = per_year_cap - len(sampled)
                if need > 0 and not rest.empty:
                    sampled = pd.concat([sampled,
                                         rest.sample(n=min(need, len(rest)), random_state=seed)],
                                        ignore_index=False)
            out.append(sampled)
        else:
            out.append(g.sample(n=per_year_cap, random_state=seed))

    sort_cols = [year_col] + (["tipo_territorio"] if "tipo_territorio" in df.columns else []) + (["territorio"] if "territorio" in df.columns else [])
    return (pd.concat(out, ignore_index=True)
              .sort_values(sort_cols)
              .reset_index(drop=True))

def limitar_filas_total(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows or "anio" not in df.columns:
        return df
    # segunda barrera: sample global manteniendo proporción por año
    frac = max_rows / len(df)
    sampled = (df.groupby("anio", dropna=False, group_keys=False)
                 .apply(lambda s: s.sample(max(1, int(round(len(s)*frac))), random_state=RANDOM_SEED)))
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_SEED)
    return sampled.reset_index(drop=True)

# ---------- unión ----------
def unir_paro_censo(paro: pd.DataFrame, censo: pd.DataFrame) -> pd.DataFrame:
    # clave de unión: year + territory (mismo nombre exacto)
    merged = censo.merge(paro, on=["year","territory"], how="left", suffixes=("_censo","_paro"))

    # si falta tasa, rellena con la media del año (suave) y recalcula diferencia
    if "unemp_rate" in merged.columns and merged["unemp_rate"].isna().any():
        merged["unemp_rate"] = merged.groupby("year")["unemp_rate"].transform(lambda s: s.fillna(s.mean()))
        merged = merged.sort_values(["territory","year"])
        merged["unemp_rate_pp"] = merged.groupby(["territory"])["unemp_rate"].diff(1)

    # elegir una sola columna de tipo_territorio (preferimos la del censo; si no, la del paro)
    merged["tipo_territorio"] = merged["territory_type_censo"].fillna(merged.get("territory_type_paro"))
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("territory_type_")], errors="ignore")

    # ordenar y renombrar a español
    primero = ["year","date_censo","tipo_territorio","territory","sex","nationality","age_group","age_min","age_max",
               "population","unemp_rate","unemp_rate_yoy","unemp_rate_pp"]
    primero = [c for c in primero if c in merged.columns]
    resto   = [c for c in merged.columns if c not in primero]
    merged  = merged[primero + resto]

    rename = {
        "year": "anio",
        "date_censo": "fecha",
        "territory": "territorio",
        "sex": "sexo",
        "nationality": "nacionalidad",
        "age_group": "grupo_edad",
        "population": "poblacion",
        "unemp_rate": "tasa_paro",
        "unemp_rate_yoy": "tasa_paro_var_interanual_pct",
        "unemp_rate_pp": "tasa_paro_cambio_pp",
        "tipo_territorio": "tipo_territorio"
    }
    merged = merged.rename(columns=rename)

    # limpiar "total" en categorías; quitar edad_min/max si no aportan
    if "sexo" in merged.columns:
        merged["sexo"] = merged["sexo"].replace({"total": np.nan, "todos": np.nan, "ambos": np.nan})
    if "grupo_edad" in merged.columns:
        merged["grupo_edad"] = merged["grupo_edad"].replace({"total": np.nan, "todos": np.nan})
    merged = merged.drop(columns=["age_min","age_max"], errors="ignore")

    # redondeo tasas
    for col in ["tasa_paro","tasa_paro_var_interanual_pct","tasa_paro_cambio_pp"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").round(2)

    # tipados básicos
    if "anio" in merged.columns:
        merged["anio"] = pd.to_numeric(merged["anio"], errors="coerce").astype("Int64")
    if "fecha" in merged.columns:
        merged["fecha"] = pd.to_datetime(merged["fecha"], errors="coerce")

    return merged

# ---------- main ----------
def main():
    paro_raw  = leer_csv(PARO_FILE)
    censo_raw = leer_csv(CENSO_FILE)

    paro  = limpiar_paro(paro_raw)
    censo = limpiar_censo(censo_raw)

    # guardados intermedios
    paro.to_csv(OUT_CLEAN_PARO, index=False, encoding="utf-8-sig")
    censo.to_csv(OUT_CLEAN_CENSO, index=False, encoding="utf-8-sig")

    final_df = unir_paro_censo(paro, censo)

    # equilibrar cupo por año para no perder 2021–2024
    if ENFORCE_PER_YEAR_CAP:
        before = len(final_df)
        final_df = limitar_por_cupo_anual(final_df, year_col="anio", per_year_cap=PER_YEAR_CAP, seed=RANDOM_SEED)
        print(f"[INFO] Cupo anual aplicado: {before:,} -> {len(final_df):,} (≈ {PER_YEAR_CAP} por año)")

    # si aún supera MAX_ROWS, segunda barrera
    if len(final_df) > MAX_ROWS:
        before2 = len(final_df)
        final_df = limitar_filas_total(final_df, MAX_ROWS)
        print(f"[INFO] Limitador total aplicado: {before2:,} -> {len(final_df):,}")

    # export final
    final_df.to_csv(OUT_FINAL_CSV, index=False, encoding="utf-8-sig")
    try:
        final_df.to_parquet(OUT_FINAL_PARQUET, index=False)
        print(f"[OK] Parquet -> {OUT_FINAL_PARQUET}")
    except Exception:
        print("[AVISO] No se pudo exportar Parquet (pyarrow).")

    print(f"[OK] clean_paro: {paro.shape} -> {OUT_CLEAN_PARO}")
    print(f"[OK] clean_censo: {censo.shape} -> {OUT_CLEAN_CENSO}")
    print(f"[OK] final_merged: {final_df.shape} -> {OUT_FINAL_CSV}")
    print("Años únicos en final:", sorted(pd.to_numeric(final_df['anio'], errors='coerce').dropna().unique().tolist()))
    print("Tipos territorio (final):", final_df["tipo_territorio"].dropna().unique()[:10])
    print("Ejemplo territorios:", final_df["territorio"].dropna().astype(str).head(12).tolist())

if __name__ == "__main__":
    main()