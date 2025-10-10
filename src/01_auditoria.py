# -*- coding: utf-8 -*-
# src/01_auditoria.py

import pandas as pd
from pathlib import Path
import re

# === RUTAS (ajusta solo si tus archivos tienen otros nombres) ===
RAW = Path("data/raw")
REPORTS = Path("reports")
TABLES = REPORTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

PARO_FILE  = "evolucion-del-porcentaje-de-paro-registrado.csv"
CENSO_FILE = "poblacion-censada-por-edad-nacionalidad-y-sexo.-censo-anual.csv"