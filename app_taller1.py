# app.py
# ------------------------------------------------------------
# PCA (numéricas) + MCA (categóricas) con umbral del 80%
# - Mapping de diagnósticos OPCIONAL (sidebar o archivo local)
# - Reduce categorías raras/alta cardinalidad a "OTHER"
# - Muestra paso a paso y permite descargar el dataset final
# ------------------------------------------------------------

import io
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Intento seguro de importar prince.MCA
try:
    from prince import MCA
    PRINCE_OK = True
    PRINCE_ERR = ""
except Exception as e:
    PRINCE_OK = False
    PRINCE_ERR = str(e)

# ================= Utilidades =================

def drop_identifier_like_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Elimina SOLO identificadores reales (para no distorsionar PCA/MCA)."""
    ids_reales = {"encounter_id", "patient_nbr"}
    dropped = [c for c in df.columns if c in ids_reales]
    return df.drop(columns=dropped, errors="ignore"), dropped

def apply_diag_mapping(df_in: pd.DataFrame, ids_map: pd.DataFrame | None) -> pd.DataFrame:
    """Aplica mapeo de diagnósticos si el archivo está disponible (flexible con nombres de columnas)."""
    if ids_map is None:
        return df_in

    code_candidates = ["code", "diag_code", "icd9", "ICD9", "ICD9_CODE"]
    group_candidates = ["group", "category", "CCS", "ccs_group"]

    col_code = next((c for c in code_candidates if c in ids_map.columns), None)
    col_group = next((c for c in group_candidates if c in ids_map.columns), None)

    if col_code is None or col_group is None:
        st.warning("IDS_mapping.csv no tiene columnas esperadas (p. ej., 'code' y 'group'). Se omite el mapeo.")
        return df_in

    mapping = ids_map.set_index(col_code)[col_group].to_dict()
    df = df_in.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(df[col].astype(str))
    st.info("Mapping de diagnósticos aplicado.")
    return df

def split_numeric_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_df = df.select_dtypes(include=["number"]).copy()
    cat_df = df.select_dtypes(exclude=["number"]).copy()
    return num_df, cat_df

def fit_pca(num_df: pd.DataFrame, threshold: float = 0.80):
    """Estandariza, ajusta PCA y retorna PCs hasta alcanzar el umbral."""
    if num_df.shape[1] == 0:
        return pd.DataFrame(index=num_df.index), None, np.array([])

    X = num_df.fillna(num_df.mean(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)

    pca = PCA()
    Xp = pca.fit_transform(Xs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cum_var, threshold) + 1)

    pcs = pd.DataFrame(Xp[:, :ncomp], index=num_df.index,
                       columns=[f"PC{i+1}" for i in range(ncomp)])
    return pcs, pca, cum_var

def reduce_categorical_cardinality(cat: pd.DataFrame, max_modalities_per_col=50, rare_min_count=30):
    """
    - Convierte a str y rellena NaN -> 'missing'
    - Si una columna tiene demasiadas categorías, se quedan las top-(max-1) y el resto -> 'OTHER'
    - Además, toda categoría con conteo < rare_min_count -> 'OTHER'
    """
    cat = cat.fillna("missing").astype(str).copy()
    for c in cat.columns:
        vc = cat[c].value_counts(dropna=False)

        # Recortar a top categorías si hay demasiadas
        if len(vc) > max_modalities_per_col:
            keep = set(vc.index[:max_modalities_per_col - 1])  # deja espacio para OTHER
            cat[c] = np.where(cat[c].isin(keep), cat[c], "OTHER")
            vc = cat[c].value_counts(dropna=False)  # recomputa tras recorte

        # Enviar categorías muy raras a OTHER
        rare_vals = set(vc[vc < rare_min_count].index)
        if rare_vals:
            cat[c] = cat[c].where(~cat[c].isin(rare_vals), "OTHER")
    return cat

def safe_explained_inertia(mca):
    """Compatibilidad multi-versión de 'prince' para obtener la inercia explicada."""
    if hasattr(mca, "explained_inertia_"):
        return np.asarray(mca.explained_inertia_, dtype=float)
    if hasattr(mca, "eigenvalues_"):
        ev = np.asarray(mca.eigenvalues_, dtype=float).ravel()
        total = ev.sum() or 1.0
        return ev / total
    if hasattr(mca, "singular_values_"):
        sv = np.asarray(mca.singular_values_, dtype=float).ravel()
        ev = sv ** 2
        total = ev.sum() or 1.0
        return ev / total
    raise AttributeError("No fue posible obtener la inercia explicada de 'prince.MCA'.")

def fit_mca_full(cat_df: pd.DataFrame, threshold: float = 0.80, n_components: int = 50,
                 max_modalities_per_col: int = 50, rare_min_count: int = 30):
    """Ajusta MCA (sin muestreo) con reducción de cardinalidad; devuelve dims hasta el umbral."""
    if cat_df.shape[1] == 0:
        return pd.DataFrame(index=cat_df.index), None, np.array([])

    if not PRINCE_OK:
        raise ImportError(
            "No se pudo importar 'prince' (requerido para MCA). "
            "Instala con: pip install prince\nDetalle: " + PRINCE_ERR
        )

    cat = reduce_categorical_cardinality(cat_df, max_modalities_per_col=max_modalities_per_col,
                                         rare_min_count=rare_min_count)

    mca = MCA(n_components=n_components)
    mca = mca.fit(cat)

    inertia = safe_explained_inertia(mca)
    cum_inertia = np.cumsum(inertia)
    ndims = int(np.searchsorted(cum_inertia, threshold) + 1)
    ndims = min(ndims, len(inertia))

    coords = mca.transform(cat)
    ndims = min(ndims, coords.shape[1])
    dims = coords.iloc[:, :ndims].copy()
    dims.columns = [f"Dim{i+1}" for i in range(ndims)]
    return dims, mca, cum_inertia

# ================= Interfaz =================

st.set_page_config(page_title="PCA + MCA | Diabetes", layout="wide")

st.title("PCA + MCA del dataset de Diabetes en EE. UU.")
st.markdown(
    " **Objetivo:** aplicar **PCA** a variables **numéricas** y **MCA** a **categóricas**, "
    "retener componentes/dimensiones hasta alcanzar **≥ 80%** de varianza/inercia acumulada, "
    "y **crear un nuevo dataset** con las **PCs** y las **Dimensiones** concatenadas."
)

with st.expander("🧩 Requisitos e instrucciones rápidas", expanded=True):
    st.markdown("- Dependencias: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `prince`")
    st.code("pip install -r requirements.txt", language="bash")
    st.markdown("- Ejecuta la app con:")
    st.code("streamlit run app.py", language="bash")

# === Rutas base ===
BASE_DIR = Path(__file__).resolve().parent

# === Carga opcional del mapping desde sidebar o archivo local ===
ids_map = None
st.sidebar.subheader("Opcional: mapping de diagnósticos (IDS_mapping.csv)")
map_file = st.sidebar.file_uploader("Sube IDS_mapping.csv", type=["csv"])

if map_file is not None:
    try:
        ids_map = pd.read_csv(map_file)
        st.sidebar.success("Mapping cargado desde la subida.")
    except Exception as e:
        st.sidebar.warning(f"No se pudo leer el mapping subido: {e}")
else:
    local_map = BASE_DIR / "IDS_mapping.csv"
    if local_map.exists():
        try:
            ids_map = pd.read_csv(local_map)
            st.sidebar.success(f"Mapping cargado desde archivo local: {local_map.name}")
        except Exception as e:
            st.sidebar.warning(f"No se pudo leer el mapping local: {e}")
    else:
        st.sidebar.info("Sin mapping: la app continuará sin agrupar diagnósticos.")

# === Cargar dataset principal ===
DATA_PATH = BASE_DIR / "diabetic_data.csv"
if not DATA_PATH.exists():
    st.error("No se encontró `diabetic_data.csv` en el mismo directorio que `app.py`.")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.exception(e)
    st.stop()

st.success(f"Dataset cargado: {DATA_PATH.name} — Forma: {df.shape[0]} filas × {df.shape[1]} columnas")
st.dataframe(df.head(10), use_container_width=True)

# Paso 1: Limpieza IDs + mapping (si existe)
st.header("Paso 1: Limpieza de posibles columnas identificadoras y mapping opcional")
try:
    df_clean, dropped = drop_identifier_like_columns(df)
    st.write("Columnas eliminadas:", dropped if dropped else "Ninguna")
    st.code("df_clean, dropped = drop_identifier_like_columns(df)", language="python")

    df_clean = apply_diag_mapping(df_clean, ids_map)
    st.dataframe(df_clean.head(5), use_container_width=True)
except Exception as e:
    st.exception(e)
    st.stop()

# Paso 2: Separación
st.header("Paso 2: Separación de variables numéricas y categóricas")
try:
    num_df, cat_df = split_numeric_categorical(df_clean)
    st.write(f"Numéricas: {num_df.shape[1]} columnas | Categóricas: {cat_df.shape[1]} columnas")
    st.code("num_df, cat_df = split_numeric_categorical(df_clean)", language="python")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Vista numéricas")
        st.dataframe(num_df.head(5), use_container_width=True)
    with c2:
        st.subheader("Vista categóricas")
        st.dataframe(cat_df.head(5), use_container_width=True)
except Exception as e:
    st.exception(e)
    st.stop()

# Paso 3: PCA
st.header("Paso 3: PCA sobre variables numéricas (umbral 80%)")
try:
    pcs, pca_model, cum_var = fit_pca(num_df, threshold=0.80)
    if pca_model is None or pcs.shape[1] == 0:
        st.warning("No se generaron PCs (¿no hay columnas numéricas?).")
    else:
        st.write(f"Componentes retenidas: **{pcs.shape[1]}**")
        st.code("pcs, pca_model, cum_var = fit_pca(num_df, threshold=0.80)", language="python")
        st.line_chart(pd.DataFrame(cum_var, columns=["Varianza acumulada"]))
        st.dataframe(pcs.head(10), use_container_width=True)
except Exception as e:
    st.exception(e)
    st.stop()

# Paso 4: MCA
st.header("Paso 4: MCA sobre variables categóricas (umbral 80%)")
try:
    if not PRINCE_OK:
        raise ImportError("`prince` no está instalado correctamente. Ejecuta: pip install -r requirements.txt")

    with st.spinner("Calculando MCA (reduciendo categorías para acelerar)..."):
        dims, mca_model, cum_inertia = fit_mca_full(
            cat_df,
            threshold=0.80,
            n_components=50,            # suficientes dims para llegar al 80%
            max_modalities_per_col=50,  # recorta cardinalidad por columna
            rare_min_count=30
        )

    if mca_model is None or dims.shape[1] == 0:
        st.warning("No se generaron Dimensiones (¿no hay columnas categóricas?).")
    else:
        st.write(f"Dimensiones retenidas: **{dims.shape[1]}**")
        st.code(
            "dims, mca_model, cum_inertia = fit_mca_full(cat_df, threshold=0.80, n_components=50, max_modalities_per_col=50, rare_min_count=30)",
            language="python"
        )
        st.line_chart(pd.DataFrame(cum_inertia, columns=["Inercia acumulada"]))
        st.dataframe(dims.head(10), use_container_width=True)

except Exception as e:
    st.exception(e)
    # Si falla MCA, seguimos con PCs solamente
    dims = pd.DataFrame(index=df.index)

# Paso 5: Concatenación
st.header("Paso 5: Nuevo dataset con PCs + Dimensiones concatenadas")
try:
    out_df = pd.concat([pcs, dims], axis=1)
    if out_df.shape[1] == 0:
        st.error("No hay columnas transformadas para mostrar.")
    else:
        st.success(f"Dataset final: {out_df.shape[0]} filas × {out_df.shape[1]} columnas")
        st.dataframe(out_df.head(50), use_container_width=True)

        buf = io.BytesIO()
        out_df.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button(
            label="⬇️ Descargar dataset concatenado (CSV)",
            data=buf,
            file_name="dataset_pca_mca.csv",
            mime="text/csv",
        )
except Exception as e:
    st.exception(e)

st.markdown("---")
st.caption("Umbral fijo al 80%. La reducción de cardinalidad acelera el MCA sin perder interpretabilidad.")

