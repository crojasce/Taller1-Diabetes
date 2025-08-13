# app.py
# ------------------------------------------------------------
# Taller PCA + MCA (Tarea 1 y Tarea 2)
# - Tarea 1: PCA (numéricas) + MCA (categóricas), umbral 80%
# - Tarea 2: Selección de variables (f_classif / chi2) -> PCA/MCA, umbral 80%
# - Mapping de diagnósticos OPCIONAL (archivo local IDS_mapping.csv)
# - Descarga del dataset final (PCs + Dimensiones)
# ------------------------------------------------------------

import io
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# Intento seguro de importar prince.MCA
try:
    from prince import MCA
    PRINCE_OK = True
    PRINCE_ERR = ""
except Exception as e:
    PRINCE_OK = False
    PRINCE_ERR = str(e)

# ================= Utilidades comunes =================

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
        return df_in
    mapping = ids_map.set_index(col_code)[col_group].to_dict()
    df = df_in.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(df[col].astype(str))
    return df

def split_numeric_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    num_df = df.select_dtypes(include=["number"]).copy()
    cat_df = df.select_dtypes(exclude=["number"]).copy()
    return num_df, cat_df

# ================= PCA (común) =================

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

# ================= MCA (común) =================

def reduce_categorical_cardinality(cat: pd.DataFrame, max_modalities_per_col=50, rare_min_count=30):
    """
    Reducir cardinalidad para acelerar/estabilizar el MCA en datasets grandes:
    - Convierte a str y rellena NaN -> 'missing'
    - Si una columna tiene demasiadas categorías, guarda top-(max-1) y el resto -> 'OTHER'
    - Además, toda categoría con conteo < rare_min_count -> 'OTHER'
    """
    cat = cat.fillna("missing").astype(str).copy()
    for c in cat.columns:
        vc = cat[c].value_counts(dropna=False)
        if len(vc) > max_modalities_per_col:
            keep = set(vc.index[:max_modalities_per_col - 1])
            cat[c] = np.where(cat[c].isin(keep), cat[c], "OTHER")
            vc = cat[c].value_counts(dropna=False)
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

def fit_mca(cat_df: pd.DataFrame, threshold: float = 0.80,
            n_components: int = 50,
            reduce_cardinality: bool = True,
            max_modalities_per_col: int = 50,
            rare_min_count: int = 30):
    """Ajusta MCA con opción de reducir cardinalidad; devuelve dims hasta el umbral."""
    if cat_df.shape[1] == 0:
        return pd.DataFrame(index=cat_df.index), None, np.array([])
    if not PRINCE_OK:
        raise ImportError("No se pudo importar 'prince'. Instala: pip install prince\nDetalle: " + PRINCE_ERR)
    cat = cat_df.fillna("missing").astype(str).copy()
    if reduce_cardinality:
        cat = reduce_categorical_cardinality(cat, max_modalities_per_col, rare_min_count)
    mca = MCA(n_components=n_components).fit(cat)
    inertia = safe_explained_inertia(mca)
    cum_inertia = np.cumsum(inertia)
    ndims = int(np.searchsorted(cum_inertia, threshold) + 1)
    ndims = min(ndims, len(inertia))
    coords = mca.transform(cat)
    ndims = min(ndims, coords.shape[1])
    dims = coords.iloc[:, :ndims].copy()
    dims.columns = [f"Dim{i+1}" for i in range(ndims)]
    return dims, mca, cum_inertia

# ================= Selección (Tarea 2) =================

def select_numeric_by_f_classif(df: pd.DataFrame, y, threshold: float = 0.80):
    """Selecciona variables numéricas por importancia acumulada (ANOVA F / SelectKBest)."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) == 0:
        return [], np.array([])
    X = df[num_cols].fillna(0.0)
    Xs = StandardScaler().fit_transform(X)
    sel = SelectKBest(score_func=f_classif, k="all").fit(Xs, y)
    scores = sel.scores_
    order = np.argsort(scores)[::-1]
    total = scores.sum() or 1.0
    cum = np.cumsum(scores[order]) / total
    k = int(np.searchsorted(cum, threshold) + 1)
    chosen = [num_cols[i] for i in order[:k]]
    return chosen, cum

def select_categorical_by_chi2(df: pd.DataFrame, y, target_col: str, threshold: float = 0.80):
    """Selecciona variables categóricas por importancia acumulada (Chi2 / SelectKBest)."""
    cat_cols_all = df.select_dtypes(exclude=["number"]).columns.tolist()
    cat_cols = [c for c in cat_cols_all if c != target_col]
    if len(cat_cols) == 0:
        return [], np.array([])
    # Codificación label por columna
    df_enc = df[cat_cols].fillna("NA").astype(str).apply(lambda s: LabelEncoder().fit_transform(s))
    sel = SelectKBest(score_func=chi2, k="all").fit(df_enc, y)
    scores = sel.scores_
    order = np.argsort(scores)[::-1]
    total = scores.sum() or 1.0
    cum = np.cumsum(scores[order]) / total
    k = int(np.searchsorted(cum, threshold) + 1)
    chosen = [cat_cols[i] for i in order[:k]]
    return chosen, cum

def pca_on_selected(df: pd.DataFrame, cols: List[str], threshold: float = 0.80):
    """PCA sobre columnas numéricas seleccionadas."""
    if len(cols) == 0:
        return pd.DataFrame(index=df.index), np.array([])
    X = df[cols].fillna(0.0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cum, threshold) + 1)
    pcs = pd.DataFrame(pca.transform(Xs)[:, :n], index=df.index,
                       columns=[f"PC{i+1}" for i in range(n)])
    return pcs, cum

def mca_on_selected(df: pd.DataFrame, cols: List[str], threshold: float = 0.80,
                    n_components: int = 50,
                    reduce_cardinality: bool = True):
    """MCA sobre columnas categóricas seleccionadas (si ≥2); si no, devuelve vacío."""
    if len(cols) < 2:
        return pd.DataFrame(index=df.index), np.array([])
    cat = df[cols].copy()
    dims, mca_model, cum_inertia = fit_mca(
        cat_df=cat,
        threshold=threshold,
        n_components=n_components,
        reduce_cardinality=reduce_cardinality
    )
    return dims, cum_inertia

# ================= Interfaz =================

st.set_page_config(page_title="Taller PCA + MCA | Tarea 1 y 2", layout="wide")

st.title("Taller PCA + MCA — **Tarea 1** y **Tarea 2**")
st.markdown(
    "- **Tarea 1:** PCA (numéricas) + MCA (categóricas) y concatenar (umbral ≥ 80%).\n"
    "- **Tarea 2:** Selección de variables (ANOVA F / χ²) → PCA/MCA y concatenar (umbral ≥ 80%)."
)

with st.sidebar:
    st.header("⚙️ Configuración")
    mode = st.radio("Modo del taller", ["Tarea 1 (PCA+MCA)", "Tarea 2 (Selección → PCA/MCA)"], index=0)
    threshold = st.slider("Umbral acumulado", min_value=0.50, max_value=0.95, value=0.80, step=0.01)
    reduce_flag = st.checkbox("Reducir cardinalidad en MCA", value=True)
    st.caption("Reducir cardinalidad (raras → 'OTHER') acelera MCA en datasets grandes.")

# === Rutas base ===
BASE_DIR = Path(__file__).resolve().parent

# === Cargar mapping local si existe (opcional) ===
ids_map = None
local_map = BASE_DIR / "IDS_mapping.csv"
if local_map.exists():
    try:
        ids_map = pd.read_csv(local_map)
    except Exception:
        ids_map = None  # si no se puede leer, se omite sin romper

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

# ===== Flujo común inicial =====
st.header("Paso 1: Limpieza de identificadores y (opcional) mapping de diagnósticos")
df_clean, dropped = drop_identifier_like_columns(df)
st.write("Columnas eliminadas:", dropped if dropped else "Ninguna")
df_clean = apply_diag_mapping(df_clean, ids_map)
st.dataframe(df_clean.head(5), use_container_width=True)

st.header("Paso 2: Separación de variables numéricas y categóricas")
num_df, cat_df = split_numeric_categorical(df_clean)
st.write(f"Numéricas: {num_df.shape[1]} | Categóricas: {cat_df.shape[1]}")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Vista numéricas")
    st.dataframe(num_df.head(5), use_container_width=True)
with c2:
    st.subheader("Vista categóricas")
    st.dataframe(cat_df.head(5), use_container_width=True)

# ===== Modo Tarea 1 =====
if mode.startswith("Tarea 1"):
    st.header("**Tarea 1** — Paso 3: PCA sobre numéricas")
    pcs, pca_model, cum_var = fit_pca(num_df, threshold=threshold)
    if pcs.shape[1] == 0:
        st.warning("No se generaron PCs.")
    else:
        st.write(f"Componentes retenidas: **{pcs.shape[1]}**")
        st.line_chart(pd.DataFrame(cum_var, columns=["Varianza acumulada"]))
        st.dataframe(pcs.head(10), use_container_width=True)

    st.header("**Tarea 1** — Paso 4: MCA sobre categóricas")
    try:
        dims, mca_model, cum_inertia = fit_mca(
            cat_df=cat_df,
            threshold=threshold,
            n_components=50,
            reduce_cardinality=reduce_flag
        )
        if dims.shape[1] == 0:
            st.warning("No se generaron Dimensiones.")
        else:
            st.write(f"Dimensiones retenidas: **{dims.shape[1]}**")
            st.line_chart(pd.DataFrame(cum_inertia, columns=["Inercia acumulada"]))
            st.dataframe(dims.head(10), use_container_width=True)
    except Exception as e:
        st.exception(e)
        dims = pd.DataFrame(index=df.index)

    st.header("**Tarea 1** — Paso 5: Dataset final (PCs + Dimensiones)")
    out_df = pd.concat([pcs, dims], axis=1)
    st.success(f"Dataset final: {out_df.shape[0]} filas × {out_df.shape[1]} columnas")
    st.dataframe(out_df.head(50), use_container_width=True)
    buf = io.BytesIO()
    out_df.to_csv(buf, index=False); buf.seek(0)
    st.download_button("⬇️ Descargar dataset (CSV)", buf, "dataset_pca_mca.csv", "text/csv")

# ===== Modo Tarea 2 =====
else:
    st.header("**Tarea 2** — Paso 3: Selección de variables")
    # Target por defecto típico del dataset
    default_target = "readmitted" if "readmitted" in df_clean.columns else df_clean.columns[-1]
    target_col = st.selectbox("Variable objetivo (para f_classif / chi2)", options=list(df_clean.columns),
                              index=list(df_clean.columns).index(default_target))
    y = df_clean[target_col]
    if y.dtype == "O":
        y = LabelEncoder().fit_transform(y.astype(str))

    selected_num, cum_num = select_numeric_by_f_classif(df_clean, y, threshold=threshold)
    st.write(f"Numéricas seleccionadas ({len(selected_num)}):", selected_num[:20], "..." if len(selected_num) > 20 else "")
    if len(cum_num) > 0:
        st.line_chart(pd.DataFrame(cum_num, columns=["% relevancia acumulada (num)"]))

    selected_cat, cum_cat = select_categorical_by_chi2(df_clean, y, target_col, threshold=threshold)
    st.write(f"Categóricas seleccionadas ({len(selected_cat)}):", selected_cat[:20], "..." if len(selected_cat) > 20 else "")
    if len(cum_cat) > 0:
        st.line_chart(pd.DataFrame(cum_cat, columns=["% relevancia acumulada (cat)"]))

    st.header("**Tarea 2** — Paso 4: PCA/MCA sobre variables seleccionadas")
    pcs, cum_pca = pca_on_selected(df_clean, selected_num, threshold=threshold)
    if pcs.shape[1] == 0:
        st.warning("No se generaron PCs (selección numérica vacía).")
    else:
        st.write(f"PCs retenidas: **{pcs.shape[1]}**")
        st.line_chart(pd.DataFrame(cum_pca, columns=["Varianza acumulada (PCA)"]))
        st.dataframe(pcs.head(10), use_container_width=True)

    dims, cum_mca = mca_on_selected(df_clean, selected_cat, threshold=threshold,
                                    n_components=50, reduce_cardinality=reduce_flag)
    if dims.shape[1] == 0 and len(selected_cat) < 2:
        st.info("⚠ Solo queda 1 (o 0) variable categórica seleccionada → MCA no aplicado (criterio del taller).")
    elif dims.shape[1] == 0:
        st.warning("No se generaron Dimensiones.")
    else:
        st.write(f"Dimensiones retenidas: **{dims.shape[1]}**")
        st.line_chart(pd.DataFrame(cum_mca, columns=["Inercia acumulada (MCA)"]))
        st.dataframe(dims.head(10), use_container_width=True)

    st.header("**Tarea 2** — Paso 5: Dataset final (PCs + Dimensiones)")
    out_df = pd.concat([pcs, dims], axis=1)
    st.success(f"Dataset final: {out_df.shape[0]} filas × {out_df.shape[1]} columnas")
    st.dataframe(out_df.head(50), use_container_width=True)
    buf = io.BytesIO()
    out_df.to_csv(buf, index=False); buf.seek(0)
    st.download_button("⬇️ Descargar dataset (CSV)", buf, "dataset_pca_mca_selected.csv", "text/csv")

st.markdown("---")
st.caption("Tarea 1 y Tarea 2 del taller con umbral por defecto del 80%.")


