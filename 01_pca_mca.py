# 01_pca_mca.py
# -----------------------------------------------
# Tarea 1: PCA (numéricas) + MCA (categóricas)
# Umbral por defecto: 0.80
# Opcional: mapping de diagnósticos y reducción de cardinalidad
# -----------------------------------------------

import argparse
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from prince import MCA
except Exception as e:
    raise ImportError("Falta 'prince' (MCA). Instala: pip install prince") from e


# -------- Utils en línea con app.py --------
def drop_identifier_like_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    ids_reales = {"encounter_id", "patient_nbr"}
    dropped = [c for c in df.columns if c in ids_reales]
    return df.drop(columns=dropped, errors="ignore"), dropped

def apply_diag_mapping(df_in: pd.DataFrame, ids_map: pd.DataFrame | None) -> pd.DataFrame:
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

def split_numeric_categorical(df: pd.DataFrame):
    num_df = df.select_dtypes(include=["number"]).copy()
    cat_df = df.select_dtypes(exclude=["number"]).copy()
    return num_df, cat_df

def fit_pca(num_df: pd.DataFrame, threshold: float = 0.80):
    if num_df.shape[1] == 0:
        return pd.DataFrame(index=num_df.index), None, np.array([])
    X = num_df.fillna(num_df.mean(numeric_only=True))
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(cum, threshold) + 1)
    pcs = pd.DataFrame(pca.transform(Xs)[:, :n], index=num_df.index,
                       columns=[f"PC{i+1}" for i in range(n)])
    return pcs, cum

def reduce_categorical_cardinality(cat: pd.DataFrame, max_modalities_per_col=50, rare_min_count=30):
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

def safe_explained_inertia(mca) -> np.ndarray:
    if hasattr(mca, "explained_inertia_"):
        return np.asarray(mca.explained_inertia_, dtype=float)
    if hasattr(mca, "eigenvalues_"):
        ev = np.asarray(mca.eigenvalues_, dtype=float).ravel()
        tot = ev.sum() or 1.0
        return ev / tot
    if hasattr(mca, "singular_values_"):
        sv = np.asarray(mca.singular_values_, dtype=float).ravel()
        ev = sv ** 2
        tot = ev.sum() or 1.0
        return ev / tot
    raise AttributeError("No se pudo obtener inercia explicada de prince.MCA.")

def fit_mca(cat_df: pd.DataFrame, threshold: float = 0.80,
            n_components: int = 50,
            reduce_cardinality: bool = True,
            max_modalities_per_col: int = 50,
            rare_min_count: int = 30):
    if cat_df.shape[1] == 0:
        return pd.DataFrame(index=cat_df.index), np.array([])
    cat = cat_df.fillna("missing").astype(str)
    if reduce_cardinality:
        cat = reduce_categorical_cardinality(cat, max_modalities_per_col, rare_min_count)
    mca = MCA(n_components=n_components).fit(cat)
    inertia = safe_explained_inertia(mca)
    cum = np.cumsum(inertia)
    n = int(np.searchsorted(cum, threshold) + 1)
    n = min(n, len(inertia))
    coords = mca.transform(cat)
    n = min(n, coords.shape[1])
    dims = pd.DataFrame(coords.iloc[:, :n].to_numpy(), index=cat.index,
                        columns=[f"Dim{i+1}" for i in range(n)])
    return dims, cum


def main(args):
    df = pd.read_csv(args.input)
    # IDs fuera
    df, dropped = drop_identifier_like_columns(df)
    # Mapping opcional
    ids_map = pd.read_csv(args.mapping) if args.mapping and os.path.exists(args.mapping) else None
    df = apply_diag_mapping(df, ids_map)

    num_df, cat_df = split_numeric_categorical(df)
    pcs, cum_pca = fit_pca(num_df, threshold=args.threshold)
    dims, cum_mca = fit_mca(
        cat_df,
        threshold=args.threshold,
        n_components=args.n_components,
        reduce_cardinality=args.reduce_cardinality,
        max_modalities_per_col=args.max_modalities_per_col,
        rare_min_count=args.rare_min_count
    )
    out = pd.concat([pcs, dims], axis=1)
    out.to_csv(args.output, index=False)

    print(f"IDs eliminados: {dropped}")
    print(f"Umbral: {args.threshold:.0%}")
    print(f"PCA -> componentes: {pcs.shape[1]}")
    print(f"MCA -> dimensiones: {dims.shape[1]}")
    print(f"Salida: {args.output}  |  Forma: {out.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="diabetic_data.csv")
    ap.add_argument("--output", default="dataset_pca_mca.csv")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--mapping", type=str, default=None, help="Ruta a IDS_mapping.csv (opcional)")
    ap.add_argument("--reduce-cardinality", action="store_true", help="Agrupar raras en 'OTHER' (recomendado)")
    ap.add_argument("--max-modalities-per-col", type=int, default=50)
    ap.add_argument("--rare-min-count", type=int, default=30)
    ap.add_argument("--n-components", type=int, default=50)
    args = ap.parse_args()
    main(args)

