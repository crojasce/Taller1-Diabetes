# 02_select_vars_then_pca_mca.py
# ------------------------------------------------
# Tarea 2: selección de variables y luego PCA/MCA
# Umbral por defecto: 0.80; target por defecto: 'readmitted'
# Opcional: mapping y reducción de cardinalidad en MCA
# ------------------------------------------------

import argparse
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2

try:
    from prince import MCA
except Exception as e:
    raise ImportError("Falta 'prince' (MCA). Instala: pip install prince") from e


# -------- Utils compartidas con app.py --------
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

def fit_pca_on_cols(df: pd.DataFrame, cols: List[str], threshold: float = 0.80):
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

def fit_mca_on_cols(df: pd.DataFrame, cols: List[str], threshold: float = 0.80,
                    n_components: int = 50, reduce_card: bool = True,
                    max_modalities_per_col: int = 50, rare_min_count: int = 30):
    if len(cols) < 2:
        # alineado con la app: sin al menos 2 categóricas no se aplica MCA
        return pd.DataFrame(index=df.index), np.array([])
    cat = df[cols].copy()
    if reduce_card:
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


# -------- Selección (Tarea 2) --------
def select_numeric_by_f_classif(df: pd.DataFrame, y, threshold: float = 0.80):
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
    cat_all = df.select_dtypes(exclude=["number"]).columns.tolist()
    cat_cols = [c for c in cat_all if c != target_col]
    if len(cat_cols) == 0:
        return [], np.array([])
    enc = df[cat_cols].fillna("NA").astype(str).apply(lambda s: LabelEncoder().fit_transform(s))
    sel = SelectKBest(score_func=chi2, k="all").fit(enc, y)
    scores = sel.scores_
    order = np.argsort(scores)[::-1]
    total = scores.sum() or 1.0
    cum = np.cumsum(scores[order]) / total
    k = int(np.searchsorted(cum, threshold) + 1)
    chosen = [cat_cols[i] for i in order[:k]]
    return chosen, cum


def main(args):
    df = pd.read_csv(args.input)
    # IDs fuera + mapping opcional
    df, dropped = drop_identifier_like_columns(df)
    ids_map = pd.read_csv(args.mapping) if args.mapping and os.path.exists(args.mapping) else None
    df = apply_diag_mapping(df, ids_map)

    # target
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' no está en el dataset.")
    y = df[args.target]
    if y.dtype == "O":
        y = LabelEncoder().fit_transform(y.astype(str))

    # selección
    sel_num, cum_num = select_numeric_by_f_classif(df, y, threshold=args.threshold)
    sel_cat, cum_cat = select_categorical_by_chi2(df, y, args.target, threshold=args.threshold)

    # PCA/MCA sobre seleccionadas
    pcs, cum_pca = fit_pca_on_cols(df, sel_num, threshold=args.threshold)
    dims, cum_mca = fit_mca_on_cols(
        df, sel_cat, threshold=args.threshold,
        n_components=args.n_components,
        reduce_card=args.reduce_cardinality,
        max_modalities_per_col=args.max_modalities_per_col,
        rare_min_count=args.rare_min_count
    )

    out = pd.concat([pcs, dims], axis=1)
    out.to_csv(args.output, index=False)

    print(f"IDs eliminados: {dropped}")
    print(f"Umbral: {args.threshold:.0%}")
    print(f"Num seleccionadas: {len(sel_num)} -> PCs: {pcs.shape[1]}")
    print(f"Cat seleccionadas: {len(sel_cat)} -> Dims: {dims.shape[1]}")
    print(f"Salida: {args.output}  |  Forma: {out.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="diabetic_data.csv")
    ap.add_argument("--output", default="dataset_pca_mca_sel.csv")
    ap.add_argument("--target", default="readmitted")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--mapping", type=str, default=None, help="Ruta a IDS_mapping.csv (opcional)")
    ap.add_argument("--reduce-cardinality", action="store_true", help="Agrupar raras en 'OTHER' (recomendado)")
    ap.add_argument("--max-modalities-per-col", type=int, default=50)
    ap.add_argument("--rare-min-count", type=int, default=30)
    ap.add_argument("--n-components", type=int, default=50)
    args = ap.parse_args()
    main(args)
