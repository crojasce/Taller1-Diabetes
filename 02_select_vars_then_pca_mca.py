# 02_select_vars_then_pca_mca.py
# ------------------------------------------------
# Tarea 2: selección de variables y luego PCA/MCA
# Umbral por defecto: 0.80; target por defecto: 'readmitted'
# ------------------------------------------------

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2

try:
    import prince
except Exception as e:
    raise ImportError(
        "Falta la librería 'prince' (para MCA). Instala: pip install prince"
    ) from e


def select_numeric(df, y, threshold=0.80):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        return [], np.array([])
    X = df[num_cols].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    sel = SelectKBest(score_func=f_classif, k="all").fit(Xs, y)
    scores = sel.scores_
    order = np.argsort(scores)[::-1]
    cum = np.cumsum(scores[order]) / (scores.sum() or 1.0)
    k = int(np.argmax(cum >= threshold) + 1)
    chosen = [num_cols[i] for i in order[:k]]
    return chosen, cum


def select_categorical(df, y, target_col, threshold=0.80):
    cat_cols_all = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols_all if c != target_col]
    if len(cat_cols) == 0:
        return [], np.array([])
    # Codificar cada columna categórica por separado
    df_enc = df[cat_cols].fillna("NA").astype(str).apply(lambda s: LabelEncoder().fit_transform(s))
    sel = SelectKBest(score_func=chi2, k="all").fit(df_enc, y)
    scores = sel.scores_
    order = np.argsort(scores)[::-1]
    cum = np.cumsum(scores[order]) / (scores.sum() or 1.0)
    k = int(np.argmax(cum >= threshold) + 1)
    chosen = [cat_cols[i] for i in order[:k]]
    return chosen, cum


def pca_on_selected(df, cols, threshold=0.80):
    if len(cols) == 0:
        return pd.DataFrame(index=df.index), np.array([])
    X = df[cols].fillna(0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA()
    Xp = pca.fit_transform(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.argmax(cum >= threshold) + 1)
    pcs = pd.DataFrame(Xp[:, :n], index=df.index, columns=[f"PC{i+1}" for i in range(n)])
    return pcs, cum


def mca_on_selected(df, cols, threshold=0.80, random_state=42):
    if len(cols) == 0:
        return pd.DataFrame(index=df.index), np.array([])
    if len(cols) == 1:
        # Con una sola variable categórica no tiene sentido MCA -> la devolvemos tal cual
        return df[cols].reset_index(drop=True).to_frame() if isinstance(df[cols], pd.Series) else df[cols], np.array([])
    Xc = df[cols].fillna("NA").astype(str)
    mca = prince.MCA(n_components=len(cols), random_state=random_state).fit(Xc)
    coords = mca.transform(Xc)
    eigvals = np.asarray(mca.eigenvalues_, dtype=float).ravel()
    ratios = eigvals / eigvals.sum() if eigvals.sum() else np.zeros_like(eigvals)
    cum = np.cumsum(ratios)
    n = int(np.argmax(cum >= threshold) + 1)
    dims = pd.DataFrame(np.array(coords)[:, :n], index=df.index,
                        columns=[f"Dim{i+1}" for i in range(n)])
    return dims, cum


def main(input_csv: str, output_csv: str, target_col: str, threshold: float):
    df = pd.read_csv(input_csv)

    # Variable objetivo (como en tu Colab)
    y = df[target_col]
    if y.dtype == "O":
        y = LabelEncoder().fit_transform(y.astype(str))

    # Selección de variables
    selected_num, cum_num = select_numeric(df, y, threshold=threshold)
    selected_cat, cum_cat = select_categorical(df, y, target_col, threshold=threshold)

    # PCA / MCA según seleccionadas
    pcs, cum_pca = pca_on_selected(df, selected_num, threshold=threshold)
    dims, cum_mca = mca_on_selected(df, selected_cat, threshold=threshold)

    # Concatenar y guardar
    out = pd.concat([pcs, dims], axis=1)
    out.to_csv(output_csv, index=False)

    print(f"Umbral: {threshold:.0%}")
    print(f"Numéricas seleccionadas: {len(selected_num)} -> PCs retenidas: {pcs.shape[1]}")
    print(f"Categóricas seleccionadas: {len(selected_cat)} -> Dims retenidas: {dims.shape[1]}")
    print(f"Dataset final guardado en: {output_csv}")
    print(f"Forma final: {out.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="diabetic_data.csv")
    ap.add_argument("--output", default="dataset_pca_mca_sel.csv")
    ap.add_argument("--target", default="readmitted")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()
    main(args.input, args.output, args.target, args.threshold)
