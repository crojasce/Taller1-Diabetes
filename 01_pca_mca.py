# 01_pca_mca.py
# -----------------------------------------------
# Tarea 1: PCA (numéricas) + MCA (categóricas)
# Umbral por defecto: 0.80
# -----------------------------------------------

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 'prince' para MCA
try:
    import prince
except Exception as e:
    raise ImportError(
        "Falta la librería 'prince' (para MCA). Instala: pip install prince"
    ) from e


def pca_select(X_num: pd.DataFrame, threshold: float = 0.80):
    """Estandariza, aplica PCA y retorna PCs (≥ umbral de varianza)."""
    if X_num.shape[1] == 0:
        return pd.DataFrame(index=X_num.index), np.array([])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_num.fillna(0))
    pca = PCA()
    Xp = pca.fit_transform(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.argmax(cum >= threshold) + 1)
    pcs = pd.DataFrame(Xp[:, :n], index=X_num.index,
                       columns=[f"PC{i+1}" for i in range(n)])
    return pcs, cum


def mca_select(X_cat: pd.DataFrame, threshold: float = 0.80, random_state: int = 42):
    """Convierte a string, aplica MCA y retorna dims (≥ umbral de inercia)."""
    if X_cat.shape[1] == 0:
        return pd.DataFrame(index=X_cat.index), np.array([])
    Xc = X_cat.fillna("NA").astype(str)

    mca = prince.MCA(n_components=Xc.shape[1], random_state=random_state)
    mca = mca.fit(Xc)
    coords = mca.transform(Xc)

    # Inercia explicada (como en tu Colab: eigenvalues -> proporciones)
    eigvals = np.asarray(mca.eigenvalues_, dtype=float).ravel()
    ratios = eigvals / eigvals.sum() if eigvals.sum() else np.zeros_like(eigvals)
    cum = np.cumsum(ratios)
    n = int(np.argmax(cum >= threshold) + 1)

    dims = pd.DataFrame(np.array(coords)[:, :n], index=Xc.index,
                        columns=[f"Dim{i+1}" for i in range(n)])
    return dims, cum


def main(input_csv: str, output_csv: str, threshold: float):
    df = pd.read_csv(input_csv)

    # Separa numéricas/categóricas (como en el notebook)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    X_num = df[num_cols]
    X_cat = df[cat_cols]

    pcs, cum_pca = pca_select(X_num, threshold=threshold)
    dims, cum_mca = mca_select(X_cat, threshold=threshold)

    out = pd.concat([pcs, dims], axis=1)
    out.to_csv(output_csv, index=False)

    print(f"Umbral: {threshold:.0%}")
    print(f"PCA -> componentes retenidas: {pcs.shape[1]}")
    print(f"MCA -> dimensiones retenidas: {dims.shape[1]}")
    print(f"Dataset final guardado en: {output_csv}")
    print(f"Forma final: {out.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="diabetic_data.csv")
    ap.add_argument("--output", default="dataset_pca_mca.csv")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()
    main(args.input, args.output, args.threshold)
