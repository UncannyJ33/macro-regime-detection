"""Fits unsupervised models to identify macro regimes.

Trains and evaluates clustering / sequence models (e.g., K-Means, GMM,
Hidden Markov Model) on PCA-reduced features and assigns a regime label
to each time period.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config


def fit_pca(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """Standardize features and reduce dimensionality via PCA.

    Applies StandardScaler before PCA because rate-of-change features are not
    on the same scale as rolling z-scores, and PCA is sensitive to feature
    variance. The scaler is returned so the same transform can be applied to
    out-of-sample data later.

    Args:
        features: Engineered feature DataFrame from feature_engineering.py,
            with a DatetimeIndex and no NaN values.

    Returns:
        A tuple of:
        - pc_df: DataFrame of shape (n_obs, N_PCA_COMPONENTS) with columns
          named "PC1", "PC2", ..., sharing the same DatetimeIndex as features.
        - pca: Fitted PCA object (sklearn.decomposition.PCA).
        - scaler: Fitted StandardScaler object used to standardize inputs.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=config.N_PCA_COMPONENTS, random_state=42)
    components = pca.fit_transform(scaled)

    pc_columns = [f"PC{i + 1}" for i in range(config.N_PCA_COMPONENTS)]
    pc_df = pd.DataFrame(components, index=features.index, columns=pc_columns)

    # Report variance explained — useful for judging whether N_PCA_COMPONENTS
    # is capturing enough signal before clustering.
    cumulative = 0.0
    print(f"PCA variance explained ({config.N_PCA_COMPONENTS} components):")
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        print(f"  PC{i + 1}: {var:.1%}  (cumulative: {cumulative:.1%})")

    return pc_df, pca, scaler


def get_pca_loadings(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """Build a readable table of how each original feature loads onto each PC.

    A loading is the correlation between an original feature and a principal
    component. Large absolute values indicate that the feature strongly drives
    that component, which is the primary tool for interpreting what each regime
    dimension represents economically.

    Args:
        pca: A fitted PCA object (output of fit_pca).
        feature_names: Ordered list of original feature column names, matching
            the columns passed to fit_pca.

    Returns:
        DataFrame of shape (n_features, N_PCA_COMPONENTS) with feature names
        as the index and "PC1", "PC2", ... as columns. Values are loadings
        (eigenvector components), ranging from -1 to 1.
    """
    pc_columns = [f"PC{i + 1}" for i in range(pca.n_components_)]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=pc_columns,
    )
    return loadings
