"""Fits unsupervised models to identify macro regimes.

Trains and evaluates clustering / sequence models (e.g., K-Means, GMM,
Hidden Markov Model) on PCA-reduced features and assigns a regime label
to each time period.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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


def fit_kmeans(pc_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster PCA-reduced observations into macro regimes using K-Means.

    PCA components have unequal variances by construction (PC1 > PC2 > ... > PCn),
    so raw PC values would cause K-Means to weight high-variance components more
    heavily when computing distances. A second StandardScaler pass equalizes that
    before clustering. The scaler is applied internally and not returned — the
    returned DataFrame retains the original (unscaled) PC values for interpretability.

    Uses config.N_REGIMES clusters with a fixed random seed for reproducibility.
    The resulting integer labels (0 to N_REGIMES-1) have no inherent ordering —
    regime identity is determined by the cluster centroid, not the label value.

    Args:
        pc_df: PCA-transformed DataFrame from fit_pca(), with a DatetimeIndex
            and columns "PC1" through "PCn".

    Returns:
        Copy of pc_df with an additional "regime" column containing integer
        cluster labels in [0, config.N_REGIMES).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(pc_df)

    kmeans = KMeans(n_clusters=config.N_REGIMES, random_state=42, n_init=20)
    labels = kmeans.fit_predict(scaled)

    result = pc_df.copy()
    result["regime"] = labels

    counts = result["regime"].value_counts().sort_index()
    print(f"K-Means regimes (K={config.N_REGIMES}):")
    for regime, count in counts.items():
        pct = count / len(result) * 100
        print(f"  Regime {regime}: {count} months ({pct:.1f}%)")

    return result


def elbow_analysis(pc_df: pd.DataFrame) -> dict:
    """Compute inertia and silhouette score for K in [2, 8] to guide K selection.

    Inertia (within-cluster sum of squares) decreases monotonically with K —
    the "elbow" where the rate of decrease flattens is a reasonable K choice.
    Silhouette score measures how well-separated clusters are; higher is better,
    with a peak suggesting the natural number of clusters.

    Args:
        pc_df: PCA-transformed DataFrame from fit_pca(). Regime column must
            not be present — pass the raw pc_df, not the output of fit_kmeans.

    Returns:
        Dict with keys:
        - "k_values": list of int, K values tested (2 through 8).
        - "inertia": list of float, within-cluster sum of squares for each K.
        - "silhouette": list of float, mean silhouette score for each K.
    """
    # Scale PCs to equal variance before clustering, matching fit_kmeans behavior.
    scaled = StandardScaler().fit_transform(pc_df)

    k_values = list(range(2, 9))
    inertia_scores = []
    silhouette_scores = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(scaled)
        inertia_scores.append(km.inertia_)
        silhouette_scores.append(silhouette_score(scaled, labels))

    print(f"\n{'K':>4}  {'Inertia':>12}  {'Silhouette':>12}")
    print("-" * 32)
    for k, inertia, sil in zip(k_values, inertia_scores, silhouette_scores):
        marker = " <--" if k == config.N_REGIMES else ""
        print(f"{k:>4}  {inertia:>12.1f}  {sil:>12.4f}{marker}")

    return {
        "k_values": k_values,
        "inertia": inertia_scores,
        "silhouette": silhouette_scores,
    }


def fit_hmm(pc_df: pd.DataFrame) -> tuple[pd.DataFrame, GaussianHMM]:
    """Fit a Gaussian Hidden Markov Model to PCA-reduced features.

    HMM differs from K-Means by modeling regimes as a sequence of latent states
    with explicit transition probabilities — meaning the current regime depends on
    the previous one. This captures the persistence and switching dynamics of
    macro cycles that K-Means (which treats each observation independently) cannot.

    Uses multiple random initializations and keeps the fit with the highest
    log-likelihood to reduce sensitivity to initialization. The Viterbi algorithm
    then decodes the single most likely state sequence given the fitted model.

    PCA components are standardized before fitting because GaussianHMM estimates
    per-state covariance matrices; equal-scale inputs make initialization more
    stable and prevent one PC from dominating the likelihood.

    Args:
        pc_df: PCA-transformed DataFrame from fit_pca(), with a DatetimeIndex
            and columns "PC1" through "PCn". Must not contain a "regime" column.

    Returns:
        A tuple of:
        - result: Copy of pc_df with an "hmm_regime" column of integer state labels.
        - best_hmm: The fitted GaussianHMM object with the highest log-likelihood.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(pc_df)

    best_hmm = None
    best_score = -np.inf

    # Run multiple random initializations and keep the best by log-likelihood.
    for seed in range(10):
        hmm = GaussianHMM(
            n_components=config.N_REGIMES,
            covariance_type="full",
            n_iter=200,
            random_state=seed,
        )
        hmm.fit(scaled)
        score = hmm.score(scaled)
        if score > best_score:
            best_score = score
            best_hmm = hmm

    # Viterbi decoding: most likely state sequence given the observations.
    state_sequence = best_hmm.predict(scaled)

    # S2 collapsed into S1: S2 captured only a single observation (April 2020,
    # the COVID shock) and its transition matrix row shows P(S2→S1) = 1.0,
    # meaning the model itself treats it as an immediate transition to S1. A
    # one-month state is not a meaningful regime for allocation purposes, so we
    # remap it rather than expose a degenerate label downstream.
    state_sequence = np.where(state_sequence == 2, 1, state_sequence)

    result = pc_df.copy()
    result["hmm_regime"] = state_sequence

    # ── Transition matrix ────────────────────────────────────────────────────
    n = config.N_REGIMES
    header = "       " + "  ".join(f"→ S{j}" for j in range(n))
    print(f"\nHMM transition matrix (row = current state, col = next state):")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row in enumerate(best_hmm.transmat_):
        cells = "  ".join(f"{p:5.3f}" for p in row)
        print(f"  S{i} |  {cells}")

    # ── Stationary distribution ──────────────────────────────────────────────
    # Solve π = πP; normalize the left eigenvector of the transition matrix.
    eigenvalues, eigenvectors = np.linalg.eig(best_hmm.transmat_.T)
    stationary = eigenvectors[:, np.argmax(eigenvalues)].real
    stationary /= stationary.sum()

    print(f"\nStationary distribution (long-run % of time in each state):")
    for i, prob in enumerate(stationary):
        count = (state_sequence == i).sum()
        print(f"  S{i}: {prob:.1%}  (observed: {count} months, {count/len(state_sequence):.1%})")

    print(f"\nLog-likelihood (best init): {best_score:.1f}")

    return result, best_hmm


def label_regimes(df: pd.DataFrame, mapping: dict[int, str], column: str = "hmm_regime") -> pd.DataFrame:
    """Replace integer regime labels with human-readable names.

    Args:
        df: DataFrame containing an integer regime column (output of fit_hmm
            or fit_kmeans).
        mapping: Dict mapping integer state labels to descriptive names,
            e.g. {0: "Expansion", 1: "Recession", 2: "Overheating", 3: "Slowdown"}.
        column: Name of the column to relabel. Defaults to "hmm_regime".

    Returns:
        Copy of df with the specified column replaced by string regime names.
    """
    result = df.copy()
    result[column] = result[column].map(mapping)
    return result


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
