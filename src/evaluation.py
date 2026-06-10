"""
evaluation.py - Model Evaluation Metrics for the Laptop Recommendation System

Provides clustering quality metrics and Elbow/Silhouette analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compute_elbow_data(X_scaled: np.ndarray, k_range: range = range(2, 11)):
    """Compute inertia and silhouette scores for a range of k values."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def compute_cluster_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all clustering quality metrics for a given labelling."""
    return {
        "silhouette": round(silhouette_score(X_scaled, labels), 4),
        "davies_bouldin": round(davies_bouldin_score(X_scaled, labels), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_scaled, labels), 4),
    }


def evaluate_pipeline(df: pd.DataFrame, km_model: KMeans, km_scaler: MinMaxScaler):
    """
    Full evaluation: Elbow analysis + current-model metrics.
    Returns dict with 'elbow_data' and 'current_metrics', or None if no valid models.
    """
    if km_model is None or km_scaler is None:
        return None
        
    df_valid = df.dropna(subset=["Fiyat_TL"])
    if len(df_valid) < 3: # Not enough data for clustering metrics
        return None
        
    X = df_valid[["Fiyat_TL", "Total_Raw_Power"]].values
    X_scaled = km_scaler.transform(X)
    labels = km_model.predict(X_scaled)

    elbow = compute_elbow_data(X_scaled)
    # Check if multiple labels exist to prevent errors in metrics
    if len(set(labels)) > 1:
        metrics = compute_cluster_metrics(X_scaled, labels)
    else:
        metrics = {"silhouette": 0, "davies_bouldin": 0, "calinski_harabasz": 0}
        
    metrics["inertia"] = round(km_model.inertia_, 4)
    metrics["n_clusters"] = km_model.n_clusters

    return {"elbow_data": elbow, "current_metrics": metrics}
