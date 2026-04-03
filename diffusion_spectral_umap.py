import argparse
import json
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None


CM_PER_INCH = 2.54
DEFAULT_FIGURE_WIDTH_CM = 7.2 * CM_PER_INCH
DEFAULT_FIGURE_HEIGHT_CM = 5.8 * CM_PER_INCH
DEFAULT_REGION_ALPHA = 0.4
DEFAULT_REGION_LIGHTEN_RATIO = 0.5


def _lighten_rgba(color: Tuple[float, float, float, float], blend_ratio: float) -> Tuple[float, float, float, float]:
    blend_ratio = float(np.clip(blend_ratio, 0.0, 1.0))
    rgb = np.asarray(color[:3], dtype=np.float32)
    light_rgb = rgb * (1.0 - blend_ratio) + blend_ratio
    return float(light_rgb[0]), float(light_rgb[1]), float(light_rgb[2]), 1.0


def _compute_background_labels(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    unique_labels = np.unique(labels)
    class_centers = np.stack([embedding_2d[labels == label].mean(axis=0) for label in unique_labels], axis=0)

    grid_points = np.stack([x_grid, y_grid], axis=-1)
    distances_sq = np.sum((grid_points[..., None, :] - class_centers[None, None, :, :]) ** 2, axis=-1)
    return np.argmin(distances_sq, axis=-1).astype(np.int64)


def _draw_class_regions(
    ax: plt.Axes,
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
    cmap: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    region_alpha: float,
):
    if unique_labels.size == 0 or region_alpha <= 0:
        return

    norm = plt.Normalize(vmin=int(unique_labels.min()), vmax=int(unique_labels.max()))
    colormap = plt.get_cmap(cmap)
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_limits[0], x_limits[1], 300, dtype=np.float32),
        np.linspace(y_limits[0], y_limits[1], 300, dtype=np.float32),
    )
    background_labels = _compute_background_labels(embedding_2d, labels, x_grid, y_grid)
    region_colors = [
        _lighten_rgba(colormap(norm(int(label))), blend_ratio=DEFAULT_REGION_LIGHTEN_RATIO)
        for label in unique_labels
    ]

    ax.contourf(
        x_grid,
        y_grid,
        background_labels,
        levels=np.arange(unique_labels.size + 1) - 0.5,
        colors=region_colors,
        alpha=region_alpha,
        antialiased=True,
        zorder=0,
    )


def _build_label_state_legend_handles():
    return [
        Line2D([0], [0], marker="o", color="w", label="Unlabeled", markerfacecolor="gray", markersize=7),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Labeled",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=11,
        ),
    ]


def _apply_plot_text(ax: plt.Axes, title: str, show_legend: bool):
    if show_legend:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP-1", fontsize=11)
        ax.set_ylabel("UMAP-2", fontsize=11)
    else:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")


def _apply_auxiliary_artists(
    fig: plt.Figure,
    ax: plt.Axes,
    show_legend: bool,
    is_labeled: Optional[np.ndarray],
    cmap: str,
    vmin: int,
    vmax: int,
):
    if show_legend:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.96, pad=0.015)
        cbar.set_label("Class label")

    if is_labeled is not None:
        ax.legend(
        handles=_build_label_state_legend_handles(),
        loc="upper right",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
        )


def load_feature_pack(input_path: str, metadata_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    data = np.load(input_path)
    if "features" not in data or "labels" not in data:
        raise KeyError(f"{input_path} must contain keys 'features' and 'labels'.")

    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    inferred_metadata = None
    if metadata_path:
        inferred_metadata = metadata_path
    elif input_path.endswith("all_features.npz"):
        inferred_metadata = input_path.replace("all_features.npz", "all_features_metadata.npz")
    else:
        candidate = os.path.join(os.path.dirname(input_path), "all_features_metadata.npz")
        if os.path.exists(candidate):
            inferred_metadata = candidate

    is_labeled = None
    if inferred_metadata and os.path.exists(inferred_metadata):
        metadata = np.load(inferred_metadata)
        if "is_labeled" in metadata:
            is_labeled = metadata["is_labeled"].astype(np.int64)

    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same number of samples.")
    if is_labeled is not None and is_labeled.shape[0] != features.shape[0]:
        raise ValueError("is_labeled must have the same number of samples as features.")

    return features, labels, is_labeled


def maybe_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    is_labeled: Optional[np.ndarray],
    limit: int,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if limit <= 0 or features.shape[0] <= limit:
        return features, labels, is_labeled

    rng = np.random.default_rng(random_seed)
    index = rng.choice(features.shape[0], size=limit, replace=False)
    index.sort()
    features = features[index]
    labels = labels[index]
    if is_labeled is not None:
        is_labeled = is_labeled[index]
    return features, labels, is_labeled


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def build_knn_rbf_graph(features_l2: np.ndarray, k: int, tau: float) -> Tuple[sparse.csr_matrix, float, int]:
    n_samples = features_l2.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to build a graph.")

    k_eff = max(1, min(k, n_samples - 1))
    knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    knn.fit(features_l2)
    distances, indices = knn.kneighbors(features_l2, return_distance=True)

    # Drop self-neighbor in the first column.
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    distances_sq = distances ** 2

    if tau > 0:
        tau_used = float(tau)
    else:
        positive_dist_sq = distances_sq[distances_sq > 0]
        tau_used = float(np.median(positive_dist_sq)) if positive_dist_sq.size > 0 else 1.0
        tau_used = max(tau_used, 1e-12)

    weights = np.exp(-distances_sq / tau_used)
    rows = np.repeat(np.arange(n_samples), k_eff)
    cols = indices.reshape(-1)
    vals = weights.reshape(-1)

    # Use union kNN graph: edge(i, j) exists if j in Nk(i) or i in Nk(j).
    w_directed = sparse.coo_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)).tocsr()
    w_sym = w_directed.maximum(w_directed.T)
    w_sym.setdiag(0.0)
    w_sym.eliminate_zeros()
    return w_sym, tau_used, k_eff


def build_normalized_operator(w: sparse.csr_matrix) -> Tuple[sparse.csr_matrix, np.ndarray]:
    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    degree_safe = np.maximum(degree, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(degree_safe)
    d_inv_sqrt_mat = sparse.diags(d_inv_sqrt)
    s = d_inv_sqrt_mat @ w @ d_inv_sqrt_mat
    return s.tocsr(), degree


def spectral_embedding_from_operator(
    s: sparse.csr_matrix,
    spectral_dim: int,
    drop_first: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = s.shape[0]
    max_k = max(1, n_samples - 1)
    extra = 1 if drop_first else 0
    k_total = min(spectral_dim + extra, max_k)
    if k_total < 1:
        raise ValueError("Invalid eigen decomposition setting.")

    eigvals, eigvecs = eigsh(s.asfptype(), k=k_total, which="LA", tol=1e-4)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    start_idx = 1 if drop_first and eigvecs.shape[1] > 1 else 0
    end_idx = min(start_idx + spectral_dim, eigvecs.shape[1])
    embedding = eigvecs[:, start_idx:end_idx].astype(np.float32)
    return embedding, eigvals.astype(np.float32), eigvecs.astype(np.float32)


def run_umap(
    features: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_seed: int,
) -> np.ndarray:
    if umap is None:
        raise ImportError("umap-learn is not installed. Please run: pip install umap-learn")

    if features.shape[0] < 3:
        raise ValueError("UMAP needs at least 3 samples.")

    n_neighbors_eff = max(2, min(n_neighbors, features.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_seed,
    )
    return reducer.fit_transform(features).astype(np.float32)


def compute_knn_label_consistency(features: np.ndarray, labels: np.ndarray, k: int) -> float:
    n_samples = features.shape[0]
    if n_samples < 2:
        return float("nan")

    k_eff = max(1, min(k, n_samples - 1))
    knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    knn.fit(features)
    neighbor_idx = knn.kneighbors(return_distance=False)[:, 1:]
    neighbor_labels = labels[neighbor_idx]
    same_label_ratio = (neighbor_labels == labels[:, None]).mean(axis=1)
    return float(np.mean(same_label_ratio))


def compute_quality_metrics(features: np.ndarray, labels: np.ndarray, knn_k: int) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    metrics = {
        "silhouette_score": float("nan"),
        "davies_bouldin_index": float("nan"),
        "calinski_harabasz_index": float("nan"),
        "knn_label_consistency": float("nan"),
    }

    # Silhouette/DBI/CH need at least 2 clusters and enough samples.
    if features.shape[0] >= 3 and unique_labels.shape[0] >= 2:
        try:
            metrics["silhouette_score"] = float(silhouette_score(features, labels))
        except Exception:
            pass

        try:
            metrics["davies_bouldin_index"] = float(davies_bouldin_score(features, labels))
        except Exception:
            pass

        try:
            metrics["calinski_harabasz_index"] = float(calinski_harabasz_score(features, labels))
        except Exception:
            pass

    metrics["knn_label_consistency"] = compute_knn_label_consistency(features, labels, k=knn_k)
    return metrics


def print_and_save_metrics(metrics_dict: Dict[str, Dict[str, float]], output_dir: str):
    print("\n=== Quality Metrics ===")
    for space_name, metric_values in metrics_dict.items():
        print(f"[{space_name}]")
        print(f"  Silhouette Score:         {metric_values['silhouette_score']:.6f}")
        print(f"  Davies-Bouldin Index:     {metric_values['davies_bouldin_index']:.6f}")
        print(f"  Calinski-Harabasz Index:  {metric_values['calinski_harabasz_index']:.6f}")
        print(f"  kNN label consistency:    {metric_values['knn_label_consistency']:.6f}")

    metrics_path = os.path.join(output_dir, "quality_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {metrics_path}")


def _scatter_with_label_style(
    ax: plt.Axes,
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    is_labeled: Optional[np.ndarray],
    cmap: str,
):
    if is_labeled is None:
        ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap=cmap,
            s=20,
            alpha=0.8,
            linewidths=0,
        )
        return

    unlabeled_mask = is_labeled == 0
    labeled_mask = is_labeled == 1

    if np.any(unlabeled_mask):
        ax.scatter(
            embedding_2d[unlabeled_mask, 0],
            embedding_2d[unlabeled_mask, 1],
            c=labels[unlabeled_mask],
            cmap=cmap,
            s=18,
            alpha=0.62,
            marker="o",
            linewidths=0,
        )

    if np.any(labeled_mask):
        ax.scatter(
            embedding_2d[labeled_mask, 0],
            embedding_2d[labeled_mask, 1],
            c=labels[labeled_mask],
            cmap=cmap,
            s=78,
            alpha=0.95,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
        )


def save_visualization(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    is_labeled: Optional[np.ndarray],
    title: str,
    out_path: str,
    show_legend: bool,
    figure_width_cm: float,
    figure_height_cm: float,
    fill_class_regions: bool,
    region_alpha: float,
):
    unique_labels = np.unique(labels)
    cmap = "tab10" if unique_labels.size <= 10 else "tab20"
    vmin = int(unique_labels.min())
    vmax = int(unique_labels.max())
    x = embedding_2d[:, 0]
    y = embedding_2d[:, 1]
    x_span = float(np.max(x) - np.min(x))
    y_span = float(np.max(y) - np.min(y))
    x_pad = max(x_span * 0.01, 1e-6)
    y_pad = max(y_span * 0.01, 1e-6)
    x_limits = (float(np.min(x) - x_pad), float(np.max(x) + x_pad))
    y_limits = (float(np.min(y) - y_pad), float(np.max(y) + y_pad))

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(figure_width_cm / CM_PER_INCH, figure_height_cm / CM_PER_INCH),
    )

    if fill_class_regions:
        _draw_class_regions(
            ax=ax,
            embedding_2d=embedding_2d,
            labels=labels,
            unique_labels=unique_labels,
            cmap=cmap,
            x_limits=x_limits,
            y_limits=y_limits,
            region_alpha=region_alpha,
        )

    _scatter_with_label_style(
        ax=ax,
        embedding_2d=embedding_2d,
        labels=labels,
        is_labeled=is_labeled,
        cmap=cmap,
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    _apply_plot_text(ax=ax, title=title, show_legend=show_legend)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.2)
    ax.margins(x=0, y=0)

    _apply_auxiliary_artists(
        fig=fig,
        ax=ax,
        show_legend=show_legend,
        is_labeled=is_labeled,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    fig.tight_layout(pad=0.08)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_numeric_outputs(
    output_dir: str,
    features: np.ndarray,
    features_l2: np.ndarray,
    labels: np.ndarray,
    is_labeled: Optional[np.ndarray],
    w: sparse.csr_matrix,
    s: sparse.csr_matrix,
    degree: np.ndarray,
    tau_used: float,
    k_used: int,
    spectral_embedding: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    diffusion_umap: np.ndarray,
    spectral_umap: np.ndarray,
):
    os.makedirs(output_dir, exist_ok=True)

    sparse.save_npz(os.path.join(output_dir, "graph_W_knn_rbf.npz"), w)
    sparse.save_npz(os.path.join(output_dir, "operator_S.npz"), s)

    pack = {
        "diffusion_latent": features.astype(np.float32),
        "diffusion_latent_l2": features_l2.astype(np.float32),
        "labels": labels.astype(np.int64),
        "degree": degree.astype(np.float32),
        "tau": np.array([tau_used], dtype=np.float32),
        "k": np.array([k_used], dtype=np.int32),
        "spectral_embedding": spectral_embedding.astype(np.float32),
        "eigenvalues": eigenvalues.astype(np.float32),
        "eigenvectors": eigenvectors.astype(np.float32),
        "umap_diffusion_2d": diffusion_umap.astype(np.float32),
        "umap_spectral_2d": spectral_umap.astype(np.float32),
    }
    if is_labeled is not None:
        pack["is_labeled"] = is_labeled.astype(np.int64)

    np.savez_compressed(os.path.join(output_dir, "spectral_pipeline_outputs.npz"), **pack)


def parse_args():
    parser = argparse.ArgumentParser(
        description="From diffusion latent features to graph operator, spectral decomposition, and UMAP visualization."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input feature pack .npz, e.g. pseudo_label_output/all_features.npz",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="",
        help="Optional metadata .npz path containing 'is_labeled'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="spectral_analysis_output",
        help="Directory for all outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional sample cap for fast testing. 0 means use all samples.",
    )
    parser.add_argument("--k", type=int, default=15, help="k in kNN graph.")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="RBF denominator in exp(-||zi-zj||^2/tau). <=0 means auto median of neighbor dist^2.",
    )
    parser.add_argument(
        "--spectral_dim",
        type=int,
        default=16,
        help="Target spectral embedding dimension r.",
    )
    parser.add_argument(
        "--keep_first_eigvec",
        action="store_true",
        help="Keep the largest eigenvector (default behavior is to drop the trivial first eigenvector).",
    )
    parser.add_argument("--umap_neighbors", type=int, default=30, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument(
        "--figure_width_cm",
        type=float,
        default=DEFAULT_FIGURE_WIDTH_CM,
        help="Saved figure width in centimeters.",
    )
    parser.add_argument(
        "--figure_height_cm",
        type=float,
        default=DEFAULT_FIGURE_HEIGHT_CM,
        help="Saved figure height in centimeters.",
    )
    parser.add_argument(
        "--fill_class_regions",
        action="store_true",
        help="Fill the plot background with light nearest-class regions.",
    )
    parser.add_argument(
        "--region_alpha",
        type=float,
        default=DEFAULT_REGION_ALPHA,
        help="Alpha for filled class regions.",
    )
    parser.add_argument(
        "--no_legend",
        action="store_true",
        help="Disable plot legends (class colorbar and labeled/unlabeled marker legend).",
    )
    parser.add_argument("--metric_knn_k", type=int, default=10, help="k for kNN label consistency metric.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Step 1: Load diffusion latent features ===")
    features, labels, is_labeled = load_feature_pack(
        input_path=args.input,
        metadata_path=args.metadata if args.metadata else None,
    )
    print(f"Loaded features: {features.shape}, labels: {labels.shape}")
    if is_labeled is not None:
        print(f"Loaded is_labeled metadata: {is_labeled.shape}")

    features, labels, is_labeled = maybe_subsample(
        features=features,
        labels=labels,
        is_labeled=is_labeled,
        limit=args.limit,
        random_seed=args.random_seed,
    )
    if args.limit > 0:
        print(f"After subsample: {features.shape}")

    print("=== Step 2: L2 normalize diffusion latent h_i^diff ===")
    features_l2 = l2_normalize(features)
    print("L2 normalization done.")

    print("=== Step 3: Build kNN + RBF graph W ===")
    w, tau_used, k_used = build_knn_rbf_graph(
        features_l2=features_l2,
        k=args.k,
        tau=args.tau,
    )
    print(f"Graph built: shape={w.shape}, nnz={w.nnz}, k={k_used}, tau={tau_used:.6e}")

    print("=== Step 4: Build S = D^(-1/2) W D^(-1/2), then spectral decomposition ===")
    s, degree = build_normalized_operator(w)
    spectral_embed, eigvals, eigvecs = spectral_embedding_from_operator(
        s=s,
        spectral_dim=args.spectral_dim,
        drop_first=(not args.keep_first_eigvec),
    )
    print(f"Spectral embedding shape: {spectral_embed.shape}")
    print(f"Top eigenvalues: {eigvals[: min(5, eigvals.shape[0])]}")

    print("=== Step 5: UMAP 2D visualization for h_i^diff and e_i^spec ===")
    diffusion_umap = run_umap(
        features=features_l2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_seed=args.random_seed,
    )
    spectral_umap = run_umap(
        features=spectral_embed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_seed=args.random_seed,
    )
    diffusion_fig_path = os.path.join(args.output_dir, "umap_diffusion.png")
    spectral_fig_path = os.path.join(args.output_dir, "umap_spectral.png")
    save_visualization(
        embedding_2d=diffusion_umap,
        labels=labels,
        is_labeled=is_labeled,
        title="UMAP of diffusion latent $h_i^{diff}$",
        out_path=diffusion_fig_path,
        show_legend=(not args.no_legend),
        figure_width_cm=args.figure_width_cm,
        figure_height_cm=args.figure_height_cm,
        fill_class_regions=args.fill_class_regions,
        region_alpha=args.region_alpha,
    )
    save_visualization(
        embedding_2d=spectral_umap,
        labels=labels,
        is_labeled=is_labeled,
        title="UMAP of spectral embedding $e_i^{spec}$",
        out_path=spectral_fig_path,
        show_legend=(not args.no_legend),
        figure_width_cm=args.figure_width_cm,
        figure_height_cm=args.figure_height_cm,
        fill_class_regions=args.fill_class_regions,
        region_alpha=args.region_alpha,
    )
    print(f"Visualization saved to: {diffusion_fig_path}")
    print(f"Visualization saved to: {spectral_fig_path}")

    metrics = {
        "diffusion_latent_l2": compute_quality_metrics(features_l2, labels, knn_k=args.metric_knn_k),
        "spectral_embedding": compute_quality_metrics(spectral_embed, labels, knn_k=args.metric_knn_k),
        "umap_diffusion_2d": compute_quality_metrics(diffusion_umap, labels, knn_k=args.metric_knn_k),
        "umap_spectral_2d": compute_quality_metrics(spectral_umap, labels, knn_k=args.metric_knn_k),
    }
    print_and_save_metrics(metrics, args.output_dir)

    save_numeric_outputs(
        output_dir=args.output_dir,
        features=features,
        features_l2=features_l2,
        labels=labels,
        is_labeled=is_labeled,
        w=w,
        s=s,
        degree=degree,
        tau_used=tau_used,
        k_used=k_used,
        spectral_embedding=spectral_embed,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        diffusion_umap=diffusion_umap,
        spectral_umap=spectral_umap,
    )
    print(f"All numeric outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
'''
python diffusion_spectral_umap.py \
  --input pseudo_label_output_cifar100/all_features.npz   \
  --output_dir spectral_analysis_output_cifar100 \
  --figure_width_cm 12 \
  --figure_height_cm 10 \
  --fill_class_regions \
  --umap_min_dist 0.4 \
  --no_legend
 

python diffusion_spectral_umap.py   --input pseudo_label_output_mnist/all_features.npz     \
--output_dir spectral_analysis_output_mnist \
--k 15 \
--spectral_dim 12
'''

