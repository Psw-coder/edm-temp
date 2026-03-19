import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None


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
            s=10,
            alpha=0.75,
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
            s=10,
            alpha=0.55,
            marker="o",
            linewidths=0,
        )

    if np.any(labeled_mask):
        ax.scatter(
            embedding_2d[labeled_mask, 0],
            embedding_2d[labeled_mask, 1],
            c=labels[labeled_mask],
            cmap=cmap,
            s=45,
            alpha=0.95,
            marker="*",
            edgecolors="black",
            linewidths=0.4,
        )


def save_visualization(
    diffusion_umap: np.ndarray,
    spectral_umap: np.ndarray,
    labels: np.ndarray,
    is_labeled: Optional[np.ndarray],
    out_path: str,
):
    unique_labels = np.unique(labels)
    cmap = "tab10" if unique_labels.size <= 10 else "tab20"
    vmin = int(unique_labels.min())
    vmax = int(unique_labels.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_with_label_style(
        ax=axes[0],
        embedding_2d=diffusion_umap,
        labels=labels,
        is_labeled=is_labeled,
        cmap=cmap,
    )
    axes[0].set_title("UMAP of diffusion latent $h_i^{diff}$")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].grid(alpha=0.2)

    _scatter_with_label_style(
        ax=axes[1],
        embedding_2d=spectral_umap,
        labels=labels,
        is_labeled=is_labeled,
        cmap=cmap,
    )
    axes[1].set_title("UMAP of spectral embedding $e_i^{spec}$")
    axes[1].set_xlabel("UMAP-1")
    axes[1].set_ylabel("UMAP-2")
    axes[1].grid(alpha=0.2)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Class label")

    if is_labeled is not None:
        legend_handles = [
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
        axes[1].legend(handles=legend_handles, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
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
    fig_path = os.path.join(args.output_dir, "umap_diffusion_vs_spectral.png")
    save_visualization(
        diffusion_umap=diffusion_umap,
        spectral_umap=spectral_umap,
        labels=labels,
        is_labeled=is_labeled,
        out_path=fig_path,
    )
    print(f"Visualization saved to: {fig_path}")

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
