import numpy as np
import os
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import glob
from tqdm import tqdm


def compute_twonn(data):
    """
    Compute the intrinsic dimension using the Two-NN algorithm.
    Ref: Facco E, D'Errico M, Rodriguez A, Laio A (2017) Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific Reports 7:12140
    """
    N = data.shape[0]
    if N < 3:
        return 0.0

    # Find 2 nearest neighbors (k=3 because the first one is the point itself)
    # n_jobs=-1 uses all available processors
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto', n_jobs=-1).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # distances[:, 0] is distance to self (should be 0)
    # distances[:, 1] is r1 (1st NN)
    # distances[:, 2] is r2 (2nd NN)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Filter out points where r1 is 0 (duplicates) to avoid division by zero
    mask = r1 > 0
    r1 = r1[mask]
    r2 = r2[mask]

    if len(r1) == 0:
        return 0.0

    mu = r2 / r1

    # Estimate intrinsic dimension
    # d = N / sum(ln(mu))
    # Note: N here is the number of valid points (after filtering)
    d = len(mu) / np.sum(np.log(mu))

    return d


def compute_local_pca_decay(data, k=50, n_components=50):
    """
    Compute Local PCA spectral decay.
    For each point, take k neighbors, compute PCA, and average the explained variance ratio.
    """
    N, D = data.shape
    if N < k:
        print(f"Warning: Not enough data for Local PCA (N={N} < k={k}). Using N-1.")
        k = N - 1

    n_components = min(n_components, k, D)

    # Fit Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1).fit(data)
    indices = nbrs.kneighbors(data, return_distance=False)

    explained_variance_ratios = []

    # We can subsample to speed up if N is very large
    sample_indices = range(N)
    if N > 2000:
        sample_indices = np.random.choice(N, 2000, replace=False)

    for i in sample_indices:
        # Local neighborhood
        local_data = data[indices[i]]
        # Center the data
        local_data = local_data - local_data.mean(axis=0)

        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(local_data)
        explained_variance_ratios.append(pca.explained_variance_ratio_)

    # Average the curves
    mean_decay = np.mean(np.array(explained_variance_ratios), axis=0)

    return mean_decay


def compute_geodesic_consistency(data, k=10, n_components=2):
    """
    Compute Geodesic Consistency using Isomap Reconstruction Error.
    This measures how well the geodesic distances are preserved in a low-dimensional Euclidean space.
    Lower reconstruction error implies better consistency with a flat manifold of dimension n_components.

    Returns:
        reconstruction_error: float
    """
    N = data.shape[0]
    # Subsample for Isomap as it's computationally expensive O(N^2) or O(N^3)
    if N > 2000:
        indices = np.random.choice(N, 2000, replace=False)
        data_sub = data[indices]
    else:
        data_sub = data

    try:
        iso = Isomap(n_neighbors=k, n_components=n_components)
        iso.fit(data_sub)
        return iso.reconstruction_error()
    except Exception as e:
        print(f"Isomap failed: {e}")
        return -1.0


def compute_local_consistency(data, labels, k=5):
    """
    Compute Local Neighborhood Class Consistency (LCC).

    This metric measures how well the class structure is preserved in the feature space.
    For each data point, it calculates the proportion of its k-nearest neighbors that share the same class label.
    A higher LCC indicates that points of the same class are clustered together.

    Args:
        data: (N, D) feature matrix
        labels: (N,) label array
        k: number of neighbors to consider (excluding self)

    Returns:
        float: Average consistency score (0.0 to 1.0)
    """
    N = data.shape[0]
    # Need at least k+1 points (1 for self, k for neighbors)
    if N < k + 1:
        print(f"Warning: Not enough data for LCC (N={N} < k+1={k + 1}). Returning 0.0")
        return 0.0

    # Fit Nearest Neighbors (k+1 to include self)
    # algorithm='auto' usually selects kd_tree or ball_tree for metric='minkowski'
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(data)
    indices = nbrs.kneighbors(data, return_distance=False)

    # indices[:, 0] is the point itself (distance 0), so we exclude it
    neighbor_indices = indices[:, 1:]

    # Get labels of neighbors
    # neighbor_labels shape: (N, k)
    neighbor_labels = labels[neighbor_indices]

    # Compare with query point labels
    # labels.reshape(-1, 1) makes it (N, 1) for broadcasting against (N, k)
    # matches is a boolean matrix (N, k)
    matches = (neighbor_labels == labels.reshape(-1, 1))

    # Calculate consistency for each point (mean across k neighbors)
    consistency_per_point = np.mean(matches, axis=1)

    # Average across the entire dataset
    mean_consistency = np.mean(consistency_per_point)

    return mean_consistency


def process_directory(base_dir, max_dim=None, metrics=None):
    # Find all all_features.npz files recursively
    # Structure: base_dir/epoch/all_features.npz

    if metrics is None:
        metrics = ['twonn', 'pca', 'geo', 'lcc']
    print(f"Calculating metrics: {', '.join(metrics)}")

    results = []
    # Store full decay curves separately
    decay_curves = {}

    # We look for immediate subdirectories that are numbers
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Try to sort by number if possible
    try:
        subdirs.sort(key=lambda x: int(x))
    except ValueError:
        subdirs.sort()

    print(f"Found {len(subdirs)} subdirectories in {base_dir}")
    if max_dim is not None:
        print(f"Limiting feature dimension to {max_dim}")

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        feature_file = os.path.join(subdir_path, "all_features.npz")

        if not os.path.exists(feature_file):
            continue

        try:
            print(f"Processing {subdir}...", end="", flush=True)
            data_file = np.load(feature_file)
            features = data_file['features']

            # Check if labels exist
            if 'labels' in data_file:
                labels = data_file['labels']
            else:
                labels = None

            # Flatten features if they are not 2D (N, D)
            if len(features.shape) > 2:
                features = features.reshape(features.shape[0], -1)

            original_dim = features.shape[1]
            print(f" (Shape: {features.shape})", end="", flush=True)

            if max_dim is not None and original_dim > max_dim:
                features = features[:, :max_dim]
                print(f" -> truncated to {features.shape}", end="", flush=True)

            result_entry = {'epoch': subdir}
            log_msg = " Done."

            # 1. Two-NN
            if 'twonn' in metrics:
                dim_twonn = compute_twonn(features)
                result_entry['twonn_dim'] = dim_twonn
                log_msg += f" Two-NN: {dim_twonn:.2f},"

            # 2. Local PCA Decay
            if 'pca' in metrics:
                # Use top 50 components
                decay_curve = compute_local_pca_decay(features, k=50, n_components=50)
                decay_curves[subdir] = decay_curve

                # Calculate dimension for 95% variance
                cum_var = np.cumsum(decay_curve)
                dim_pca_95 = np.searchsorted(cum_var, 0.95) + 1

                result_entry['pca_95_dim'] = dim_pca_95
                log_msg += f" PCA95: {dim_pca_95},"

            # 3. Geodesic Consistency (Isomap Error)
            if 'geo' in metrics:
                # Use intrinsic dimension estimate (rounded) or fixed 2 for visualization consistency
                # Here we use 2 as a standard baseline, or we could use int(dim_twonn)
                geo_consistency = compute_geodesic_consistency(features, k=15, n_components=2)
                result_entry['geodesic_consistency_err'] = geo_consistency
                log_msg += f" GeoErr: {geo_consistency:.4e},"

            # 4. Local Class Consistency (LCC)
            if 'lcc' in metrics:
                if labels is not None:
                    lcc_score = compute_local_consistency(features, labels, k=20)
                else:
                    lcc_score = -1.0
                result_entry['local_consistency'] = lcc_score
                log_msg += f" LCC: {lcc_score:.4f}"

            results.append(result_entry)

            print(log_msg.rstrip(','))

        except Exception as e:
            print(f"\nError processing {feature_file}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No results computed.")
        return

    # Save CSV results
    output_csv = os.path.join(base_dir, "analysis_results.csv")
    if results:
        fieldnames = ['epoch']
        if 'twonn' in metrics: fieldnames.append('twonn_dim')
        if 'pca' in metrics: fieldnames.append('pca_95_dim')
        if 'geo' in metrics: fieldnames.append('geodesic_consistency_err')
        if 'lcc' in metrics: fieldnames.append('local_consistency')

        with open(output_csv, 'w') as f:
            f.write(",".join(fieldnames) + "\n")
            for res in results:
                line = []
                for field in fieldnames:
                    line.append(str(res.get(field, '')))
                f.write(",".join(line) + "\n")

    # Save decay curves
    if 'pca' in metrics and decay_curves:
        output_curves = os.path.join(base_dir, "local_pca_curves.npz")
        np.savez(output_curves, **decay_curves)
        print(f"Full PCA decay curves saved to {output_curves}")

    print(f"\nSummary results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Intrinsic Dimension and Manifold Consistency metrics")
    parser.add_argument('--dir', type=str, help='Directory containing feature subfolders',default='/data/psw/edm/pseudo_label_output_uncond')
    parser.add_argument('--max_dim', type=int, default=None,
                        help='Maximum dimension of features to use. If features are longer, they will be truncated.')
    parser.add_argument('--metrics', nargs='+', default=['twonn', 'pca', 'geo', 'lcc'],
                        help='List of metrics to calculate. Options: twonn, pca, geo, lcc. Default: all.')

    args = parser.parse_args()

    process_directory(args.dir, args.max_dim, args.metrics)
