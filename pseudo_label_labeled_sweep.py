import csv
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import click
import numpy as np
import torch
from tqdm import tqdm


def parse_int_sweep_values(spec: str) -> List[int]:
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("sweep spec is empty")

    values: List[int] = []
    if ":" in spec:
        parts = [part.strip() for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"invalid range spec: {spec}")
        start, stop, step = (int(part) for part in parts)
        if step == 0:
            raise ValueError("range step must not be zero")
        if start <= stop and step < 0:
            raise ValueError("range step must be positive for ascending ranges")
        if start >= stop and step > 0:
            step = -step
        end = stop + (1 if step > 0 else -1)
        values = list(range(start, end, step))
    else:
        values = [int(part.strip()) for part in spec.split(",") if part.strip()]

    if not values:
        raise ValueError(f"failed to parse any values from: {spec}")
    if any(value <= 0 for value in values):
        raise ValueError("all sweep values must be positive integers")
    return values


def infer_num_classes(dataset_type: str) -> int:
    return 10 if dataset_type in {"cifar10", "fashion-mnist", "mnist", "svhn", "stl10"} else 100


def split_labeled_unlabeled_indices_per_class(
    labels: np.ndarray,
    labeled_per_class: int,
    num_classes: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labeled_per_class <= 0:
        raise ValueError("labeled_per_class must be positive")

    rng = np.random.default_rng(seed)
    labeled_parts: List[np.ndarray] = []
    unlabeled_parts: List[np.ndarray] = []

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.shape[0] < labeled_per_class:
            raise ValueError(
                f"class {class_id} only has {class_indices.shape[0]} samples, "
                f"but labeled_per_class={labeled_per_class}"
            )
        shuffled = rng.permutation(class_indices)
        labeled_parts.append(shuffled[:labeled_per_class])
        unlabeled_parts.append(shuffled[labeled_per_class:])

    labeled_indices = np.concatenate(labeled_parts, axis=0)
    unlabeled_indices = np.concatenate(unlabeled_parts, axis=0)
    labeled_indices = rng.permutation(labeled_indices).astype(np.int64)
    unlabeled_indices = rng.permutation(unlabeled_indices).astype(np.int64)
    return labeled_indices, unlabeled_indices


def _extract_features_in_batches(
    images: np.ndarray,
    batch_size: int,
    extractor: Callable[[np.ndarray], np.ndarray],
    desc: str,
) -> np.ndarray:
    feature_batches: List[np.ndarray] = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(images), batch_size), total=num_batches, desc=desc, unit="batch"):
        batch = images[start:start + batch_size]
        feature_batches.append(extractor(batch))
    return np.concatenate(feature_batches, axis=0)


def extract_all_features(
    images: np.ndarray,
    feature_backbone: str,
    network_pkl: Optional[str],
    batch_size: int,
    dataset_type: str,
    acgan_ckpt: Optional[str] = None,
    acgan_layer: str = "flatten",
    acgan_latent_dim: Optional[int] = None,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    num_steps: int = 100,
    rho: float = 7.0,
    step_idx_values: Optional[Sequence[int]] = None,
) -> np.ndarray:
    import pseudo_label as pl

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step_idx = list(step_idx_values) if step_idx_values is not None else [77]

    if feature_backbone == "diffusion":
        if not network_pkl:
            raise click.ClickException("--network_pkl is required for diffusion features")
        net = pl.load_network(network_pkl, device)
        sigma_schedule = pl.generate_sigma_schedule(sigma_min, sigma_max, num_steps, rho)
        sigma = [float(sigma_schedule[idx].item()) for idx in step_idx]
        print(f"Using diffusion step_idx={step_idx}, sigma={sigma}")
        return _extract_features_in_batches(
            images=images,
            batch_size=batch_size,
            extractor=lambda batch: pl.extract_features_with_noise(net, batch, sigma, step_idx, device),
            desc="Extract diffusion features",
        )

    if feature_backbone != "acgan":
        raise click.ClickException(f"unsupported feature_backbone: {feature_backbone}")
    if acgan_ckpt is None:
        raise click.ClickException("--acgan_ckpt is required for ACGAN features")

    num_classes = infer_num_classes(dataset_type)
    img_size = int(images.shape[2]) if images.ndim == 4 else 32
    img_channels = 1 if dataset_type in {"mnist", "fashion-mnist"} else 3
    acgan_model = pl.load_acgan_model(
        acgan_ckpt_path=acgan_ckpt,
        device=device,
        img_channels=img_channels,
        num_classes=num_classes,
        img_size=img_size,
        latent_dim=acgan_latent_dim,
    )
    print(f"Using ACGAN layer={acgan_layer}")
    return _extract_features_in_batches(
        images=images,
        batch_size=batch_size,
        extractor=lambda batch: pl.extract_features_with_acgan(acgan_model, batch, layer=acgan_layer, device=device),
        desc="Extract ACGAN features",
    )


def sweep_pseudo_label_accuracy_by_labeled_per_class(
    all_features: np.ndarray,
    labels: np.ndarray,
    labeled_per_class_values: Sequence[int],
    num_classes: int,
    similarity_method: str,
    cosine_threshold: float,
    distance_threshold: float,
    device: str,
    use_separate_clustering: bool,
    unlabeled_cluster_ratio: float,
    min_unlabeled_clusters: int,
    max_unlabeled_clusters: int,
    pseudo_label_fn: Optional[Callable[..., Tuple[np.ndarray, np.ndarray, Dict[int, int]]]] = None,
    evaluate_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    seed: int = 0,
) -> List[Dict[str, float]]:
    if pseudo_label_fn is None or evaluate_fn is None:
        import pseudo_label as pl

        if pseudo_label_fn is None:
            pseudo_label_fn = pl.perform_clustering_and_pseudo_labeling
        if evaluate_fn is None:
            evaluate_fn = pl.evaluate_pseudo_label_assignment

    labels = np.asarray(labels, dtype=np.int64)
    all_features = np.asarray(all_features)
    results: List[Dict[str, float]] = []

    for offset, labeled_per_class in enumerate(labeled_per_class_values):
        labeled_indices, unlabeled_indices = split_labeled_unlabeled_indices_per_class(
            labels=labels,
            labeled_per_class=int(labeled_per_class),
            num_classes=num_classes,
            seed=seed + offset,
        )
        labeled_features = all_features[labeled_indices]
        labeled_labels = labels[labeled_indices]
        unlabeled_features = all_features[unlabeled_indices]
        unlabeled_labels = labels[unlabeled_indices]

        unlabeled_n_clusters = None
        if use_separate_clustering:
            target_clusters = int(len(unlabeled_features) * unlabeled_cluster_ratio)
            unlabeled_n_clusters = max(
                min_unlabeled_clusters,
                min(max_unlabeled_clusters, target_clusters),
            )

        pseudo_labels, _, _ = pseudo_label_fn(
            labeled_features,
            labeled_labels,
            unlabeled_features,
            num_classes=num_classes,
            save_visualization=None,
            similarity_method=similarity_method,
            cosine_threshold=cosine_threshold,
            distance_threshold=distance_threshold,
            device=device,
            use_separate_clustering=use_separate_clustering,
            unlabeled_n_clusters=unlabeled_n_clusters,
        )
        metrics = dict(evaluate_fn(pseudo_labels, unlabeled_labels))
        result = {
            "labeled_per_class": int(labeled_per_class),
            "labeled_count": int(len(labeled_indices)),
            "unlabeled_count": int(len(unlabeled_indices)),
            "labeled_ratio": float(len(labeled_indices) / len(labels)),
            "unlabeled_n_clusters": int(unlabeled_n_clusters) if unlabeled_n_clusters is not None else None,
            "accuracy": float(metrics["accuracy"]),
            "valid_count": int(metrics["valid_count"]),
            "total_count": int(metrics["total_count"]),
            "discard_rate": float(metrics["discard_rate"]),
        }
        results.append(result)
        print(
            f"labeled_per_class={result['labeled_per_class']}, "
            f"labeled={result['labeled_count']}, "
            f"unlabeled={result['unlabeled_count']}, "
            f"K={result['unlabeled_n_clusters']}, "
            f"acc={result['accuracy']:.4f}, "
            f"valid={result['valid_count']}/{result['total_count']}, "
            f"discard={result['discard_rate']:.4f}"
        )

    return results


def save_results(results: Sequence[Dict[str, float]], outdir: str) -> Tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, "labeled_per_class_sweep_results.json")
    csv_path = os.path.join(outdir, "labeled_per_class_sweep_results.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(results), f, indent=2, ensure_ascii=False)

    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    return json_path, csv_path


@click.command()
@click.option("--data", type=str, required=True, help="Dataset path, same format as pseudo_label.py")
@click.option("--outdir", type=str, default="pseudo_label_labeled_sweep_output", show_default=True)
@click.option("--network_pkl", type=str, default=None, help="Diffusion network pickle path")
@click.option("--feature_backbone", type=click.Choice(["diffusion", "acgan"]), default="diffusion", show_default=True)
@click.option("--acgan_ckpt", type=str, default=None, help="ACGAN checkpoint path")
@click.option("--acgan_layer", type=str, default="flatten", show_default=True)
@click.option("--acgan_latent_dim", type=int, default=None)
@click.option(
    "--dataset_type",
    type=click.Choice(["cifar10", "cifar100", "fashion-mnist", "mnist", "svhn", "stl10"]),
    default="cifar10",
    show_default=True,
)
@click.option("--max_images", type=int, default=50000, show_default=True)
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option(
    "--labeled_per_class_values",
    type=str,
    required=True,
    help="Comma list or range start:end:step, e.g. 1,5,10,20 or 10:100:10",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--similarity_method", type=click.Choice(["cosine", "distance"]), default="cosine", show_default=True)
@click.option("--cosine_threshold", type=float, default=0.3, show_default=True)
@click.option("--distance_threshold", type=float, default=3200.0, show_default=True)
@click.option("--unlabeled_cluster_ratio", type=float, default=2.0, show_default=True)
@click.option("--min_unlabeled_clusters", type=int, default=10, show_default=True)
@click.option("--max_unlabeled_clusters", type=int, default=500, show_default=True)
@click.option("--use_separate_clustering/--no_use_separate_clustering", default=True, show_default=True)
@click.option("--sigma_max", type=float, default=80.0, show_default=True)
@click.option("--sigma_min", type=float, default=0.002, show_default=True)
@click.option("--num_steps", type=int, default=100, show_default=True)
@click.option("--rho", type=float, default=7.0, show_default=True)
@click.option("--step_idx_values", type=str, default="77", show_default=True, help="Comma list or range for diffusion step indices")
def main(
    data: str,
    outdir: str,
    network_pkl: Optional[str],
    feature_backbone: str,
    acgan_ckpt: Optional[str],
    acgan_layer: str,
    acgan_latent_dim: Optional[int],
    dataset_type: str,
    max_images: int,
    batch_size: int,
    labeled_per_class_values: str,
    seed: int,
    similarity_method: str,
    cosine_threshold: float,
    distance_threshold: float,
    unlabeled_cluster_ratio: float,
    min_unlabeled_clusters: int,
    max_unlabeled_clusters: int,
    use_separate_clustering: bool,
    sigma_max: float,
    sigma_min: float,
    num_steps: int,
    rho: float,
    step_idx_values: str,
) -> None:
    import pseudo_label as pl

    labeled_counts = parse_int_sweep_values(labeled_per_class_values)
    step_indices = parse_int_sweep_values(step_idx_values)
    num_classes = infer_num_classes(dataset_type)

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading {dataset_type} data...")
    images, labels = pl.load_dataset_data(data, dataset_type, max_images=max_images)
    labels = labels.astype(np.int64)
    print(f"Loaded images={images.shape}, labels={labels.shape}")

    class_counts = np.bincount(labels, minlength=num_classes)
    max_supported = int(class_counts.min())
    if max(labeled_counts) > max_supported:
        raise click.ClickException(
            f"requested labeled_per_class up to {max(labeled_counts)}, "
            f"but the smallest class only has {max_supported} samples"
        )

    features = extract_all_features(
        images=images,
        feature_backbone=feature_backbone,
        network_pkl=network_pkl,
        batch_size=batch_size,
        dataset_type=dataset_type,
        acgan_ckpt=acgan_ckpt,
        acgan_layer=acgan_layer,
        acgan_latent_dim=acgan_latent_dim,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        num_steps=num_steps,
        rho=rho,
        step_idx_values=step_indices,
    )
    print(f"Feature extraction finished: {features.shape}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = sweep_pseudo_label_accuracy_by_labeled_per_class(
        all_features=features,
        labels=labels,
        labeled_per_class_values=labeled_counts,
        num_classes=num_classes,
        similarity_method=similarity_method,
        cosine_threshold=cosine_threshold,
        distance_threshold=distance_threshold,
        device=device,
        use_separate_clustering=use_separate_clustering,
        unlabeled_cluster_ratio=unlabeled_cluster_ratio,
        min_unlabeled_clusters=min_unlabeled_clusters,
        max_unlabeled_clusters=max_unlabeled_clusters,
        seed=seed,
    )

    json_path, csv_path = save_results(results, outdir)
    print(f"Saved JSON results to: {json_path}")
    print(f"Saved CSV results to: {csv_path}")


if __name__ == "__main__":
    main()
