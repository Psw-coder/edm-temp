import json
import os
import random
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import torch
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from pseudo_label import (
    build_spectral_features_joint,
    extract_features_with_acgan,
    extract_features_with_noise,
    generate_sigma_schedule,
    load_acgan_model,
    load_dataset_data,
    load_network,
    split_labeled_unlabeled,
)


TABLE_ROWS = [
    ("Classifier Feature", "classifier_feature"),
    ("Pixel Space", "pixel_space"),
    ("Diffusion Latent", "diffusion_latent"),
    ("GAN Spectral", "gan_spectral"),
    ("Diffusion Spectral", "diffusion_spectral"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


def compute_knn_consistency(features: np.ndarray, labels: np.ndarray, k: int) -> float:
    n_samples = int(features.shape[0])
    if n_samples < 2:
        return float("nan")
    k_eff = max(1, min(int(k), n_samples - 1))
    knn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    knn.fit(features)
    indices = knn.kneighbors(return_distance=False)[:, 1:]
    neighbor_labels = labels[indices]
    same = (neighbor_labels == labels[:, None]).mean(axis=1)
    return float(np.mean(same))


def compute_representation_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    knn_k: int,
    silhouette_max_samples: int,
    random_seed: int,
) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    unique_labels = np.unique(labels)
    metrics = {
        "silhouette": float("nan"),
        "dbi": float("nan"),
        "chi": float("nan"),
        "knn_consistency": float("nan"),
    }

    if features.shape[0] >= 3 and unique_labels.shape[0] >= 2:
        sil_sample = None
        if silhouette_max_samples > 0 and features.shape[0] > silhouette_max_samples:
            sil_sample = int(silhouette_max_samples)
        try:
            metrics["silhouette"] = float(
                silhouette_score(
                    features,
                    labels,
                    sample_size=sil_sample,
                    random_state=random_seed,
                )
            )
        except Exception:
            pass
        try:
            metrics["dbi"] = float(davies_bouldin_score(features, labels))
        except Exception:
            pass
        try:
            metrics["chi"] = float(calinski_harabasz_score(features, labels))
        except Exception:
            pass

    metrics["knn_consistency"] = compute_knn_consistency(features, labels, knn_k)
    return metrics


def parse_step_indices(step_indices: str, default_step: int) -> List[int]:
    if not step_indices.strip():
        return [default_step]
    out = []
    for part in step_indices.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        return [default_step]
    return out


def build_sigma_values(
    sigma_min: float,
    sigma_max: float,
    num_steps: int,
    rho: float,
    step_indices: List[int],
) -> List[float]:
    schedule = generate_sigma_schedule(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        rho=rho,
    )
    values = []
    for idx in step_indices:
        if idx < 0 or idx >= len(schedule):
            raise ValueError(f"step index {idx} out of range [0, {len(schedule) - 1}]")
        values.append(float(schedule[idx].item()))
    return values


def extract_diffusion_features_batched(
    net,
    images: np.ndarray,
    sigma_values: List[float],
    step_indices: List[int],
    batch_size: int,
    device: str,
    desc: str,
) -> np.ndarray:
    all_feats = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), total=total_batches, desc=desc):
        batch_images = images[i : i + batch_size]
        batch_feat = extract_features_with_noise(
            net=net,
            images=batch_images,
            sigma=sigma_values,
            step_idx=step_indices,
            device=device,
        )
        all_feats.append(batch_feat)
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def extract_acgan_features_batched(
    acgan_model,
    images: np.ndarray,
    layer: str,
    batch_size: int,
    device: str,
    desc: str,
) -> np.ndarray:
    all_feats = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), total=total_batches, desc=desc):
        batch_images = images[i : i + batch_size]
        batch_feat = extract_features_with_acgan(
            acgan_model=acgan_model,
            images=batch_images,
            layer=layer,
            device=device,
        )
        all_feats.append(batch_feat)
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def image_to_pixel_features(images: np.ndarray) -> np.ndarray:
    x = images.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return x.reshape(x.shape[0], -1).astype(np.float32)


def load_classifier_features(
    classifier_features_npz: str,
    expected_num_samples: int,
    fallback_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pack = np.load(classifier_features_npz)
    if "features" not in pack:
        raise KeyError(f"{classifier_features_npz} must contain key 'features'")
    features = pack["features"].astype(np.float32)
    if features.shape[0] != expected_num_samples:
        raise ValueError(
            f"classifier features sample count mismatch: {features.shape[0]} vs expected {expected_num_samples}"
        )

    if "labels" in pack:
        labels = pack["labels"].astype(np.int64)
        if labels.shape[0] != expected_num_samples:
            raise ValueError(
                f"classifier labels sample count mismatch: {labels.shape[0]} vs expected {expected_num_samples}"
            )
    else:
        labels = fallback_labels.astype(np.int64)
    return features, labels


def format_metric(value: float, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def metric_line_for_print(space_name: str, metric_values: Optional[Dict[str, float]]) -> str:
    if metric_values is None:
        return f"{space_name:<20} | Silhouette=--, DBI=--, CHI=--, kNN=--"
    return (
        f"{space_name:<20} | "
        f"Silhouette={format_metric(metric_values['silhouette'], 3)}, "
        f"DBI={format_metric(metric_values['dbi'], 3)}, "
        f"CHI={format_metric(metric_values['chi'], 2)}, "
        f"kNN={format_metric(metric_values['knn_consistency'], 3)}"
    )


def generate_latex_table(metrics_by_space: Dict[str, Optional[Dict[str, float]]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Quantitative analysis of representation quality across different feature spaces for semantic cluster alignment. Higher is better for Silhouette, CHI, and kNN Consistency, while lower is better for DBI.}",
        r"\label{tab:representation_quality}",
        r"\resizebox{0.95\linewidth}{!}{",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Feature Space} & \textbf{Silhouette $\uparrow$} & \textbf{DBI $\downarrow$} & \textbf{CHI $\uparrow$} & \textbf{kNN Consistency $\uparrow$} \\",
        r"\midrule",
    ]

    for display_name, key in TABLE_ROWS:
        m = metrics_by_space.get(key)
        if m is None:
            row = f"{display_name:<18} & -- & -- & -- & -- " + r"\\"
        else:
            row = (
                f"{display_name:<18} & "
                f"{format_metric(m['silhouette'], 3)} & "
                f"{format_metric(m['dbi'], 3)} & "
                f"{format_metric(m['chi'], 2)} & "
                f"{format_metric(m['knn_consistency'], 3)} "
                + r"\\"
            )
        lines.append(row)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


@click.command()
@click.option("--data", type=str, required=True, help="Dataset path, same format as pseudo_label.py")
@click.option(
    "--dataset_type",
    type=click.Choice(["cifar10", "cifar100", "fashion-mnist", "mnist", "svhn", "stl10"]),
    default="cifar10",
    show_default=True,
)
@click.option("--network_pkl", type=str, default="", show_default=False, help="Required for diffusion spaces")
@click.option("--acgan_ckpt", type=str, default="", show_default=False, help="Optional, enables GAN spectral row")
@click.option("--acgan_layer", type=str, default="flatten", show_default=True)
@click.option(
    "--classifier_features_npz",
    type=str,
    default="",
    show_default=False,
    help="Optional .npz containing classifier features with key 'features' (optional key: 'labels').",
)
@click.option("--outdir", type=str, default="representation_quality_output", show_default=True)
@click.option("--max_images", type=int, default=10000, show_default=True)
@click.option("--label_ratio", type=float, default=0.05, show_default=True)
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--random_seed", type=int, default=42, show_default=True)
@click.option("--step_indices", type=str, default="77", show_default=True)
@click.option("--sigma_min", type=float, default=0.002, show_default=True)
@click.option("--sigma_max", type=float, default=80.0, show_default=True)
@click.option("--num_steps", type=int, default=100, show_default=True)
@click.option("--rho", type=float, default=7.0, show_default=True)
@click.option("--spectral_k", type=int, default=15, show_default=True)
@click.option("--spectral_tau", type=float, default=0.0, show_default=True)
@click.option("--spectral_dim", type=int, default=16, show_default=True)
@click.option("--spectral_keep_first_eigvec", is_flag=True, default=False)
@click.option("--knn_k", type=int, default=10, show_default=True)
@click.option("--silhouette_max_samples", type=int, default=3000, show_default=True)
@click.option("--l2_normalize_features", is_flag=True, default=False)
def main(
    data: str,
    dataset_type: str,
    network_pkl: str,
    acgan_ckpt: str,
    acgan_layer: str,
    classifier_features_npz: str,
    outdir: str,
    max_images: int,
    label_ratio: float,
    batch_size: int,
    device: str,
    random_seed: int,
    step_indices: str,
    sigma_min: float,
    sigma_max: float,
    num_steps: int,
    rho: float,
    spectral_k: int,
    spectral_tau: float,
    spectral_dim: int,
    spectral_keep_first_eigvec: bool,
    knn_k: int,
    silhouette_max_samples: int,
    l2_normalize_features: bool,
):
    os.makedirs(outdir, exist_ok=True)
    set_seed(random_seed)

    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{device}' but CUDA is unavailable. Fallback to cpu.")
        device = "cpu"

    print("=== 1) Load dataset ===")
    images, labels = load_dataset_data(data, dataset_type, max_images=max_images)
    labels = labels.astype(np.int64)
    labeled_images, labeled_labels, unlabeled_images, unlabeled_labels = split_labeled_unlabeled(
        images, labels, label_ratio
    )
    print(f"labeled: {labeled_images.shape}, unlabeled: {unlabeled_images.shape}")

    eval_images = unlabeled_images
    eval_labels = unlabeled_labels.astype(np.int64)
    print(f"evaluation split: unlabeled, samples={len(eval_labels)}")

    metrics_by_space: Dict[str, Optional[Dict[str, float]]] = {
        "classifier_feature": None,
        "pixel_space": None,
        "diffusion_latent": None,
        "gan_spectral": None,
        "diffusion_spectral": None,
    }

    print("=== 2) Pixel space metrics ===")
    pixel_feat = image_to_pixel_features(eval_images)
    if l2_normalize_features:
        pixel_feat = l2_normalize(pixel_feat)
    metrics_by_space["pixel_space"] = compute_representation_metrics(
        features=pixel_feat,
        labels=eval_labels,
        knn_k=knn_k,
        silhouette_max_samples=silhouette_max_samples,
        random_seed=random_seed,
    )

    if classifier_features_npz:
        print("=== 3) Classifier feature metrics ===")
        classifier_feat, classifier_labels = load_classifier_features(
            classifier_features_npz=classifier_features_npz,
            expected_num_samples=len(eval_labels),
            fallback_labels=eval_labels,
        )
        if l2_normalize_features:
            classifier_feat = l2_normalize(classifier_feat)
        metrics_by_space["classifier_feature"] = compute_representation_metrics(
            features=classifier_feat,
            labels=classifier_labels,
            knn_k=knn_k,
            silhouette_max_samples=silhouette_max_samples,
            random_seed=random_seed,
        )
    else:
        print("Classifier features not provided, classifier row will be '--'.")

    need_diffusion = True
    if need_diffusion and not network_pkl:
        raise click.ClickException("--network_pkl is required for diffusion latent/spectral rows.")

    print("=== 4) Diffusion latent + diffusion spectral metrics ===")
    net = load_network(network_pkl, device=device)
    step_idx_list = parse_step_indices(step_indices, default_step=77)
    sigma_values = build_sigma_values(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        rho=rho,
        step_indices=step_idx_list,
    )
    print(f"use step indices: {step_idx_list}")
    print(f"use sigma values: {[round(v, 6) for v in sigma_values]}")

    diffusion_labeled_feat = extract_diffusion_features_batched(
        net=net,
        images=labeled_images,
        sigma_values=sigma_values,
        step_indices=step_idx_list,
        batch_size=batch_size,
        device=device,
        desc="Diffusion labeled",
    )
    diffusion_eval_feat = extract_diffusion_features_batched(
        net=net,
        images=eval_images,
        sigma_values=sigma_values,
        step_indices=step_idx_list,
        batch_size=batch_size,
        device=device,
        desc="Diffusion eval",
    )
    diffusion_for_metric = diffusion_eval_feat
    if l2_normalize_features:
        diffusion_for_metric = l2_normalize(diffusion_for_metric)
    metrics_by_space["diffusion_latent"] = compute_representation_metrics(
        features=diffusion_for_metric,
        labels=eval_labels,
        knn_k=knn_k,
        silhouette_max_samples=silhouette_max_samples,
        random_seed=random_seed,
    )

    _, diffusion_eval_spec, diffusion_graph_meta = build_spectral_features_joint(
        labeled_features=diffusion_labeled_feat,
        unlabeled_features=diffusion_eval_feat,
        knn_k=spectral_k,
        tau=spectral_tau,
        spectral_dim=spectral_dim,
        drop_first_eigvec=(not spectral_keep_first_eigvec),
    )
    diffusion_spec_for_metric = diffusion_eval_spec
    if l2_normalize_features:
        diffusion_spec_for_metric = l2_normalize(diffusion_spec_for_metric)
    metrics_by_space["diffusion_spectral"] = compute_representation_metrics(
        features=diffusion_spec_for_metric,
        labels=eval_labels,
        knn_k=knn_k,
        silhouette_max_samples=silhouette_max_samples,
        random_seed=random_seed,
    )

    gan_graph_meta = None
    if acgan_ckpt:
        print("=== 5) GAN spectral metrics ===")
        num_classes = 10 if dataset_type in ["cifar10", "fashion-mnist", "mnist", "svhn", "stl10"] else 100
        img_size = int(images.shape[2]) if images.ndim == 4 else 32
        img_channels = 1 if dataset_type in ["mnist", "fashion-mnist"] else 3
        acgan_model = load_acgan_model(
            acgan_ckpt_path=acgan_ckpt,
            device=device,
            img_channels=img_channels,
            num_classes=num_classes,
            img_size=img_size,
            latent_dim=None,
        )
        gan_labeled_feat = extract_acgan_features_batched(
            acgan_model=acgan_model,
            images=labeled_images,
            layer=acgan_layer,
            batch_size=batch_size,
            device=device,
            desc="ACGAN labeled",
        )
        gan_eval_feat = extract_acgan_features_batched(
            acgan_model=acgan_model,
            images=eval_images,
            layer=acgan_layer,
            batch_size=batch_size,
            device=device,
            desc="ACGAN eval",
        )
        _, gan_eval_spec, gan_graph_meta = build_spectral_features_joint(
            labeled_features=gan_labeled_feat,
            unlabeled_features=gan_eval_feat,
            knn_k=spectral_k,
            tau=spectral_tau,
            spectral_dim=spectral_dim,
            drop_first_eigvec=(not spectral_keep_first_eigvec),
        )
        gan_spec_for_metric = gan_eval_spec
        if l2_normalize_features:
            gan_spec_for_metric = l2_normalize(gan_spec_for_metric)
        metrics_by_space["gan_spectral"] = compute_representation_metrics(
            features=gan_spec_for_metric,
            labels=eval_labels,
            knn_k=knn_k,
            silhouette_max_samples=silhouette_max_samples,
            random_seed=random_seed,
        )
    else:
        print("ACGAN checkpoint not provided, GAN spectral row will be '--'.")

    print("\n=== Summary ===")
    for display_name, key in TABLE_ROWS:
        print(metric_line_for_print(display_name, metrics_by_space.get(key)))

    payload = {
        "settings": {
            "data": data,
            "dataset_type": dataset_type,
            "max_images": int(max_images),
            "label_ratio": float(label_ratio),
            "batch_size": int(batch_size),
            "device": device,
            "random_seed": int(random_seed),
            "step_indices": step_idx_list,
            "sigma_values": sigma_values,
            "spectral_k": int(spectral_k),
            "spectral_tau": float(spectral_tau),
            "spectral_dim": int(spectral_dim),
            "spectral_keep_first_eigvec": bool(spectral_keep_first_eigvec),
            "knn_k": int(knn_k),
            "silhouette_max_samples": int(silhouette_max_samples),
            "l2_normalize_features": bool(l2_normalize_features),
        },
        "graph_meta": {
            "diffusion_spectral": diffusion_graph_meta,
            "gan_spectral": gan_graph_meta,
        },
        "metrics": metrics_by_space,
    }

    json_path = os.path.join(outdir, "representation_quality_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"json saved: {json_path}")

    csv_path = os.path.join(outdir, "representation_quality_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("feature_space,silhouette,dbi,chi,knn_consistency\n")
        for display_name, key in TABLE_ROWS:
            m = metrics_by_space.get(key)
            if m is None:
                f.write(f"{display_name},,,,\n")
            else:
                f.write(
                    f"{display_name},{m['silhouette']},{m['dbi']},{m['chi']},{m['knn_consistency']}\n"
                )
    print(f"csv saved: {csv_path}")

    latex = generate_latex_table(metrics_by_space)
    tex_path = os.path.join(outdir, "representation_quality_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"latex saved: {tex_path}")


if __name__ == "__main__":
    main()

