import json
import os
from typing import Dict, Tuple

import click
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from pseudo_label import (
    extract_features_with_noise,
    generate_sigma_schedule,
    load_dataset_data,
    load_network,
)


def parse_sigma_values(
    sigma_values: str,
    sigma_min: float,
    sigma_max: float,
    num_steps: int,
    rho: float,
) -> np.ndarray:
    if sigma_values and sigma_values.strip():
        values = [float(x.strip()) for x in sigma_values.split(",") if x.strip()]
        if not values:
            raise ValueError("--sigma_values 解析后为空")
        return np.array(values, dtype=np.float32)

    schedule = generate_sigma_schedule(
        sigma_min=sigma_min, sigma_max=sigma_max, num_steps=num_steps, rho=rho
    ).cpu().numpy()
    return schedule.astype(np.float32)


def extract_features_in_batches(
    net,
    images: np.ndarray,
    sigma_array: np.ndarray,
    batch_size: int,
    step_idx: int,
    device: str,
) -> np.ndarray:
    feat_list = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), total=total_batches, desc="Extracting features"):
        batch_images = images[i : i + batch_size]
        batch_feat = extract_features_with_noise(
            net=net,
            images=batch_images,
            sigma=sigma_array,
            step_idx=step_idx,
            device=device,
        )
        feat_list.append(batch_feat)
    return np.concatenate(feat_list, axis=0).astype(np.float32)


def compute_knn_purity(features: np.ndarray, labels: np.ndarray, k: int) -> float:
    n_samples = features.shape[0]
    if n_samples <= 1:
        return float("nan")
    k_eff = max(1, min(k, n_samples - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(features)
    indices = nn.kneighbors(return_distance=False)[:, 1:]
    nbr_labels = labels[indices]
    same = (nbr_labels == labels[:, None]).mean(axis=1)
    return float(np.mean(same))


def compute_trustworthiness(
    images: np.ndarray,
    features: np.ndarray,
    n_neighbors: int,
    max_samples: int,
    random_seed: int,
) -> float:
    n = len(images)
    if n <= 2:
        return float("nan")
    n_use = min(n, max_samples)
    if n_use < n:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n, size=n_use, replace=False)
        x = images[idx]
        z = features[idx]
    else:
        x = images
        z = features

    x_flat = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
    k_eff = max(1, min(n_neighbors, x_flat.shape[0] - 1))
    return float(trustworthiness(x_flat, z, n_neighbors=k_eff))


def compute_cluster_metrics(
    features: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int,
    random_seed: int,
    silhouette_max_samples: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    kmeans = KMeans(n_clusters=num_classes, random_state=random_seed, n_init=10)
    pred_clusters = kmeans.fit_predict(features)

    sil_sample = None
    if silhouette_max_samples > 0 and features.shape[0] > silhouette_max_samples:
        sil_sample = silhouette_max_samples

    metrics = {
        "silhouette": float(
            silhouette_score(
                features,
                pred_clusters,
                sample_size=sil_sample,
                random_state=random_seed,
            )
        ),
        "dbi": float(davies_bouldin_score(features, pred_clusters)),
        "ch": float(calinski_harabasz_score(features, pred_clusters)),
        "ari": float(adjusted_rand_score(true_labels, pred_clusters)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_clusters)),
    }
    return pred_clusters, metrics


@click.command()
@click.option("--data", type=str, required=True, help="数据集路径（与 pseudo_label.py 一致）")
@click.option("--dataset_type", type=click.Choice(["cifar10", "cifar100", "fashion-mnist", "mnist", "svhn", "stl10"]),
              default="cifar10", show_default=True)
@click.option("--network_pkl", type=str, required=True, help="EDM 模型 pkl 路径")
@click.option("--max_images", type=int, default=10000, show_default=True)
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--num_steps", type=int, default=18, show_default=True, help="sigma 调度步数")
@click.option("--sigma_min", type=float, default=0.002, show_default=True)
@click.option("--sigma_max", type=float, default=80.0, show_default=True)
@click.option("--rho", type=float, default=7.0, show_default=True)
@click.option("--sigma_values", type=str, default="", show_default=False,
              help="手动指定 sigma 列表，例如: 0.5,1.0,2.0；为空时使用调度序列")
@click.option("--step_idx", type=int, default=0, show_default=True, help="兼容 extract_features_with_noise 接口")
@click.option("--knn_k", type=int, default=10, show_default=True)
@click.option("--trust_n_neighbors", type=int, default=10, show_default=True)
@click.option("--trust_max_samples", type=int, default=3000, show_default=True)
@click.option("--silhouette_max_samples", type=int, default=3000, show_default=True)
@click.option("--random_seed", type=int, default=42, show_default=True)
@click.option("--save_json", type=str, default="", show_default=False, help="可选：保存指标到 json 文件")
def main(
    data,
    dataset_type,
    network_pkl,
    max_images,
    batch_size,
    device,
    num_steps,
    sigma_min,
    sigma_max,
    rho,
    sigma_values,
    step_idx,
    knn_k,
    trust_n_neighbors,
    trust_max_samples,
    silhouette_max_samples,
    random_seed,
    save_json,
):
    np.random.seed(random_seed)

    print("=== 1) Load model ===")
    net = load_network(network_pkl, device=device)

    print("=== 2) Load dataset ===")
    images, labels = load_dataset_data(data, dataset_type, max_images=max_images)
    labels = labels.astype(np.int64)
    num_classes = int(len(np.unique(labels)))
    print(f"images: {images.shape}, labels: {labels.shape}, num_classes: {num_classes}")

    print("=== 3) Prepare sigma list ===")
    sigma_array = parse_sigma_values(
        sigma_values=sigma_values,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        rho=rho,
    )
    print(f"sigma count: {len(sigma_array)}, first3: {sigma_array[:3]}, last3: {sigma_array[-3:]}")

    print("=== 4) Extract diffusion intermediate features ===")
    features = extract_features_in_batches(
        net=net,
        images=images,
        sigma_array=sigma_array,
        batch_size=batch_size,
        step_idx=step_idx,
        device=device,
    )
    print(f"features: {features.shape}")

    print("=== 5) Compute clustering and alignment metrics ===")
    pred_clusters, metric_values = compute_cluster_metrics(
        features=features,
        true_labels=labels,
        num_classes=num_classes,
        random_seed=random_seed,
        silhouette_max_samples=silhouette_max_samples,
    )

    metric_values["knn_purity"] = compute_knn_purity(features, labels, k=knn_k)
    metric_values["trustworthiness"] = compute_trustworthiness(
        images=images,
        features=features,
        n_neighbors=trust_n_neighbors,
        max_samples=trust_max_samples,
        random_seed=random_seed,
    )

    # 方便后续扩展或调试
    metric_values["num_samples"] = int(len(labels))
    metric_values["num_classes"] = int(num_classes)
    metric_values["kmeans_unique_clusters"] = int(len(np.unique(pred_clusters)))

    print("\n=== Metrics ===")
    print(f"Silhouette (↑):      {metric_values['silhouette']:.6f}")
    print(f"DBI (↓):             {metric_values['dbi']:.6f}")
    print(f"CH (↑):              {metric_values['ch']:.6f}")
    print(f"kNN purity (↑):      {metric_values['knn_purity']:.6f}")
    print(f"ARI (↑):             {metric_values['ari']:.6f}")
    print(f"NMI (↑):             {metric_values['nmi']:.6f}")
    print(f"Trustworthiness (↑): {metric_values['trustworthiness']:.6f}")

    if save_json:
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(metric_values, f, ensure_ascii=False, indent=2)
        print(f"\n指标已保存到: {save_json}")


if __name__ == "__main__":
    main()
