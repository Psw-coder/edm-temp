import os
import pickle
import tarfile
import numpy as np
import torch
import click
import dnnlib
from torch_utils import misc
from training.networks import EDMPrecond
import training.networks as edm_networks
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


# ----------------------------------------------------------------------------
# 网络加载函数

def load_network(network_pkl_path, device='cuda'):
    """加载预训练的扩散模型网络"""
    print(f'Loading network from "{network_pkl_path}"...')
    with dnnlib.util.open_url(network_pkl_path) as f:
        loaded_data = pickle.load(f)
        old_net = loaded_data['ema']

    # 重新创建网络
    net = EDMPrecond(**old_net.init_kwargs)
    net.sigma_max = 80.0
    misc.copy_params_and_buffers(src_module=old_net, dst_module=net, require_all=True)
    net = net.to(device)
    net.eval()

    print(f"Network loaded successfully!")
    return net


# ----------------------------------------------------------------------------
# EDM 采样器 (用于验证模型)

def edm_sampler(
        net, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDM采样器，用于生成图像验证模型是否正常工作"""
    # 调整噪声水平
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # 时间步离散化
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # 主采样循环
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # 临时增加噪声
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler步骤
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 应用二阶修正
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def test_model_sampling(net, device='cuda', num_samples=4, num_steps=18, save_path=None):
    """
    测试模型是否能正常采样，验证模型参数是否正确加载

    Args:
        net: 加载的扩散模型网络
        device: 设备
        num_samples: 采样数量
        num_steps: 采样步数
        save_path: 保存路径

    Returns:
        samples: 生成的样本 (num_samples, 3, 32, 32)
    """
    print(f"开始采样测试，生成 {num_samples} 个样本...")

    # 生成噪声时间步
    sigma_min = 0.002
    sigma_max = 80
    rho = 7.0

    print(f"噪声范围: [{sigma_min:.6f}, {sigma_max:.6f}]")
    print(f"采样步数: {num_steps}")

    # 初始化随机噪声
    latents = torch.randn([num_samples, net.img_channels, net.img_resolution, net.img_resolution], device=device)

    print(f"初始噪声形状: {latents.shape}")
    print(f"初始噪声范围: [{latents.min().item():.3f}, {latents.max().item():.3f}]")

    # 使用EDM采样器进行采样
    with torch.no_grad():
        samples = edm_sampler(
            net=net,
            latents=latents,
            class_labels=None,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho
        )

    # 后处理到 [0, 255] 范围
    samples = (samples * 127.5 + 128).clip(0, 255).to(torch.uint8)

    print(f"采样完成！最终样本形状: {samples.shape}")
    print(f"最终样本范围: [{samples.min().item()}, {samples.max().item()}]")

    # 保存样本（如果指定了路径）
    if save_path:
        save_samples_as_grid(samples.cpu().numpy(), save_path)

    return samples.cpu().numpy()


def save_samples_as_grid(samples, save_path):
    """
    将采样结果保存为网格图像

    Args:
        samples: 采样结果 (num_samples, 3, 32, 32)
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt

        num_samples = samples.shape[0]
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for i in range(num_samples):
            row = i // cols
            col = i % cols

            # 转换为 HWC 格式用于显示
            img = samples[i].transpose(1, 2, 0)  # CHW -> HWC

            axes[row][col].imshow(img)
            axes[row][col].set_title(f'Sample {i + 1}')
            axes[row][col].axis('off')

        # 隐藏多余的子图
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row][col].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"采样结果已保存到: {save_path}")

    except ImportError:
        print("警告: matplotlib未安装，无法保存可视化结果")
        # 简单保存为numpy文件
        np.save(save_path.replace('.png', '.npy'), samples)
        print(f"采样结果已保存为numpy文件: {save_path.replace('.png', '.npy')}")


def check_model_parameters(net):
    """
    检查模型参数的基本统计信息

    Args:
        net: 模型网络
    """
    print("\n=== 模型参数检查 ===")

    total_params = 0
    param_stats = []

    for name, param in net.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params

            param_mean = param.data.mean().item()
            param_std = param.data.std().item()
            param_min = param.data.min().item()
            param_max = param.data.max().item()

            param_stats.append({
                'name': name,
                'shape': list(param.shape),
                'num_params': num_params,
                'mean': param_mean,
                'std': param_std,
                'min': param_min,
                'max': param_max
            })

    print(f"总参数数量: {total_params:,}")
    print(f"参数层数: {len(param_stats)}")

    # 显示前几层和后几层的统计信息
    print("\n前5层参数统计:")
    for stat in param_stats[:5]:
        print(f"  {stat['name']}: {stat['shape']}, 均值={stat['mean']:.6f}, 标准差={stat['std']:.6f}")

    print("\n后5层参数统计:")
    for stat in param_stats[-5:]:
        print(f"  {stat['name']}: {stat['shape']}, 均值={stat['mean']:.6f}, 标准差={stat['std']:.6f}")

    # 检查是否有异常值
    zero_params = [s for s in param_stats if abs(s['mean']) < 1e-8 and s['std'] < 1e-8]
    if zero_params:
        print(f"\n警告: 发现 {len(zero_params)} 层参数可能未正确初始化（接近零）:")
        for stat in zero_params[:3]:  # 只显示前3个
            print(f"  {stat['name']}: 均值={stat['mean']:.8f}, 标准差={stat['std']:.8f}")


def compute_model_flops_and_params(model, forward_args=None, forward_kwargs=None):
    total_flops = 0
    handles = []

    def conv2d_hook(module, inputs, output):
        nonlocal total_flops
        w = getattr(module, 'weight', None)
        if w is None:
            return
        x = inputs[0]
        n = output.shape[0]
        cout = output.shape[1]
        hout = output.shape[2]
        wout = output.shape[3]
        cin = w.shape[1]
        kh = w.shape[2]
        kw = w.shape[3]
        total_flops += int(n * cout * hout * wout * cin * kh * kw * 2)

    def linear_hook(module, inputs, output):
        nonlocal total_flops
        w = getattr(module, 'weight', None)
        if w is None:
            return
        x = inputs[0]
        n = x.shape[0] if x.dim() > 1 else 1
        in_features = w.shape[1]
        out_features = w.shape[0]
        total_flops += int(n * in_features * out_features * 2)

    def unet_block_attention_hook(module, inputs, output):
        nonlocal total_flops
        if getattr(module, 'num_heads', 0):
            n = output.shape[0]
            h = output.shape[2]
            w = output.shape[3]
            q = h * w
            num_heads = module.num_heads
            c_per_head = module.out_channels // num_heads
            n_heads_total = n * num_heads
            total_flops += int(4 * n_heads_total * c_per_head * q * q)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, edm_networks.Conv2d):
            handles.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, torch.nn.Linear) or isinstance(m, edm_networks.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, edm_networks.UNetBlock):
            handles.append(m.register_forward_hook(unet_block_attention_hook))

    with torch.no_grad():
        args = () if forward_args is None else forward_args
        kwargs = {} if forward_kwargs is None else forward_kwargs
        model(*args, **kwargs)

    for h in handles:
        h.remove()

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    params_mb = params_bytes / (1024 ** 2)
    return total_flops, params_mb


# ----------------------------------------------------------------------------
# 特征提取函数

def extract_features_with_noise(net, images, sigma, step_idx, device='cuda'):
    """
    对图像添加噪声并提取UNet中间特征，支持多个sigma值的特征平均

    Args:
        net: 预训练的扩散模型网络
        images: 图像数据 (numpy array, 0-255)
        sigma: 噪声水平，可以是单个值或多个值的列表/数组
        step_idx: 噪声步数索引
        device: 设备

    Returns:
        features: 拼接后的特征 (batch_size, feature_dim)
    """
    # 预处理图像
    images_tensor = preprocess_image(images).to(device)

    # 确保sigma是列表格式
    if isinstance(sigma, (int, float)):
        sigma_list = [sigma]
    else:
        sigma_list = list(sigma) if hasattr(sigma, '__iter__') else [sigma]

    # 存储所有sigma下的特征
    all_features = []

    # 对每个sigma值提取特征
    for sigma_val in sigma_list:
        # 添加噪声
        noisy_images = add_noise_edm(images_tensor, sigma_val, device)

        # 提取特征
        with torch.no_grad():
            sigma_tensor = torch.full((images_tensor.shape[0],), sigma_val, device=device, dtype=torch.float32)
            _, features = net(noisy_images, sigma_tensor, None, return_features=True)

        # 提取dec_7到dec_12的特征并按channels维度拼接
        # target_keys = [f'dec_{i}' for i in range(3, 7)]
        target_keys = [f'dec_{i}' for i in range(5, 7)]
        feature_list = []

        for key in target_keys:
            if key in features:
                feature_list.append(features[key])

        if len(feature_list) > 0:
            # 按channels维度拼接特征 (batch_size, total_channels, H, W)
            concatenated_feat = torch.cat(feature_list, dim=1)
            batch_size = concatenated_feat.shape[0]
            # 展平为 (batch_size, -1)
            flattened_feat = concatenated_feat.view(batch_size, -1)
            all_features.append(flattened_feat)

    # 对所有sigma的特征求平均
    if len(all_features) > 0:
        stacked_features = torch.stack(all_features, dim=0)
        averaged_features = torch.mean(stacked_features, dim=0)
        return averaged_features.cpu().numpy()
    else:
        raise ValueError("没有提取到有效的特征")


def apply_local_dp_to_features(features, epsilon, delta, l2_clip=1.0, rng=None):
    """
    对中间层特征进行本地差分隐私处理（L2 归一化 + 高斯噪声）。

    数学形式：
        1) 归一化（L2-Normalization，将 L2 敏感度限制为 1）:
            f_hat = f / max(1, ||f||_2)
        2) 高斯噪声注入:
            f_tilde = f_hat + N(0, σ^2 I)
           其中 σ = sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        features: 特征矩阵，形状为 (N, D)，可以是 numpy.ndarray 或 torch.Tensor
        epsilon: 隐私预算 ε (> 0)
        delta:   失败概率 δ (0 < delta < 1)
        l2_clip: L2 范数截断阈值，默认 1.0
        rng:     可选的 numpy.random.Generator，用于控制随机性

    Returns:
        与输入同类型的差分隐私特征矩阵，形状为 (N, D)
    """
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数")
    if not (0 < delta < 1):
        raise ValueError("delta 必须在 (0, 1) 区间内")

    is_torch = isinstance(features, torch.Tensor)
    device = None

    if is_torch:
        device = features.device
        x = features.detach().cpu().numpy()
    else:
        x = np.asarray(features, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"features 形状应为 (N, D)，当前为 {x.shape}")

    # 1) L2 归一化 + 截断：f_hat = f / max(1, ||f||_2 / l2_clip)
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    scale = np.maximum(1.0, norms / float(l2_clip))
    x_hat = x / scale

    # 2) 计算高斯噪声标准差 σ
    sigma = np.sqrt(2.0 * np.log(1.25 / float(delta))) / float(epsilon)

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(loc=0.0, scale=sigma, size=x_hat.shape).astype(x_hat.dtype)
    x_tilde = x_hat + noise

    if is_torch:
        return torch.from_numpy(x_tilde).to(device)
    return x_tilde


def save_features_for_tsne(features, labels, save_path):
    """
    保存特征和标签用于t-SNE可视化

    Args:
        features: 特征矩阵 (N, D)
        labels: 对应的标签 (N,)
        save_path: 保存路径 (.npz)
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存为压缩的numpy格式
        np.savez_compressed(save_path, features=features, labels=labels)
        print(f"✅ 特征和标签已保存到: {save_path}")
        print(f"   特征维度: {features.shape}, 标签维度: {labels.shape}")
    except Exception as e:
        print(f"❌ 保存特征失败: {e}")


# ----------------------------------------------------------------------------
# 聚类和伪标签生成函数

def visualize_clustering_quality(labeled_features, labeled_labels, cluster_assignments, cluster_to_label, save_path):
    """
    可视化聚类质量，分析有标签数据在聚类中的分布

    Args:
        labeled_features: 有标签数据的特征
        labeled_labels: 有标签数据的真实标签
        cluster_assignments: 聚类分配结果
        cluster_to_label: 聚类到标签的映射
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('聚类质量分析', fontsize=16, fontweight='bold')

        # 1. 聚类-标签混淆矩阵
        confusion_matrix = np.zeros((10, 10))  # 10个聚类 x 10个标签
        for cluster_id in range(10):
            mask = cluster_assignments == cluster_id
            if np.sum(mask) > 0:
                cluster_labels = labeled_labels[mask]
                for label in range(10):
                    confusion_matrix[cluster_id, label] = np.sum(cluster_labels == label)

        sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues',
                    ax=axes[0, 0], cbar_kws={'label': '样本数量'})
        axes[0, 0].set_title('聚类-标签混淆矩阵')
        axes[0, 0].set_xlabel('真实标签')
        axes[0, 0].set_ylabel('聚类ID')

        # 2. 每个聚类的纯度分析
        cluster_purities = []
        cluster_sizes = []
        for cluster_id in range(10):
            mask = cluster_assignments == cluster_id
            if np.sum(mask) > 0:
                cluster_labels = labeled_labels[mask]
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                purity = np.max(counts) / np.sum(counts)  # 最多标签占比
                cluster_purities.append(purity)
                cluster_sizes.append(np.sum(counts))
            else:
                cluster_purities.append(0)
                cluster_sizes.append(0)

        bars = axes[0, 1].bar(range(10), cluster_purities, color='skyblue', alpha=0.7)
        axes[0, 1].set_title('各聚类的纯度 (同一标签占比)')
        axes[0, 1].set_xlabel('聚类ID')
        axes[0, 1].set_ylabel('纯度')
        axes[0, 1].set_ylim(0, 1)

        # 在柱状图上添加数值标签
        for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.2f}\n({size})', ha='center', va='bottom', fontsize=8)

        # 3. PCA降维可视化
        if labeled_features.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            features_2d = pca.fit_transform(labeled_features)
            explained_var = pca.explained_variance_ratio_
        else:
            features_2d = labeled_features
            explained_var = [1.0, 0.0]

        # 按真实标签着色
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for label in range(10):
            mask = labeled_labels == label
            if np.sum(mask) > 0:
                axes[1, 0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                                   c=[colors[label]], label=f'标签 {label}', alpha=0.6, s=20)

        axes[1, 0].set_title(
            f'PCA降维可视化 (按真实标签着色)\n解释方差比: {explained_var[0]:.3f}, {explained_var[1]:.3f}')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 4. 按聚类结果着色
        for cluster_id in range(10):
            mask = cluster_assignments == cluster_id
            if np.sum(mask) > 0:
                assigned_label = cluster_to_label.get(cluster_id, cluster_id)
                axes[1, 1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                                   c=[colors[cluster_id]], label=f'簇{cluster_id}→标签{assigned_label}',
                                   alpha=0.6, s=20)

        axes[1, 1].set_title('PCA降维可视化 (按聚类结果着色)')
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 打印统计信息
        print(f"\n=== 聚类质量分析 ===")
        print(f"平均聚类纯度: {np.mean(cluster_purities):.4f}")
        print(f"聚类纯度标准差: {np.std(cluster_purities):.4f}")

        # 计算每个标签的聚类分散度
        print(f"\n各标签的聚类分散情况:")
        for label in range(10):
            mask = labeled_labels == label
            if np.sum(mask) > 0:
                label_clusters = cluster_assignments[mask]
                unique_clusters = np.unique(label_clusters)
                print(f"  标签 {label}: 分散在 {len(unique_clusters)} 个聚类中 {unique_clusters}")

        print(f"聚类可视化已保存到: {save_path}")

    except ImportError as e:
        print(f"警告: 缺少可视化依赖 ({e})，跳过聚类可视化")
    except Exception as e:
        print(f"聚类可视化出错: {e}")


def gpu_kmeans_clustering(data_tensor, n_clusters, device, max_iters=100, tol=1e-4):
    """
    GPU加速的K-means聚类实现

    Args:
        data_tensor: 输入数据张量 [N, D]
        n_clusters: 聚类数量
        device: 计算设备
        max_iters: 最大迭代次数
        tol: 收敛容忍度

    Returns:
        cluster_centers: 聚类中心 [K, D]
        cluster_assignments: 聚类分配 [N]
    """
    print(f"开始GPU加速K-means聚类，数据形状: {data_tensor.shape}, 聚类数: {n_clusters}")

    N, D = data_tensor.shape

    # 初始化聚类中心（使用K-means++初始化）
    cluster_centers = kmeans_plus_plus_init(data_tensor, n_clusters, device)

    prev_centers = cluster_centers.clone()

    for iteration in range(max_iters):
        # 计算每个点到所有聚类中心的距离
        distances = torch.cdist(data_tensor, cluster_centers, p=2)  # [N, K]

        # 分配每个点到最近的聚类中心
        cluster_assignments = torch.argmin(distances, dim=1)  # [N]

        # 更新聚类中心
        new_centers = torch.zeros_like(cluster_centers)
        for k in range(n_clusters):
            mask = cluster_assignments == k
            if mask.sum() > 0:
                new_centers[k] = data_tensor[mask].mean(dim=0)
            else:
                # 如果某个聚类为空，重新随机初始化
                new_centers[k] = data_tensor[torch.randint(0, N, (1,))].squeeze()

        # 检查收敛
        center_shift = torch.norm(new_centers - cluster_centers, dim=1).max()
        cluster_centers = new_centers

        if iteration % 10 == 0:
            print(f"  迭代 {iteration}: 中心偏移 = {center_shift:.6f}")

        if center_shift < tol:
            print(f"  收敛于迭代 {iteration}")
            break

    print(f"GPU K-means聚类完成，最终迭代次数: {iteration + 1}")
    return cluster_centers, cluster_assignments


def kmeans_plus_plus_init(data_tensor, n_clusters, device):
    """
    K-means++初始化方法
    """
    N, D = data_tensor.shape
    centers = torch.zeros(n_clusters, D, device=device)

    # 随机选择第一个中心
    centers[0] = data_tensor[torch.randint(0, N, (1,))]

    for k in range(1, n_clusters):
        # 计算每个点到已选中心的最小距离
        distances = torch.cdist(data_tensor, centers[:k], p=2)  # [N, k]
        min_distances = torch.min(distances, dim=1)[0]  # [N]

        # 按距离平方的概率选择下一个中心
        probabilities = min_distances ** 2
        probabilities = probabilities / probabilities.sum()

        # 累积分布函数采样
        cumulative_probs = torch.cumsum(probabilities, dim=0)
        r = torch.rand(1, device=device)
        selected_idx = torch.searchsorted(cumulative_probs, r)

        centers[k] = data_tensor[selected_idx]

    return centers


def perform_clustering_and_pseudo_labeling(labeled_features, labeled_labels, unlabeled_features, num_classes=10,
                                           save_visualization=None, similarity_method='cosine',
                                           cosine_threshold=0.5, distance_threshold=10.0, device='cpu',
                                           use_separate_clustering=True, unlabeled_n_clusters=None):
    """
    对有标签数据进行聚类，并为无标签数据分配伪标签
    """
    print(f"开始处理，有标签数据: {labeled_features.shape[0]}, 无标签数据: {unlabeled_features.shape[0]}")

    # 计算每个类别的特征平均值作为类别中心
    print(f"\n=== 有标签数据类别中心计算 ===")
    cluster_centers = np.zeros((num_classes, labeled_features.shape[1]))
    cluster_to_label = {}
    cluster_assignment_details = {}

    for class_id in range(num_classes):
        # 找到属于当前类别的所有样本
        class_mask = labeled_labels == class_id
        class_samples = np.sum(class_mask)

        if class_samples > 0:
            # 计算该类别的特征平均值
            class_features = labeled_features[class_mask]
            class_center = np.mean(class_features, axis=0)
            cluster_centers[class_id] = class_center
            cluster_to_label[class_id] = class_id

            # 记录详细信息
            cluster_assignment_details[class_id] = {
                'assigned_label': int(class_id),
                'total_samples': int(class_samples),
                'label_distribution': {class_id: class_samples},
                'purity': 1.0  # 直接使用真实标签，纯度为100%
            }

            # 计算类内特征的标准差作为紧密度指标
            class_std = np.mean(np.std(class_features, axis=0))

            print(f"类别 {class_id} → 样本数: {class_samples}, 特征维度: {class_center.shape[0]}")
            print(f"  特征均值范围: [{np.min(class_center):.4f}, {np.max(class_center):.4f}]")
            print(f"  类内特征标准差: {class_std:.4f}")
        else:
            # 如果某个类别没有样本，使用零向量作为中心
            cluster_centers[class_id] = np.zeros(labeled_features.shape[1])
            cluster_to_label[class_id] = class_id
            cluster_assignment_details[class_id] = {
                'assigned_label': int(class_id),
                'total_samples': 0,
                'label_distribution': {},
                'purity': 0.0
            }
            print(f"类别 {class_id} → 无样本，使用零向量作为中心")

    # 计算类别间的距离统计
    print(f"\n=== 类别中心距离分析 ===")
    center_distances = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if cluster_assignment_details[i]['total_samples'] > 0 and cluster_assignment_details[j][
                'total_samples'] > 0:
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                center_distances.append(dist)
                print(f"类别 {i} ↔ 类别 {j}: 距离 = {dist:.4f}")

    if center_distances:
        print(f"类别间平均距离: {np.mean(center_distances):.4f}")
        print(f"类别间最小距离: {np.min(center_distances):.4f}")
        print(f"类别间最大距离: {np.max(center_distances):.4f}")

    # 统计有标签数据的分布
    total_labeled_samples = len(labeled_labels)
    valid_classes = sum([1 for details in cluster_assignment_details.values() if details['total_samples'] > 0])
    print(f"\n有标签数据统计:")
    print(f"  总样本数: {total_labeled_samples}")
    print(f"  有效类别数: {valid_classes}/{num_classes}")
    print(f"  类别分布: {[details['total_samples'] for details in cluster_assignment_details.values()]}")

    # 可视化聚类质量（如果指定了保存路径）
    if save_visualization:
        # 由于我们现在直接使用类别中心而不是聚类，所以使用真实标签作为"聚类分配"
        visualize_clustering_quality(labeled_features, labeled_labels, labeled_labels,
                                     cluster_to_label, save_visualization)

    # 改进的伪标签分配策略：对无标签数据进行独立聚类
    print(f"\n=== 无标签数据聚类 ===")
    print(f"使用 {similarity_method} 方法进行伪标签分配")

    if use_separate_clustering:
        # 使用独立的无标签聚类策略
        if unlabeled_n_clusters is None:
            unlabeled_n_clusters = min(max(unlabeled_features.shape[0] // 50, num_classes * 2), 100)

        print(f"对无标签数据进行独立聚类，聚类数: {unlabeled_n_clusters}")

        # 使用GPU加速的K-means聚类
        unlabeled_features_tensor = torch.from_numpy(unlabeled_features).float().to(device)
        unlabeled_cluster_centers, unlabeled_cluster_assignments = gpu_kmeans_clustering(
            unlabeled_features_tensor, unlabeled_n_clusters, device, max_iters=100
        )

        # 计算无标签聚类中心与有标签聚类中心的相似度
        labeled_centers_tensor = torch.from_numpy(cluster_centers).float().to(device)

        if similarity_method == 'cosine':
            # 余弦相似度计算
            print(f"余弦相似度阈值: {cosine_threshold}")

            # 归一化
            unlabeled_centers_norm = torch.nn.functional.normalize(unlabeled_cluster_centers, p=2, dim=1)
            labeled_centers_norm = torch.nn.functional.normalize(labeled_centers_tensor, p=2, dim=1)

            # 计算相似度矩阵 [unlabeled_clusters, labeled_clusters]
            similarity_matrix = torch.mm(unlabeled_centers_norm, labeled_centers_norm.t())

            # 为每个无标签聚类分配标签
            unlabeled_cluster_to_label = {}
            discarded_clusters = set()

            for unlabeled_cluster_id in range(unlabeled_n_clusters):
                max_similarity = torch.max(similarity_matrix[unlabeled_cluster_id])

                if max_similarity >= cosine_threshold:
                    best_labeled_cluster = torch.argmax(similarity_matrix[unlabeled_cluster_id]).item()
                    assigned_label = cluster_to_label[best_labeled_cluster]
                    unlabeled_cluster_to_label[unlabeled_cluster_id] = assigned_label
                    # print(f"  无标签聚类 {unlabeled_cluster_id} → 标签 {assigned_label} (相似度: {max_similarity:.4f})")
                else:
                    unlabeled_cluster_to_label[unlabeled_cluster_id] = -1  # 标记为丢弃
                    discarded_clusters.add(unlabeled_cluster_id)
                    # print(
                    #     f"  无标签聚类 {unlabeled_cluster_id} → 丢弃 (最大相似度: {max_similarity:.4f} < {cosine_threshold})")

        else:  # distance method
            print(f"欧氏距离阈值: {distance_threshold}")

            # 计算距离矩阵 [unlabeled_clusters, labeled_clusters]
            distance_matrix = torch.cdist(unlabeled_cluster_centers, labeled_centers_tensor, p=2)

            # 为每个无标签聚类分配标签
            unlabeled_cluster_to_label = {}
            discarded_clusters = set()

            for unlabeled_cluster_id in range(unlabeled_n_clusters):
                min_distance = torch.min(distance_matrix[unlabeled_cluster_id])

                if min_distance <= distance_threshold:
                    best_labeled_cluster = torch.argmin(distance_matrix[unlabeled_cluster_id]).item()
                    assigned_label = cluster_to_label[best_labeled_cluster]
                    unlabeled_cluster_to_label[unlabeled_cluster_id] = assigned_label
                    # print(f"  无标签聚类 {unlabeled_cluster_id} → 标签 {assigned_label} (距离: {min_distance:.4f})")
                else:
                    unlabeled_cluster_to_label[unlabeled_cluster_id] = -1  # 标记为丢弃
                    discarded_clusters.add(unlabeled_cluster_id)
                    # print(
                    # f"  无标签聚类 {unlabeled_cluster_id} → 丢弃 (最小距离: {min_distance:.4f} > {distance_threshold})")

        # 根据无标签数据的聚类结果分配伪标签
        pseudo_labels = []
        discarded_count = 0

        unlabeled_cluster_assignments_cpu = unlabeled_cluster_assignments.cpu().numpy()

        for i in range(unlabeled_features.shape[0]):
            cluster_id = unlabeled_cluster_assignments_cpu[i]
            assigned_label = unlabeled_cluster_to_label[cluster_id]

            if assigned_label != -1:
                pseudo_labels.append(assigned_label)
            else:
                pseudo_labels.append(-1)  # 丢弃的样本
                discarded_count += 1

        # 添加详细的无标签聚类分配统计
        print(f"\n=== 无标签聚类详细分配统计 ===")
        cluster_sample_counts = {}
        for cluster_id in range(unlabeled_n_clusters):
            cluster_mask = unlabeled_cluster_assignments_cpu == cluster_id
            cluster_size = np.sum(cluster_mask)
            assigned_label = unlabeled_cluster_to_label[cluster_id]
            cluster_sample_counts[cluster_id] = {
                'size': cluster_size,
                'assigned_label': assigned_label
            }
            #
            # if assigned_label != -1:
            #     print(f"无标签聚类 {cluster_id} → 类别 {assigned_label} (样本数: {cluster_size})")
            # else:
            #     print(f"无标签聚类 {cluster_id} → 丢弃 (样本数: {cluster_size})")

        # 按类别汇总统计
        label_summary = {}
        discarded_samples = 0
        for cluster_id, info in cluster_sample_counts.items():
            label = info['assigned_label']
            size = info['size']
            if label != -1:
                if label not in label_summary:
                    label_summary[label] = {'clusters': [], 'total_samples': 0}
                label_summary[label]['clusters'].append(cluster_id)
                label_summary[label]['total_samples'] += size
            else:
                discarded_samples += size

        # print(f"\n=== 按类别汇总的无标签聚类分配 ===")
        # for label in sorted(label_summary.keys()):
        #     info = label_summary[label]
        #     print(f"类别 {label}: {len(info['clusters'])} 个聚类, 总样本数: {info['total_samples']}")
        #     print(f"  聚类ID: {info['clusters']}")
        #
        # if discarded_samples > 0:
        #     discarded_cluster_ids = [cid for cid, info in cluster_sample_counts.items() if info['assigned_label'] == -1]
        #     print(f"丢弃: {len(discarded_cluster_ids)} 个聚类, 总样本数: {discarded_samples}")
        #     print(f"  丢弃聚类ID: {discarded_cluster_ids}")
    else:
        # 使用传统的聚类策略（原有逻辑）
        print("使用传统聚类策略")
        unlabeled_features_tensor = torch.from_numpy(unlabeled_features).float().to(device)
        cluster_centers_tensor = torch.from_numpy(cluster_centers).float().to(device)

        if similarity_method == 'cosine':
            # 计算余弦相似度
            unlabeled_norm = torch.nn.functional.normalize(unlabeled_features_tensor, p=2, dim=1)
            centers_norm = torch.nn.functional.normalize(cluster_centers_tensor, p=2, dim=1)
            similarities = torch.mm(unlabeled_norm, centers_norm.t())

            # 找到最相似的聚类中心
            max_similarities, closest_clusters = torch.max(similarities, dim=1)

            # 分配伪标签
            pseudo_labels = []
            discarded_count = 0

            for i in range(len(unlabeled_features)):
                if max_similarities[i] >= cosine_threshold:
                    cluster_id = closest_clusters[i].item()
                    pseudo_labels.append(cluster_to_label[cluster_id])
                else:
                    pseudo_labels.append(-1)  # 丢弃的样本
                    discarded_count += 1
        else:
            # 计算欧氏距离
            distances = torch.cdist(unlabeled_features_tensor, cluster_centers_tensor, p=2)
            min_distances, closest_clusters = torch.min(distances, dim=1)

            # 分配伪标签
            pseudo_labels = []
            discarded_count = 0

            for i in range(len(unlabeled_features)):
                if min_distances[i] <= distance_threshold:
                    cluster_id = closest_clusters[i].item()
                    pseudo_labels.append(cluster_to_label[cluster_id])
                else:
                    pseudo_labels.append(-1)  # 丢弃的样本
                    discarded_count += 1

        # 为了保持一致性，创建虚拟的无标签聚类变量
        unlabeled_cluster_to_label = {}
        discarded_clusters = set()
        unlabeled_n_clusters = num_classes

    pseudo_labels = np.array(pseudo_labels)

    # 统计丢弃率
    total_samples = len(pseudo_labels)
    discard_rate = discarded_count / total_samples
    valid_samples = total_samples - discarded_count

    print(f"\n=== 伪标签分配统计 ===")
    print(f"总样本数: {total_samples}")
    print(f"有效样本数: {valid_samples}")
    print(f"丢弃样本数: {discarded_count}")
    print(f"丢弃率: {discard_rate:.4f} ({discard_rate * 100:.2f}%)")

    # 统计无标签聚类的分配情况
    print(f"\n=== 无标签聚类分配统计 ===")
    valid_clusters = len([k for k, v in unlabeled_cluster_to_label.items() if v != -1])
    discarded_clusters_count = len(discarded_clusters)
    print(f"总聚类数: {unlabeled_n_clusters}")
    print(f"有效聚类数: {valid_clusters}")
    print(f"丢弃聚类数: {discarded_clusters_count}")
    print(f"聚类有效率: {valid_clusters / unlabeled_n_clusters:.4f}")

    # 统计每个聚类的样本数分布（仅在使用独立聚类时）
    if use_separate_clustering and 'unlabeled_cluster_assignments_cpu' in locals():
        cluster_sizes = {}
        for cluster_id in range(unlabeled_n_clusters):
            cluster_mask = unlabeled_cluster_assignments_cpu == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_sizes[cluster_id] = cluster_size

        print(f"聚类大小统计:")
        print(f"  平均聚类大小: {np.mean(list(cluster_sizes.values())):.1f}")
        print(f"  最大聚类大小: {max(cluster_sizes.values())}")
        print(f"  最小聚类大小: {min(cluster_sizes.values())}")

    # 统计有效伪标签的分布
    valid_labels = pseudo_labels[pseudo_labels != -1]
    if len(valid_labels) > 0:
        label_distribution = np.bincount(valid_labels, minlength=num_classes)
        print(f"有效伪标签分布: {label_distribution}")

        # 计算标签分布的均匀性
        expected_per_class = len(valid_labels) / num_classes
        distribution_variance = np.var(label_distribution)
        print(f"标签分布方差: {distribution_variance:.2f} (期望均匀分布方差: {0:.2f})")
    else:
        print("警告: 没有有效的伪标签！")

    # 提供参数调优建议
    print(f"\n=== 参数调优建议 ===")
    if discard_rate > 0.5:
        print(f"⚠️  丢弃率过高 ({discard_rate:.2%})，建议:")
        if similarity_method == 'cosine':
            print(f"   - 降低余弦相似度阈值 (当前: {cosine_threshold})")
        else:
            print(f"   - 增加欧氏距离阈值 (当前: {distance_threshold})")
        print(f"   - 增加无标签聚类数量 (当前: {unlabeled_n_clusters})")
        print(f"   - 检查特征提取质量")
    elif discard_rate < 0.1:
        print(f"✅ 丢弃率较低 ({discard_rate:.2%})，可以考虑:")
        print(f"   - 提高相似度阈值以获得更高质量的伪标签")
        print(f"   - 减少聚类数量以提高计算效率")

    if use_separate_clustering and valid_clusters / unlabeled_n_clusters < 0.3:
        print(f"⚠️  聚类有效率较低，建议:")
        print(f"   - 减少无标签聚类数量")
        print(f"   - 调整聚类初始化策略")
        print(f"   - 检查数据预处理")

    if len(valid_labels) > 0:
        label_imbalance = max(label_distribution) / (min(label_distribution) + 1e-8)
        if label_imbalance > 5:
            print(f"⚠️  标签分布不均衡 (最大/最小比例: {label_imbalance:.1f})，建议:")
            print(f"   - 使用分层聚类")
            print(f"   - 调整相似度阈值")
            print(f"   - 增加有标签数据的多样性")

    return pseudo_labels, cluster_centers, cluster_to_label


# ----------------------------------------------------------------------------
# 图像保存函数

def save_images_by_pseudo_labels(images, pseudo_labels, true_labels, outdir, max_images_per_class=100,
                                 dataset_type='cifar10'):
    """
    根据伪标签分配结果保存图片到不同的文件夹

    Args:
        images: 图像数据 (N, 3, 32, 32) numpy array, 值范围 0-255
        pseudo_labels: 伪标签数组 (N,), -1表示被丢弃的样本
        true_labels: 真实标签数组 (N,), 用于验证
        outdir: 输出目录
        max_images_per_class: 每个类别最多保存的图像数量
        dataset_type: 数据集类型，用于显示正确的类别名称
    """
    print(f"\n=== 开始保存伪标签分配的图像 ===")

    # 确定类别名称
    if dataset_type == 'stl10':
        # STL-10: 1-airplane, 2-bird, 3-car, 4-cat, 5-deer, 6-dog, 7-horse, 8-monkey, 9-ship, 10-truck
        # Zero-indexed: 0-airplane, 1-bird, 2-car, 3-cat, 4-deer, 5-dog, 6-horse, 7-monkey, 8-ship, 9-truck
        class_names = [
            'airplane', 'bird', 'car', 'cat', 'deer',
            'dog', 'horse', 'monkey', 'ship', 'truck'
        ]
    elif dataset_type == 'cifar10':
        # CIFAR-10类别名称
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_type == 'fashion-mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_type == 'mnist':
        class_names = [str(i) for i in range(10)]
    elif dataset_type == 'svhn':
        class_names = [str(i) for i in range(10)]
    else:
        # 默认通用名称
        class_names = [f'class_{i}' for i in range(100)]

    # 创建保存目录
    save_dir = os.path.join(outdir, 'pseudo_labeled_images')
    os.makedirs(save_dir, exist_ok=True)

    num_classes = len(class_names)

    # 为每个类别创建子目录
    for class_id in range(num_classes):
        class_dir = os.path.join(save_dir, f'class_{class_id}_{class_names[class_id]}')
        os.makedirs(class_dir, exist_ok=True)

    # 创建丢弃样本目录
    discarded_dir = os.path.join(save_dir, 'discarded')
    os.makedirs(discarded_dir, exist_ok=True)

    # 统计信息
    saved_counts = {i: 0 for i in range(num_classes)}
    discarded_count = 0
    accuracy_counts = {i: {'correct': 0, 'total': 0} for i in range(num_classes)}

    print(f"保存目录: {save_dir}")
    print(f"每个类别最多保存: {max_images_per_class} 张图像")

    # 遍历所有图像
    for idx, (image, pseudo_label, true_label) in enumerate(zip(images, pseudo_labels, true_labels)):
        # 转换图像格式 (3, 32, 32) -> (32, 32, 3)
        if image.shape[0] == 3:  # CHW格式
            image_hwc = image.transpose(1, 2, 0)
        else:  # 已经是HWC格式
            image_hwc = image

        # 确保图像值在正确范围内
        if image_hwc.max() <= 1.0:  # 如果是[0,1]范围，转换为[0,255]
            image_hwc = (image_hwc * 255).astype(np.uint8)
        else:
            image_hwc = image_hwc.astype(np.uint8)

        if pseudo_label == -1:
            # 保存被丢弃的样本
            if discarded_count < max_images_per_class:
                filename = f'discarded_{discarded_count:04d}_true_{true_label}_{class_names[true_label]}.png'
                save_path = os.path.join(discarded_dir, filename)
                save_single_image(image_hwc, save_path)
                discarded_count += 1
        else:
            # 保存有伪标签的样本
            if saved_counts[pseudo_label] < max_images_per_class:
                # 检查预测是否正确
                is_correct = (pseudo_label == true_label)
                accuracy_counts[pseudo_label]['total'] += 1
                if is_correct:
                    accuracy_counts[pseudo_label]['correct'] += 1

                # 生成文件名
                correct_str = 'correct' if is_correct else 'wrong'
                filename = f'{correct_str}_{saved_counts[pseudo_label]:04d}_pred_{pseudo_label}_true_{true_label}.png'

                class_dir = os.path.join(save_dir, f'class_{pseudo_label}_{class_names[pseudo_label]}')
                save_path = os.path.join(class_dir, filename)

                save_single_image(image_hwc, save_path)
                saved_counts[pseudo_label] += 1

    # 打印保存统计信息
    print(f"\n=== 图像保存统计 ===")
    total_saved = sum(saved_counts.values())
    print(f"总共保存图像: {total_saved + discarded_count}")
    print(f"丢弃样本: {discarded_count}")

    print(f"\n各类别保存数量:")
    for class_id in range(num_classes):
        count = saved_counts[class_id]
        if accuracy_counts[class_id]['total'] > 0:
            acc = accuracy_counts[class_id]['correct'] / accuracy_counts[class_id]['total']
            print(f"  类别 {class_id} ({class_names[class_id]}): {count} 张, 准确率: {acc:.3f}")
        else:
            print(f"  类别 {class_id} ({class_names[class_id]}): {count} 张")

    print(f"\n图像已保存到: {save_dir}")


def save_single_image(image_array, save_path):
    """
    保存单张图像

    Args:
        image_array: 图像数组 (H, W, 3), 值范围 0-255
        save_path: 保存路径
    """
    try:
        from PIL import Image

        # 确保图像是uint8格式
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        # 创建PIL图像并保存
        image = Image.fromarray(image_array)
        image.save(save_path)

    except ImportError:
        # 如果没有PIL，使用matplotlib保存
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(2, 2))
            plt.imshow(image_array)
            plt.axis('off')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()

        except ImportError:
            # 如果都没有，保存为numpy文件
            np.save(save_path.replace('.png', '.npy'), image_array)


# ----------------------------------------------------------------------------
# 数据处理函数

def split_labeled_unlabeled(images, labels, label_ratio):
    """将数据分割为有标签和无标签部分"""
    n_total = len(images)
    n_labeled = int(n_total * label_ratio)

    indices = np.random.permutation(n_total)
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]

    labeled_images = images[labeled_indices]
    labeled_labels = labels[labeled_indices]
    unlabeled_images = images[unlabeled_indices]
    unlabeled_labels = labels[unlabeled_indices]

    print(f"数据分割完成: 有标签 {len(labeled_images)}, 无标签 {len(unlabeled_images)}")
    return labeled_images, labeled_labels, unlabeled_images, unlabeled_labels


def load_cifar10_data(tarball_path: str, max_images: Optional[int] = None):
    """从CIFAR-10 tar.gz文件加载数据"""
    images = []
    labels = []

    with tarfile.open(tarball_path, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)

    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]

    return images, labels


def load_mnist_data(data_path: str, max_images: Optional[int] = None):
    """从MNIST数据文件加载数据

    Args:
        data_path: MNIST数据文件路径（可以是.gz文件或目录）
        max_images: 最大图像数量限制

    Returns:
        images: 图像数据 (N, 1, 28, 28) - 单通道灰度图像
        labels: 标签数据 (N,)
    """
    import gzip
    import struct

    # MNIST类别名称
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    try:
        # 尝试使用torchvision加载MNIST
        import torchvision

        # 如果data_path是目录，使用torchvision加载
        if os.path.isdir(data_path):
            dataset = torchvision.datasets.MNIST(
                root=data_path, train=True, download=True, transform=None
            )

            # 转换为numpy格式，保持原始的单通道28x28格式
            images = []
            labels = []
            for i in range(len(dataset)):
                img, label = dataset[i]
                # 直接使用PIL图像转numpy，不使用tensor
                img_array = np.array(img)  # (28, 28)
                images.append(img_array)
                labels.append(label)

            images = np.array(images)  # (N, 28, 28)
            labels = np.array(labels)

            # 添加通道维度：(N, 28, 28) -> (N, 1, 28, 28)
            images = np.expand_dims(images, axis=1)

        else:
            # 手动解析MNIST文件格式
            # 假设data_path指向训练图像文件
            if data_path.endswith('.gz'):
                open_func = gzip.open
            else:
                open_func = open

            # 加载图像文件
            with open_func(data_path, 'rb') as f:
                magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
                if magic != 2051:
                    raise ValueError(f"Invalid magic number {magic} for MNIST images")

                images_data = f.read()
                images = np.frombuffer(images_data, dtype=np.uint8)
                images = images.reshape(num_images, rows, cols)

            # 加载标签文件（假设在同一目录下）
            label_path = data_path.replace('images', 'labels').replace('train-images', 'train-labels')
            with open_func(label_path, 'rb') as f:
                magic, num_labels = struct.unpack('>II', f.read(8))
                if magic != 2049:
                    raise ValueError(f"Invalid magic number {magic} for MNIST labels")

                labels_data = f.read()
                labels = np.frombuffer(labels_data, dtype=np.uint8)

            # 保持原始格式：(N, 28, 28) -> (N, 1, 28, 28)
            images = np.expand_dims(images, axis=1)  # (N, 1, 28, 28)

    except ImportError:
        raise ImportError("需要安装torchvision来加载MNIST数据集")

    # 限制图像数量
    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]

    print(f"MNIST数据加载完成: {len(images)} 张图像, 形状: {images.shape}")
    print(f"类别分布: {np.bincount(labels)}")

    return images, labels


def load_fashion_mnist_data(data_path: str, max_images: Optional[int] = None):
    """从Fashion-MNIST数据文件加载数据

    Args:
        data_path: Fashion-MNIST数据文件路径（可以是.gz文件或目录）
        max_images: 最大图像数量限制

    Returns:
        images: 图像数据 (N, 1, 28, 28) - 单通道灰度图像
        labels: 标签数据 (N,)
    """
    import gzip
    import struct

    # Fashion-MNIST类别名称
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    try:
        # 尝试使用torchvision加载Fashion-MNIST
        import torchvision

        # 如果data_path是目录，使用torchvision加载
        if os.path.isdir(data_path):
            dataset = torchvision.datasets.FashionMNIST(
                root=data_path, train=True, download=True, transform=None
            )

            # 转换为numpy格式，保持原始的单通道28x28格式
            images = []
            labels = []
            for i in range(len(dataset)):
                img, label = dataset[i]
                # 直接使用PIL图像转numpy，不使用tensor
                img_array = np.array(img)  # (28, 28)
                images.append(img_array)
                labels.append(label)

            images = np.array(images)  # (N, 28, 28)
            labels = np.array(labels)

            # 添加通道维度：(N, 28, 28) -> (N, 1, 28, 28)
            images = np.expand_dims(images, axis=1)

        else:
            # 手动解析Fashion-MNIST文件格式
            # 假设data_path指向训练图像文件
            if data_path.endswith('.gz'):
                open_func = gzip.open
            else:
                open_func = open

            # 加载图像文件
            with open_func(data_path, 'rb') as f:
                magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
                if magic != 2051:
                    raise ValueError(f"Invalid magic number {magic} for Fashion-MNIST images")

                images_data = f.read()
                images = np.frombuffer(images_data, dtype=np.uint8)
                images = images.reshape(num_images, rows, cols)

            # 加载标签文件（假设在同一目录下）
            label_path = data_path.replace('images', 'labels').replace('train-images', 'train-labels')
            with open_func(label_path, 'rb') as f:
                magic, num_labels = struct.unpack('>II', f.read(8))
                if magic != 2049:
                    raise ValueError(f"Invalid magic number {magic} for Fashion-MNIST labels")

                labels_data = f.read()
                labels = np.frombuffer(labels_data, dtype=np.uint8)

            # 保持原始格式：(N, 28, 28) -> (N, 1, 28, 28)
            images = np.expand_dims(images, axis=1)  # (N, 1, 28, 28)

    except ImportError:
        raise ImportError("需要安装torchvision来加载Fashion-MNIST数据集")

    # 限制图像数量
    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]

    print(f"Fashion-MNIST数据加载完成: {len(images)} 张图像, 形状: {images.shape}")
    print(f"类别分布: {np.bincount(labels)}")

    return images, labels


def load_cifar100_data(tarball_path: str, max_images: Optional[int] = None):
    """从CIFAR-100 tar.gz文件加载数据"""
    images = []
    labels = []

    with tarfile.open(tarball_path, 'r:gz') as tar:
        # CIFAR-100只有一个训练文件
        member = tar.getmember('cifar-100-python/train')
        with tar.extractfile(member) as file:
            data = pickle.load(file, encoding='latin1')
        images = data['data'].reshape(-1, 3, 32, 32)
        # labels = data['fine_labels']  # 使用细粒度标签（100个类别）
        labels = np.array(data['fine_labels'])  # 转换为numpy数组并使用细粒度标签（100个类别）

    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]

    return images, labels


def load_svhn_data(data_path: str, max_images: Optional[int] = None):
    """从SVHN数据集加载数据

    支持两种来源：
    - 若 `data_path` 是目录：使用 torchvision 自动下载并加载 `SVHN(split='train')`
    - 若 `data_path` 指向 `.mat` 文件：尝试使用 scipy.io 解析 `train_32x32.mat`/`test_32x32.mat`

    返回值：
    - images: 图像数据 (N, 3, 32, 32)，uint8，范围 0-255
    - labels: 标签数据 (N,)，范围 0-9（将原始 10 归一化为 0）
    """
    images = []
    labels = []

    try:
        import torchvision
    except ImportError:
        torchvision = None

    if os.path.isdir(data_path) and torchvision is not None:
        # 使用 torchvision 加载 SVHN
        dataset = torchvision.datasets.SVHN(root=data_path, split='train', download=True, transform=None)
        for i in range(len(dataset)):
            img, label = dataset[i]  # img: PIL.Image, label: int (0-9 或 10)
            img_np = np.array(img)  # (32, 32, 3)
            # 转为 CHW
            images.append(img_np.transpose(2, 0, 1))
            labels.append(int(label))

        images = np.array(images, dtype=np.uint8)  # (N, 3, 32, 32)
        labels = np.array(labels, dtype=np.int64)
        # 归一化 10 -> 0（若存在）
        labels = labels % 10

    else:
        # 尝试解析 .mat 文件
        if not str(data_path).lower().endswith('.mat'):
            raise ValueError("SVHN数据路径应为目录（用于torchvision下载）或 .mat 文件路径")

        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("需要安装 scipy 才能从 .mat 文件加载SVHN；或提供目录以便使用 torchvision 下载")

        mat = sio.loadmat(data_path)
        # 官方 .mat 文件字段：X: (32, 32, 3, N), y: (N, 1)
        X = mat.get('X')
        y = mat.get('y')
        if X is None or y is None:
            raise ValueError("SVHN .mat 文件缺少必要字段 'X' 或 'y'")

        # 转换到 (N, 3, 32, 32)
        images = np.transpose(X, (3, 2, 0, 1)).astype(np.uint8)
        labels = y.reshape(-1).astype(np.int64)
        labels = labels % 10

    # 限制数量
    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]

    print(f"SVHN数据加载完成: {len(images)} 张图像, 形状: {images.shape}")
    print(f"类别分布: {np.bincount(labels)}")
    return images, labels

    return images, labels


def load_stl10_data(data_path: str, max_images: Optional[int] = None):
    """从STL-10数据集加载数据 (加载train和test split用于验证)"""
    images = []
    labels = []

    try:
        import torchvision
    except ImportError:
        raise ImportError("需要安装torchvision来加载STL-10数据集")

    # 如果data_path是目录，使用torchvision加载
    if os.path.isdir(data_path):
        print(f"Loading STL-10 from {data_path} (train+test splits)...")
        # 加载 train split
        train_ds = torchvision.datasets.STL10(root=data_path, split='train', download=True)
        # 加载 test split
        test_ds = torchvision.datasets.STL10(root=data_path, split='test', download=True)

        # 合并数据
        images = np.concatenate([train_ds.data, test_ds.data], axis=0)  # (N, 3, 96, 96)
        labels = np.concatenate([train_ds.labels, test_ds.labels], axis=0)  # (N,)
        labels = np.array(labels, dtype=np.int64)

    else:
        raise ValueError("STL-10 data_path should be a directory.")

    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]

    print(f"STL-10数据加载完成: {len(images)} 张图像, 形状: {images.shape}")
    print(f"类别分布: {np.bincount(labels)}")

    return images, labels


def load_dataset_data(tarball_path: str, dataset_type: str, max_images: Optional[int] = None):
    """根据数据集类型加载相应的数据"""
    if dataset_type == 'cifar10':
        return load_cifar10_data(tarball_path, max_images)
    elif dataset_type == 'cifar100':
        return load_cifar100_data(tarball_path, max_images)
    elif dataset_type == 'fashion-mnist':
        return load_fashion_mnist_data(tarball_path, max_images)
    elif dataset_type == 'mnist':
        return load_mnist_data(tarball_path, max_images)
    elif dataset_type == 'svhn':
        return load_svhn_data(tarball_path, max_images)
    elif dataset_type == 'stl10':
        return load_stl10_data(tarball_path, max_images)
    else:
        raise ValueError(
            f"不支持的数据集类型: {dataset_type}，支持的类型: cifar10, cifar100, fashion-mnist, mnist, svhn, stl10")


def add_noise_edm(image, sigma, device='cpu'):
    """
    使用EDM参数化给图像添加噪声
    image: 输入图像 tensor, 范围 [-1, 1]
    sigma: 噪声水平
    """
    noise = torch.randn_like(image)
    noisy_image = image + sigma * noise
    return noisy_image


def preprocess_image(image_np):
    """将numpy图像 (0-255) 转换为tensor (-1, 1)"""
    image = torch.from_numpy(image_np.astype(np.float32)) / 127.5 - 1.0
    return image


def generate_sigma_schedule(sigma_min, sigma_max, num_steps, rho):
    """生成噪声时间步调度"""
    step_indices = torch.arange(num_steps, dtype=torch.float64)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return t_steps


# ----------------------------------------------------------------------------
# 主函数
@click.command()
@click.option('--data', help='数据集tar.gz文件路径', metavar='PATH', type=str,
              # default='/data/psw/edm/data')
              # default='/data/psw/edm/data/cifar-10-python.tar.gz')
              # default='/data/psw/DFLSemi/samples/cifar10')
              default='/data/psw/DDPM/data/cifar-10-python.tar.gz')
# default='/data/psw/DFLSemi_diffusion/samples/cifar100/cifar-100-python.tar.gz')
@click.option('--outdir', help='输出目录', metavar='DIR', type=str, default='pseudo_label_output')
@click.option('--network_pkl', help='预训练网络的pickle文件路径', metavar='PATH', type=str,
              # default='/data/psw/edm/checkpoint/edm-cifar10-32x32-uncond-vp.pkl')
              default='/data/psw/edm/run/00013-stl10-uncond-ddpmpp-edm-gpus4-batch64-fp32/network-snapshot-042540.pkl')
# default='/data/psw/edm/ckpt_mnist/network-snapshot-019344.pkl')
@click.option('--label_ratio', help='有标签数据的比例', metavar='FLOAT', type=float, default=0.05)
@click.option('--max_images', help='使用的最大图像数量', metavar='INT', type=int, default=10000)
@click.option('--batch_size', help='批处理大小', metavar='INT', type=int, default=64)
@click.option('--test_sampling', help='是否进行采样测试验证模型', is_flag=True, default=False)
@click.option('--num_sample', help='是否进行采样测试验证模型', type=int, default=16)
@click.option('--visualize_clustering', help='是否可视化聚类质量', is_flag=True, default=False)
@click.option('--similarity_method', help='相似度计算方法', type=click.Choice(['cosine', 'distance']), default='cosine')
@click.option('--cosine_threshold', help='余弦相似度阈值', metavar='FLOAT', type=float, default=0.3)
@click.option('--distance_threshold', help='欧氏距离阈值', metavar='FLOAT', type=float, default=3200.0)
@click.option('--unlabeled_cluster_ratio', help='无标签聚类数量相对于样本数的比例', metavar='FLOAT', type=float,
              default=2.0)
@click.option('--min_unlabeled_clusters', help='最小无标签聚类数量', metavar='INT', type=int, default=10)
@click.option('--max_unlabeled_clusters', help='最大无标签聚类数量', metavar='INT', type=int, default=500)
@click.option('--use_separate_clustering', help='是否使用独立的无标签聚类策略', is_flag=True, default=True)
@click.option('--save_images', help='是否保存伪标签分配的图像', is_flag=True, default=False)
@click.option('--max_images_per_class', help='每个类别最多保存的图像数量', metavar='INT', type=int, default=7000)
@click.option('--dataset_type', help='数据集类型',
              type=click.Choice(['cifar10', 'cifar100', 'fashion-mnist', 'mnist', 'svhn', 'stl10']),
              default='cifar10')
@click.option('--save_features', help='是否保存提取的特征用于t-SNE可视化', is_flag=True, default=False)
@click.option('--max_unlabeled_clusters_values', help='测试的max上界列表(逗号分隔或范围start:end:step)', metavar='STR',
              type=str, default=None)
@click.option('--use_dp', help='是否对中间层特征应用本地差分隐私', is_flag=True, default=False)
@click.option('--dp_epsilon', help='差分隐私预算 epsilon', type=float, default=1)
@click.option('--dp_delta', help='差分隐私参数 delta', type=float, default=1e-5)
@click.option('--dp_l2_clip', help='差分隐私 L2 范数截断阈值', type=float, default=100.0)
def main(data, outdir, network_pkl, label_ratio, max_images, batch_size,
         test_sampling, visualize_clustering, similarity_method, cosine_threshold, distance_threshold,
         unlabeled_cluster_ratio, min_unlabeled_clusters, max_unlabeled_clusters, use_separate_clustering,
         num_sample, save_images, max_images_per_class, dataset_type, save_features, max_unlabeled_clusters_values,
         use_dp, dp_epsilon, dp_delta, dp_l2_clip,
         sigma_max=80.0, sigma_min=0.002, num_steps=100,
         rho=7.0):
    """基于UNet中间特征的数据集伪标签构建"""
    # step_idx = [id for id in range(75,76)]
    # step_idx = [77]
    step_idx = [77]

    # 设置随机种子
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)

    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载网络
    print("加载预训练网络...")
    net = load_network(network_pkl, device)
    print(f"Params:{sum([p.numel() for p in net.parameters()])}")

    # 测试模型是否正确加载（如果启用）
    if test_sampling:
        print("\n=== 模型加载验证 ===")
        check_model_parameters(net)

        # 进行采样测试
        print("\n=== 采样测试 ===")
        try:
            sample_save_path = os.path.join(outdir, 'model_test_samples.png')
            samples = test_model_sampling(net, device, num_samples=num_sample, num_steps=18, save_path=sample_save_path)
            print("✅ 模型采样测试成功！模型参数已正确加载。")
        except Exception as e:
            print(f"❌ 模型采样测试失败: {e}")
            print("这可能表明模型参数未正确加载或模型结构有问题。")
            return
    else:
        print("跳过采样测试（使用 --test_sampling 启用）")

    # 加载数据集
    print(f"加载{dataset_type.upper()}数据...")
    images, labels = load_dataset_data(data, dataset_type, max_images)
    print(f"加载了 {len(images)} 张图像")

    # 根据数据集类型确定类别数量
    num_classes = 10 if dataset_type in ['cifar10', 'fashion-mnist', 'mnist', 'svhn', 'stl10'] else 100

    # 分割有标签和无标签数据
    labeled_images, labeled_labels, unlabeled_images, unlabeled_labels = split_labeled_unlabeled(
        images, labels, label_ratio
    )

    # 生成噪声时间步
    sigma_schedule = generate_sigma_schedule(sigma_min, sigma_max, num_steps, rho)
    sigma = []
    for step_id in step_idx:
        sigma.append(sigma_schedule[step_id].item())
        print(f"使用噪声水平 σ = {sigma_schedule[step_id].item():.6f} (步数 {step_id}/{num_steps - 1})")

    # 提取有标签数据的特征
    print("提取有标签数据的特征...")
    labeled_features_list = []
    num_labeled_batches = (len(labeled_images) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(labeled_images), batch_size),
                  desc="提取有标签特征",
                  total=num_labeled_batches,
                  unit="batch"):
        batch_images = labeled_images[i:i + batch_size]
        batch_features = extract_features_with_noise(net, batch_images, sigma, step_idx, device)
        labeled_features_list.append(batch_features)
    labeled_features = np.concatenate(labeled_features_list, axis=0)

    # 提取无标签数据的特征
    print("提取无标签数据的特征...")
    unlabeled_features_list = []
    num_unlabeled_batches = (len(unlabeled_images) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(unlabeled_images), batch_size),
                  desc="提取无标签特征",
                  total=num_unlabeled_batches,
                  unit="batch"):
        batch_images = unlabeled_images[i:i + batch_size]
        batch_features = extract_features_with_noise(net, batch_images, sigma, step_idx, device)
        unlabeled_features_list.append(batch_features)

    unlabeled_features = np.concatenate(unlabeled_features_list, axis=0)
    print(f"聚类特征维度：{unlabeled_features.shape}")
    print(f"特征提取完成: 有标签特征 {labeled_features.shape}, 无标签特征 {unlabeled_features.shape}")

    if use_dp:
        print("\n=== 对特征应用本地差分隐私 ===")
        labeled_features = apply_local_dp_to_features(
            labeled_features,
            epsilon=dp_epsilon,
            delta=dp_delta,
            l2_clip=dp_l2_clip
        )
        unlabeled_features = apply_local_dp_to_features(
            unlabeled_features,
            epsilon=dp_epsilon,
            delta=dp_delta,
            l2_clip=dp_l2_clip
        )
        print("差分隐私处理完成")

    # 保存特征用于t-SNE (如果启用)
    if save_features:
        print("\n=== 保存特征用于t-SNE ===")
        # 保存有标签特征
        save_features_for_tsne(
            labeled_features,
            labeled_labels,
            os.path.join(outdir, 'labeled_features.npz')
        )
        # 保存无标签特征
        save_features_for_tsne(
            unlabeled_features,
            unlabeled_labels,
            os.path.join(outdir, 'unlabeled_features.npz')
        )

        # 保存合并的特征
        all_features = np.concatenate([labeled_features, unlabeled_features], axis=0)
        all_labels = np.concatenate([labeled_labels, unlabeled_labels], axis=0)

        # 创建标记数组: 0表示无标签(原本), 1表示有标签
        is_labeled = np.zeros(len(all_labels), dtype=int)
        is_labeled[:len(labeled_labels)] = 1

        save_features_for_tsne(
            all_features,
            all_labels,
            os.path.join(outdir, 'all_features.npz')
        )
        # 另外保存一个包含是否为有标签数据的元数据
        np.savez_compressed(
            os.path.join(outdir, 'all_features_metadata.npz'),
            is_labeled=is_labeled
        )
        print("✅ 所有特征保存完成")
        return

    # 执行聚类和伪标签生成（包含可视化）
    print("执行聚类和伪标签生成...")
    visualization_path = os.path.join(outdir, 'clustering_quality_analysis.png') if visualize_clustering else None

    if max_unlabeled_clusters_values is not None:
        s = max_unlabeled_clusters_values.strip()
        m_list = []
        if ':' in s:
            parts = s.split(':')
            if len(parts) == 3:
                a, b, c = parts
                try:
                    m_min = int(a)
                    m_max = int(b)
                    m_step = int(c) if int(c) != 0 else 1
                    if m_min <= m_max:
                        m_list = list(range(m_min, m_max + 1, m_step))
                    else:
                        m_list = list(range(m_min, m_max - 1, -m_step))
                except Exception:
                    pass
        else:
            try:
                m_list = [int(x) for x in s.split(',') if x.strip()]
            except Exception:
                m_list = []

        if len(m_list) == 0:
            print("max上界列表解析失败，使用默认单次评估")
        else:
            print(f"max上界列表: {m_list}")
            results = []
            base_target = int(len(unlabeled_features) * unlabeled_cluster_ratio)
            for m in m_list:
                k_candidate = max(min_unlabeled_clusters, min(m, base_target))
                pseudo_labels, cluster_centers, cluster_to_label = perform_clustering_and_pseudo_labeling(
                    labeled_features, labeled_labels, unlabeled_features,
                    num_classes=num_classes,
                    save_visualization=None,
                    similarity_method=similarity_method,
                    cosine_threshold=cosine_threshold,
                    distance_threshold=distance_threshold,
                    device=device,
                    use_separate_clustering=True,
                    unlabeled_n_clusters=k_candidate
                )
                valid_mask = pseudo_labels != -1
                valid_pseudo_labels = pseudo_labels[valid_mask]
                valid_true_labels = unlabeled_labels[valid_mask]
                if len(valid_pseudo_labels) > 0:
                    acc = accuracy_score(valid_true_labels, valid_pseudo_labels)
                else:
                    acc = 0.0
                discard_rate = float(np.mean(pseudo_labels == -1))
                print(
                    f"max={m} → K={k_candidate}: 准确率={acc:.4f}, 有效样本={np.sum(valid_mask)}/{len(pseudo_labels)}, 丢弃率={discard_rate:.4f}")
                results.append((m, k_candidate, acc, int(np.sum(valid_mask)), int(len(pseudo_labels)), discard_rate))

            if len(results) > 0:
                best = max(results, key=lambda x: x[2])
                print("\n=== max_unlabeled_clusters Sweep 结果 ===")
                for r in results:
                    print(f"max={r[0]} → K={r[1]}, 准确率={r[2]:.4f}, 有效样本={r[3]}/{r[4]}, 丢弃率={r[5]:.4f}")
                print(f"最佳max={best[0]} → K={best[1]}, 准确率={best[2]:.4f}")
            return

    # 计算无标签聚类数量
    if use_separate_clustering:
        unlabeled_n_clusters = max(min_unlabeled_clusters,
                                   min(max_unlabeled_clusters,
                                       int(len(unlabeled_features) * unlabeled_cluster_ratio)))
        print(f"使用独立无标签聚类策略，聚类数量: {unlabeled_n_clusters}")
    else:
        unlabeled_n_clusters = None
        print("使用传统聚类策略")

    pseudo_labels, cluster_centers, cluster_to_label = perform_clustering_and_pseudo_labeling(
        labeled_features, labeled_labels, unlabeled_features,
        num_classes=num_classes,
        save_visualization=visualization_path,
        similarity_method=similarity_method,
        cosine_threshold=cosine_threshold,
        distance_threshold=distance_threshold,
        device=device,
        use_separate_clustering=use_separate_clustering,
        unlabeled_n_clusters=unlabeled_n_clusters
    )
    # 计算准确率（仅对有效的伪标签）
    valid_mask = pseudo_labels != -1
    valid_pseudo_labels = pseudo_labels[valid_mask]
    valid_true_labels = unlabeled_labels[valid_mask]

    if len(valid_pseudo_labels) > 0:
        accuracy = accuracy_score(valid_true_labels, valid_pseudo_labels)
        print(f"\n=== 伪标签准确率评估 ===")
        print(f"有效样本准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"评估样本数: {len(valid_pseudo_labels)} / {len(pseudo_labels)}")

        # 按类别统计准确率
        print("\n按类别统计:")
        for class_id in range(num_classes):
            class_mask = valid_true_labels == class_id
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(valid_true_labels[class_mask], valid_pseudo_labels[class_mask])
                print(f"  类别 {class_id}: {class_accuracy:.4f} ({np.sum(class_mask)} 个有效样本)")
    else:
        print(f"\n=== 伪标签准确率评估 ===")
        print("❌ 没有有效的伪标签，无法计算准确率！")

    if visualize_clustering:
        print(f"\n📊 聚类质量可视化已保存到: {visualization_path}")
        print("可视化包含:")
        print("  1. 聚类-标签混淆矩阵: 显示每个聚类中各标签的分布")
        print("  2. 聚类纯度分析: 每个聚类中主要标签的占比")
        print("  3. PCA降维可视化: 按真实标签着色的特征分布")
        print("  4. PCA降维可视化: 按聚类结果着色的特征分布")

    # 保存伪标签分配的图像（如果启用）
    if save_images:
        print(f"\n💾 开始保存伪标签分配的图像...")
        save_images_by_pseudo_labels(
            images=unlabeled_images,
            pseudo_labels=pseudo_labels,
            true_labels=unlabeled_labels,
            outdir=outdir,
            max_images_per_class=max_images_per_class,
            dataset_type=dataset_type
        )
        print(f"✅ 图像保存完成！")
    else:
        print(f"\n💾 跳过图像保存（使用 --save_images 启用）")


if __name__ == "__main__":
    main()
