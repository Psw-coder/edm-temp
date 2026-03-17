import torch
import numpy as np
from typing import Optional, Tuple, Union, List
import math
import pickle
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
from training.networks import EDMPrecond
from torch_utils import misc
import dnnlib
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

def edm_sampler(
        net, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """
    EDM采样器，用于从扩散模型生成图像

    Args:
        net: 训练好的扩散模型网络
        latents: 初始噪声张量
        class_labels: 类别标签（可选）
        randn_like: 随机噪声生成函数
        num_steps: 采样步数
        sigma_min: 最小噪声水平
        sigma_max: 最大噪声水平
        rho: 时间步调度参数
        S_churn: 噪声调整参数
        S_min: 最小噪声调整阈值
        S_max: 最大噪声调整阈值
        S_noise: 噪声强度

    Returns:
        生成的图像张量
    """
    # 调整噪声水平基于网络支持的范围
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


def generate_fake_dataset(
        net,
        num_images: int,
        device: str = 'cuda',
        batch_size: int = 64,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        class_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        show_progress: bool = True,
        return_format: str = 'tensor'  # 'tensor', 'numpy', 'uint8'
) -> Union[torch.Tensor, np.ndarray, Tuple[Union[torch.Tensor, np.ndarray], torch.Tensor]]:
    """
    使用EDM采样器生成fake dataset

    Args:
        net: 训练好的扩散模型网络
        num_images: 要生成的图片数量
        device: 计算设备 ('cuda' 或 'cpu')
        batch_size: 批处理大小
        num_steps: EDM采样步数
        sigma_min: 最小噪声水平
        sigma_max: 最大噪声水平
        rho: 时间步调度参数
        class_labels: 指定的类别标签张量，形状为 (num_images,)
        num_classes: 类别数量，用于随机生成标签
        seed: 随机种子
        show_progress: 是否显示进度条
        return_format: 返回格式 ('tensor': [-1,1], 'numpy': [0,255], 'uint8': [0,255])

    Returns:
        如果class_labels或num_classes被指定：
            (images, labels): 生成的图像和对应的标签
        否则：
            images: 仅返回生成的图像
    """

    # 设置随机种子
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 确保网络在评估模式
    net.eval()

    # 获取网络的图像参数
    img_channels = net.img_channels
    img_resolution = net.img_resolution

    print(f"开始生成 {num_images} 张图像...")
    print(f"图像规格: {img_channels} 通道, {img_resolution}x{img_resolution} 像素")
    print(f"批处理大小: {batch_size}, 采样步数: {num_steps}")

    # 计算需要的批次数
    num_batches = math.ceil(num_images / batch_size)

    # 存储生成的图像和标签
    all_images = []
    all_labels = []

    # 生成类别标签（如果需要）
    if class_labels is not None:
        if len(class_labels) != num_images:
            raise ValueError(f"class_labels长度 ({len(class_labels)}) 必须等于 num_images ({num_images})")
        labels_to_use = class_labels
    elif num_classes is not None:
        # 随机生成标签
        labels_to_use = torch.randint(0, num_classes, (num_images,), device=device)
    else:
        labels_to_use = None

    with torch.no_grad():
        # 使用进度条
        batch_iterator = range(num_batches)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc="生成图像批次")

        for batch_idx in batch_iterator:
            # 计算当前批次的实际大小
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_images)
            current_batch_size = end_idx - start_idx

            # 生成初始噪声
            latents = torch.randn(
                [current_batch_size, img_channels, img_resolution, img_resolution],
                device=device
            )

            # 准备当前批次的类别标签
            batch_class_labels = None
            if labels_to_use is not None:
                batch_class_labels = labels_to_use[start_idx:end_idx]

            # 使用EDM采样器生成图像
            generated_images = edm_sampler(
                net=net,
                latents=latents,
                class_labels=batch_class_labels,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho
            )

            # 存储生成的图像
            all_images.append(generated_images.cpu())

            # 存储标签（如果有）
            if batch_class_labels is not None:
                all_labels.append(batch_class_labels.cpu())

    # 合并所有批次的结果
    final_images = torch.cat(all_images, dim=0)

    # 处理返回格式
    if return_format == 'numpy' or return_format == 'uint8':
        # 转换到 [0, 255] 范围并转为numpy
        final_images = (final_images * 127.5 + 128).clamp(0, 255)
        if return_format == 'uint8':
            final_images = final_images.to(torch.uint8)
        final_images = final_images.numpy()
    elif return_format == 'tensor':
        # 保持 [-1, 1] 范围的tensor格式
        pass
    else:
        raise ValueError(f"不支持的返回格式: {return_format}")

    print(f"成功生成 {len(final_images)} 张图像")
    print(f"图像形状: {final_images.shape}")

    # 返回结果
    if labels_to_use is not None:
        final_labels = torch.cat(all_labels, dim=0) if all_labels else labels_to_use.cpu()
        return final_images, final_labels
    else:
        return final_images


def generate_class_balanced_dataset(
        net,
        num_images_per_class: int,
        num_classes: int,
        device: str = 'cuda',
        batch_size: int = 64,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        seed: Optional[int] = None,
        show_progress: bool = True,
        return_format: str = 'tensor'
) -> Tuple[Union[torch.Tensor, np.ndarray], torch.Tensor]:
    """
    生成类别平衡的fake dataset

    Args:
        net: 训练好的扩散模型网络
        num_images_per_class: 每个类别生成的图像数量
        num_classes: 类别总数
        其他参数同 generate_fake_dataset

    Returns:
        (images, labels): 生成的图像和对应的标签
    """
    total_images = num_images_per_class * num_classes

    # 创建平衡的类别标签
    class_labels = torch.repeat_interleave(
        torch.arange(num_classes),
        num_images_per_class
    ).to(device)

    # 打乱标签顺序以避免批次中的类别聚集
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(len(class_labels))
    class_labels = class_labels[perm]

    print(f"生成类别平衡数据集: {num_classes} 个类别, 每类 {num_images_per_class} 张图像")

    return generate_fake_dataset(
        net=net,
        num_images=total_images,
        device=device,
        batch_size=batch_size,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        class_labels=class_labels,
        seed=seed,
        show_progress=show_progress,
        return_format=return_format
    )


def save_generated_images(
        images: Union[torch.Tensor, np.ndarray],
        labels: Optional[torch.Tensor] = None,
        save_dir: str = "generated_images",
        image_format: str = "png",
        organize_by_class: bool = False,
        class_names: Optional[List[str]] = None,
        filename_prefix: str = "generated",
        add_timestamp: bool = True,
        show_progress: bool = True,
        quality: int = 95
) -> List[str]:
    """
    保存生成的图像到指定目录

    Args:
        images: 生成的图像数据，支持tensor或numpy格式
        labels: 图像对应的类别标签（可选）
        save_dir: 保存目录路径
        image_format: 图像格式 ('png', 'jpg', 'jpeg', 'bmp', 'tiff')
        organize_by_class: 是否按类别组织文件夹
        class_names: 类别名称列表，用于文件夹命名
        filename_prefix: 文件名前缀
        add_timestamp: 是否在文件名中添加时间戳
        show_progress: 是否显示保存进度
        quality: JPEG格式的质量参数 (1-100)

    Returns:
        保存的文件路径列表
    """

    # 验证输入参数
    if image_format.lower() not in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        raise ValueError(f"不支持的图像格式: {image_format}")

    # 处理图像数据格式
    if isinstance(images, torch.Tensor):
        # 如果是tensor格式，转换为numpy
        if images.dtype == torch.uint8:
            # 已经是uint8格式
            images_np = images.numpy()
        else:
            # 假设是[-1, 1]范围，转换到[0, 255]
            images_np = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8).numpy()
    else:
        # numpy格式
        images_np = images.astype(np.uint8)

    # 获取图像数量和形状信息
    num_images = len(images_np)
    if len(images_np.shape) == 4:
        # (N, C, H, W) 格式
        channels, height, width = images_np.shape[1], images_np.shape[2], images_np.shape[3]
        is_chw_format = True
    else:
        raise ValueError(f"不支持的图像形状: {images_np.shape}")

    print(f"准备保存 {num_images} 张图像...")
    print(f"图像规格: {channels} 通道, {height}x{width} 像素")
    print(f"保存格式: {image_format.upper()}")

    # 创建保存目录
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, f"{filename_prefix}_{timestamp}")

    os.makedirs(save_dir, exist_ok=True)

    # 如果按类别组织，创建类别文件夹
    if organize_by_class and labels is not None:
        unique_labels = torch.unique(labels).tolist()
        for label in unique_labels:
            if class_names and label < len(class_names):
                class_dir = os.path.join(save_dir, f"class_{label}_{class_names[label]}")
            else:
                class_dir = os.path.join(save_dir, f"class_{label}")
            os.makedirs(class_dir, exist_ok=True)

    # 保存图像
    saved_paths = []

    # 使用进度条
    image_iterator = range(num_images)
    if show_progress:
        image_iterator = tqdm(image_iterator, desc="保存图像")

    for i in image_iterator:
        # 获取当前图像
        img_data = images_np[i]

        # 转换图像格式 (C, H, W) -> (H, W, C)
        if is_chw_format:
            if channels == 1:
                # 灰度图像
                img_data = img_data.squeeze(0)  # (H, W)
            else:
                # 彩色图像
                img_data = np.transpose(img_data, (1, 2, 0))  # (H, W, C)

        # 创建PIL图像
        if channels == 1:
            pil_image = Image.fromarray(img_data, mode='L')
        elif channels == 3:
            pil_image = Image.fromarray(img_data, mode='RGB')
        else:
            raise ValueError(f"不支持的通道数: {channels}")

        # 确定保存路径
        if organize_by_class and labels is not None:
            label = labels[i].item()
            if class_names and label < len(class_names):
                class_dir = os.path.join(save_dir, f"class_{label}_{class_names[label]}")
            else:
                class_dir = os.path.join(save_dir, f"class_{label}")
            filename = f"{filename_prefix}_{i:06d}.{image_format.lower()}"
            filepath = os.path.join(class_dir, filename)
        else:
            if labels is not None:
                label = labels[i].item()
                filename = f"{filename_prefix}_{i:06d}_class_{label}.{image_format.lower()}"
            else:
                filename = f"{filename_prefix}_{i:06d}.{image_format.lower()}"
            filepath = os.path.join(save_dir, filename)

        # 保存图像
        if image_format.lower() in ['jpg', 'jpeg']:
            pil_image.save(filepath, format='JPEG', quality=quality, optimize=True)
        else:
            pil_image.save(filepath, format=image_format.upper())

        saved_paths.append(filepath)

    print(f"成功保存 {len(saved_paths)} 张图像到: {save_dir}")

    # 保存标签信息（如果有）
    if labels is not None:
        labels_file = os.path.join(save_dir, "labels.txt")
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write("# 图像标签信息\n")
            f.write("# 格式: 文件名 标签\n")
            for i, label in enumerate(labels):
                if organize_by_class:
                    filename = f"{filename_prefix}_{i:06d}.{image_format.lower()}"
                else:
                    filename = f"{filename_prefix}_{i:06d}_class_{label.item()}.{image_format.lower()}"
                f.write(f"{filename} {label.item()}\n")
        print(f"标签信息已保存到: {labels_file}")

    return saved_paths


def save_images_with_grid(
        images: Union[torch.Tensor, np.ndarray],
        labels: Optional[torch.Tensor] = None,
        save_path: str = "generated_grid.png",
        grid_size: Optional[Tuple[int, int]] = None,
        padding: int = 2,
        normalize: bool = True,
        show_labels: bool = True
) -> str:
    """
    将生成的图像保存为网格拼接图

    Args:
        images: 生成的图像数据
        labels: 图像标签（可选）
        save_path: 保存路径
        grid_size: 网格大小 (rows, cols)，如果为None则自动计算
        padding: 图像间的间距
        normalize: 是否标准化图像
        show_labels: 是否在图像上显示标签

    Returns:
        保存的文件路径
    """

    # 处理图像数据
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    if images.dtype != torch.uint8:
        # 转换到[0, 255]范围
        if normalize:
            images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        else:
            images = images.clamp(0, 255).to(torch.uint8)

    num_images = len(images)

    # 计算网格大小
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_size = (rows, cols)
    else:
        rows, cols = grid_size

    # 确保有足够的位置
    if rows * cols < num_images:
        raise ValueError(f"网格大小 {grid_size} 不足以容纳 {num_images} 张图像")

    # 获取图像尺寸
    if len(images.shape) == 4:
        channels, height, width = images.shape[1], images.shape[2], images.shape[3]
    else:
        raise ValueError(f"不支持的图像形状: {images.shape}")

    # 创建网格画布
    grid_height = rows * height + (rows - 1) * padding
    grid_width = cols * width + (cols - 1) * padding

    if channels == 1:
        grid_image = torch.zeros((grid_height, grid_width), dtype=torch.uint8)
    else:
        grid_image = torch.zeros((channels, grid_height, grid_width), dtype=torch.uint8)

    # 填充网格
    for i in range(min(num_images, rows * cols)):
        row = i // cols
        col = i % cols

        y_start = row * (height + padding)
        y_end = y_start + height
        x_start = col * (width + padding)
        x_end = x_start + width

        if channels == 1:
            grid_image[y_start:y_end, x_start:x_end] = images[i].squeeze(0)
        else:
            grid_image[:, y_start:y_end, x_start:x_end] = images[i]

    # 转换为PIL图像并保存
    if channels == 1:
        pil_image = Image.fromarray(grid_image.numpy(), mode='L')
    else:
        # 转换为 (H, W, C) 格式
        grid_image = grid_image.permute(1, 2, 0)
        pil_image = Image.fromarray(grid_image.numpy(), mode='RGB')

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # 保存图像
    pil_image.save(save_path)

    print(f"网格图像已保存到: {save_path}")
    print(f"网格大小: {rows}x{cols}, 图像数量: {num_images}")

    return save_path

network_pkl = f"/data/psw/edm/checkpoint/edm-cifar10-32x32-uncond-vp.pkl"
# 检查是否提供了预训练模型路径

device = "cuda:0"
net = load_network(network_pkl,device)
images = generate_fake_dataset(net,100)
save_generated_images(images)