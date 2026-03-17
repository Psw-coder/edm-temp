# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""可视化扩散模型中不同时间步的噪声添加效果。
加载CIFAR-10数据集，选择一张图片，然后添加不同程度的噪声来观察变化。
"""

import os
import pickle
import tarfile
import numpy as np
import torch
import PIL.Image
import matplotlib.pyplot as plt
import click
import dnnlib
from torch_utils import misc
from training.networks import EDMPrecond
from typing import Optional


# ----------------------------------------------------------------------------
# 网络加载函数

def load_network(network_pkl_path, device='cuda'):
    """
    加载预训练的扩散模型网络
    """
    print(f'Loading network from "{network_pkl_path}"...')
    with dnnlib.util.open_url(network_pkl_path) as f:
        # 加载整个pickle文件，通常模型在 'ema' 键下
        loaded_data = pickle.load(f)
        old_net = loaded_data['ema']

    # 获取模型的初始化参数
    print(f"Network type: {old_net._orig_class_name}")
    print(f"Loss function: {loaded_data['loss_fn']._orig_class_name}")

    # 重新创建网络
    net = EDMPrecond(**old_net.init_kwargs)

    # 复制参数和缓冲区
    misc.copy_params_and_buffers( old_net,net)

    # 移动到指定设备
    net = net.to(device)
    net.eval()

    print(f"Network loaded successfully!")
    print(f"Image resolution: {net.img_resolution}")
    print(f"Image channels: {net.img_channels}")
    print(f"Sigma range: [{net.sigma_min:.6f}, {net.sigma_max:.6f}]")

    return net


# ----------------------------------------------------------------------------
# 网络去噪函数

def denoise_with_network(net, noisy_image, sigma, class_labels=None):
    """
    使用网络对噪声图像进行一步去噪

    Args:
        net: 预训练的扩散模型网络
        noisy_image: 噪声图像 tensor, 范围 [-1, 1]
        sigma: 噪声水平 (标量或tensor)
        class_labels: 类别标签 (可选)

    Returns:
        denoised_image: 去噪后的图像 tensor
    """
    with torch.no_grad():
        # 确保sigma是正确的形状
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=noisy_image.device, dtype=torch.float32)

        # 如果输入是单张图片，添加batch维度
        if noisy_image.dim() == 3:
            noisy_image = noisy_image.unsqueeze(0)
            sigma = sigma.unsqueeze(0) if sigma.dim() == 0 else sigma

        # 网络前向传播进行去噪
        denoised,features = net(noisy_image, sigma, class_labels,return_features=True)
        # for key,value in features.items():
        #     print(f"{key}:{value.shape}")

        # 如果原来是单张图片，移除batch维度
        if denoised.shape[0] == 1:
            denoised = denoised.squeeze(0)

        return denoised


# ----------------------------------------------------------------------------
# 加载CIFAR-10数据集

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

    # 转换为CHW格式 (与EDM一致)
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]

    if max_images is not None:
        images = images[:max_images]
        labels = labels[:max_images]

    return images, labels


# ----------------------------------------------------------------------------
# 噪声添加函数 (基于EDM的参数化)

def add_noise_edm(image, sigma,device='cpu'):
    """
    使用EDM参数化给图像添加噪声
    image: 输入图像 tensor, 范围 [-1, 1]
    sigma: 噪声水平
    """


    noise = torch.randn_like(image)


    # print(torch.max(sigma *noise), torch.min(sigma *noise))
    # print(torch.max(image), torch.min(image))
    # noisy_image = image + sigma * noise_minmax_normalized
    noisy_image = image + sigma * noise


    return noisy_image


def preprocess_image(image_np):
    """
    将numpy图像 (0-255) 转换为tensor (-1, 1)
    """
    # 转换为float并归一化到[-1, 1]
    image = torch.from_numpy(image_np.astype(np.float32)) / 127.5 - 1.0
    return image


def postprocess_image(image_tensor):
    """
    将tensor (-1, 1) 转换回numpy (0-255)
    """
    image = (image_tensor + 1.0) * 127.5
    image = torch.clamp(image, 0, 255)
    return image.to(torch.uint8).numpy()


def _format_sigma_name(sigma: float) -> str:
    """将噪声强度格式化为文件名友好字符串，例如 0.002 -> '0.002'，80.0 -> '80'"""
    s = f"{sigma:.6f}".rstrip('0').rstrip('.')
    return s if s else "0"


def save_raw_image(image_array_hwc: np.ndarray, save_path: str):
    """保存 HWC 格式的 uint8 图像到指定路径"""
    img = PIL.Image.fromarray(image_array_hwc)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


# ----------------------------------------------------------------------------
# 可视化函数

def visualize_noise_and_denoise(image, sigma_values, net=None, save_path=None, save_steps=None, save_dir=None):
    """
    可视化一张图片在不同噪声水平下的变化，以及网络去噪结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预处理图像
    image_tensor = preprocess_image(image).to(device)

    # 创建子图 - 如果有网络，显示3行：原图、噪声图、去噪图
    num_sigmas = len(sigma_values)
    if net is not None:
        fig, axes = plt.subplots(3, num_sigmas + 1, figsize=(3 * (num_sigmas + 1), 9))
        row_labels = ['Original', 'Noisy', 'Denoised']
    else:
        fig, axes = plt.subplots(1, num_sigmas + 1, figsize=(3 * (num_sigmas + 1), 3))
        axes = axes.reshape(1, -1)
        row_labels = ['Original']

    # 显示原始图像
    original_img = image.transpose(1, 2, 0)  # CHW -> HWC
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original\n(σ = 0)')
    axes[0, 0].axis('off')

    # 如果有网络，在第二行和第三行也显示原图占位
    if net is not None:
        axes[1, 0].imshow(original_img)
        axes[1, 0].set_title('Original\n(σ = 0)')
        axes[1, 0].axis('off')
        axes[2, 0].imshow(original_img)
        axes[2, 0].set_title('Original\n(σ = 0)')
        axes[2, 0].axis('off')
    step_save_path = os.path.join(save_dir, f"0.png")
    clean_image = postprocess_image(image_tensor.cpu()).transpose(1, 2, 0)
    save_raw_image(clean_image, step_save_path)

    # 显示不同噪声水平的图像和去噪结果
    for i, sigma in enumerate(sigma_values):
        # 添加噪声
        noisy_image = add_noise_edm(image_tensor, sigma, device)
        noisy_np = postprocess_image(noisy_image.cpu())
        noisy_display = noisy_np.transpose(1, 2, 0)  # CHW -> HWC

        # 若指定了保存步，保存对应噪声图（命名为噪声强度）
        if save_steps is not None and i in set(save_steps):
            if save_dir is None:
                save_dir = os.path.dirname(save_path) if save_path else "."
            sigma_name = _format_sigma_name(float(sigma))
            step_save_path = os.path.join(save_dir, f"{sigma_name}.png")
            try:
                save_raw_image(noisy_display, step_save_path)
                print(f"已保存噪声图: {step_save_path}")
            except Exception as e:
                print(f"保存噪声图失败 (step={i}, σ={sigma}): {e}")

        # 第一行：原图或噪声图
        if net is not None:
            # 有网络时，第一行显示原图，第二行显示噪声图
            axes[0, i + 1].imshow(original_img)
            axes[0, i + 1].set_title(f'σ = {sigma:.3f}')
            axes[0, i + 1].axis('off')

            axes[1, i + 1].imshow(noisy_display)
            axes[1, i + 1].set_title(f'Noisy\nσ = {sigma:.3f}')
            axes[1, i + 1].axis('off')

            # 第三行：去噪结果
            try:
                denoised_image = denoise_with_network(net, noisy_image, sigma)
                denoised_np = postprocess_image(denoised_image.cpu())
                denoised_display = denoised_np.transpose(1, 2, 0)  # CHW -> HWC

                axes[2, i + 1].imshow(denoised_display)
                axes[2, i + 1].set_title(f'Denoised\nσ = {sigma:.3f}')
                axes[2, i + 1].axis('off')
            except Exception as e:
                print(f"去噪失败 (σ={sigma:.3f}): {e}")
                # 显示错误占位图
                axes[2, i + 1].text(0.5, 0.5, 'Denoise\nFailed',
                                    ha='center', va='center', transform=axes[2, i + 1].transAxes)
                axes[2, i + 1].axis('off')
        else:
            # 没有网络时，只显示噪声图
            axes[0, i + 1].imshow(noisy_display)
            axes[0, i + 1].set_title(f'σ = {sigma:.3f}')
            axes[0, i + 1].axis('off')

    # 添加行标签
    if net is not None:
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(label, rotation=90, va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")

    plt.show()


def create_noise_and_denoise_grid(images, labels, sigma_values, net=None, num_samples=3, save_path=None):
    """
    创建多张图片的噪声和去噪网格可视化
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 选择不同类别的图片
    unique_labels = np.unique(labels)
    selected_indices = []

    for label in unique_labels[:num_samples]:
        indices = np.where(labels == label)[0]
        if len(indices) > 0:
            selected_indices.append(indices[0])

    num_images = len(selected_indices)
    num_sigmas = len(sigma_values)

    # 创建网格 - 如果有网络，每张图片显示3行
    if net is not None:
        fig, axes = plt.subplots(num_images * 3, num_sigmas + 1,
                                 figsize=(3 * (num_sigmas + 1), 3 * num_images * 3))
    else:
        fig, axes = plt.subplots(num_images, num_sigmas + 1,
                                 figsize=(3 * (num_sigmas + 1), 3 * num_images))

    if num_images == 1 and net is None:
        axes = axes.reshape(1, -1)
    elif axes.ndim == 1:
        axes = axes.reshape(-1, 1)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for img_idx, idx in enumerate(selected_indices):
        image = images[idx]
        label = labels[idx]
        image_tensor = preprocess_image(image).to(device)

        if net is not None:
            # 有网络时，每张图片占3行
            base_row = img_idx * 3

            # 第一行：原图
            original_img = image.transpose(1, 2, 0)  # CHW -> HWC
            axes[base_row, 0].imshow(original_img)
            axes[base_row, 0].set_title(f'{class_names[label]}\nOriginal')
            axes[base_row, 0].axis('off')

            # 第二行和第三行的第一列也显示原图
            axes[base_row + 1, 0].imshow(original_img)
            axes[base_row + 1, 0].set_title(f'{class_names[label]}\nNoisy')
            axes[base_row + 1, 0].axis('off')

            axes[base_row + 2, 0].imshow(original_img)
            axes[base_row + 2, 0].set_title(f'{class_names[label]}\nDenoised')
            axes[base_row + 2, 0].axis('off')

            # 显示不同噪声水平
            for col, sigma in enumerate(sigma_values):
                noisy_image = add_noise_edm(image_tensor, sigma, device)
                noisy_np = postprocess_image(noisy_image.cpu())
                noisy_display = noisy_np.transpose(1, 2, 0)  # CHW -> HWC

                # 第一行：原图
                axes[base_row, col + 1].imshow(original_img)
                if img_idx == 0:  # 只在第一行显示sigma值
                    axes[base_row, col + 1].set_title(f'σ = {sigma:.3f}')
                axes[base_row, col + 1].axis('off')

                # 第二行：噪声图
                axes[base_row + 1, col + 1].imshow(noisy_display)
                axes[base_row + 1, col + 1].axis('off')

                # 第三行：去噪图
                try:
                    denoised_image = denoise_with_network(net, noisy_image, sigma)
                    denoised_np = postprocess_image(denoised_image.cpu())
                    denoised_display = denoised_np.transpose(1, 2, 0)  # CHW -> HWC

                    axes[base_row + 2, col + 1].imshow(denoised_display)
                    axes[base_row + 2, col + 1].axis('off')
                except Exception as e:
                    print(f"去噪失败 (图片{img_idx}, σ={sigma:.3f}): {e}")
                    axes[base_row + 2, col + 1].text(0.5, 0.5, 'Failed',
                                                     ha='center', va='center',
                                                     transform=axes[base_row + 2, col + 1].transAxes)
                    axes[base_row + 2, col + 1].axis('off')
        else:
            # 没有网络时，只显示噪声图
            row = img_idx

            # 显示原始图像
            original_img = image.transpose(1, 2, 0)  # CHW -> HWC
            axes[row, 0].imshow(original_img)
            axes[row, 0].set_title(f'{class_names[label]}\n(σ = 0)')
            axes[row, 0].axis('off')

            # 显示不同噪声水平
            for col, sigma in enumerate(sigma_values):
                noisy_image = add_noise_edm(image_tensor, sigma, device)
                noisy_np = postprocess_image(noisy_image.cpu())
                noisy_display = noisy_np.transpose(1, 2, 0)  # CHW -> HWC

                axes[row, col + 1].imshow(noisy_display)
                if row == 0:  # 只在第一行显示sigma值
                    axes[row, col + 1].set_title(f'σ = {sigma:.3f}')
                axes[row, col + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"网格图像已保存到: {save_path}")

    plt.show()


# ----------------------------------------------------------------------------
# 主函数

@click.command()
@click.option('--data', help='CIFAR-10 tar.gz文件路径', metavar='PATH', type=str, default='/data/psw/DDPM/data/cifar-10-python.tar.gz')
@click.option('--outdir', help='输出目录', metavar='DIR', type=str, default='noise_visualization_output')
@click.option('--image_idx', help='要可视化的图像索引', metavar='INT', type=int, default=88)
@click.option('--num_samples', help='网格可视化中的样本数量', metavar='INT', type=int, default=1)
@click.option('--sigma_min', help='最小噪声水平', metavar='FLOAT', type=float, default=0.002)
@click.option('--sigma_max', help='最大噪声水平', metavar='FLOAT', type=float, default=80.0)
@click.option('--num_steps', help='噪声水平的步数', metavar='INT', type=int, default=100)
@click.option('--rho', help='时间步指数', metavar='FLOAT', type=float, default=7.0)
# @click.option('--network_pkl', help='预训练网络的pickle文件路径（可选）', metavar='PATH', type=str, default='/data/psw/edm/checkpoint/edm-cifar10-32x32-uncond-vp.pkl')
@click.option('--network_pkl', help='预训练网络的pickle文件路径（可选）', metavar='PATH', type=str, default=None)
@click.option('--save_steps', help='保存特定步的噪声图（逗号分隔索引，如 0,3,9）', metavar='STR', type=str, default=None)
def main(data, outdir, image_idx, num_samples, sigma_min, sigma_max, num_steps, rho, network_pkl, save_steps):
    """
    可视化CIFAR-10图像在不同噪声水平下的变化，可选择加载预训练网络进行去噪

    示例:
    python noise_visualization.py --data=downloads/cifar-10-python.tar.gz --outdir=output
    python noise_visualization.py --data=downloads/cifar-10-python.tar.gz --outdir=output --network_pkl=path/to/model.pkl
    """

    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # 加载CIFAR-10数据
    print("正在加载CIFAR-10数据...")
    images, labels = load_cifar10_data(data, max_images=1000)  # 只加载前1000张图片以节省内存
    print(f"已加载 {len(images)} 张图片")

    # 生成噪声水平序列 (使用EDM的时间步离散化)
    step_indices = np.arange(num_steps)
    sigma_values = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
                    (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    print(f"噪声水平: {sigma_values} length:{len(sigma_values)}")

    # 解析要保存的步索引
    save_steps_list = None
    if save_steps:
        try:
            save_steps_list = [int(s.strip()) for s in str(save_steps).split(',') if s.strip() != '']
        except Exception as e:
            print(f"解析 --save_steps 失败: {e}")
            save_steps_list = None

    # 加载网络（如果提供）
    net = None
    if network_pkl:
        print(f"正在加载预训练网络: {network_pkl}")
        try:
            net = load_network(network_pkl)
            print("网络加载成功！")
        except Exception as e:
            print(f"网络加载失败: {e}")
            print("将继续进行噪声可视化（不包含去噪结果）")




    # 单张图片可视化
    if image_idx < len(images):
        print(f"正在可视化图像 {image_idx}...")
        single_image_path = os.path.join(outdir, f'single_image_{image_idx}_noise_progression.png')
        selected_save_dir = os.path.join(outdir, 'selected_noise_steps') if save_steps_list else None
        # 在单图可视化中按需保存指定步的噪声图，文件名即噪声强度
        visualize_noise_and_denoise(images[image_idx], sigma_values, net, single_image_path,
                                    save_steps=save_steps_list, save_dir=selected_save_dir)

    # # 多张图片网格可视化
    # print(f"正在创建 {num_samples} 张图片的噪声网格...")
    # grid_path = os.path.join(outdir, f'noise_grid_{num_samples}_samples.png')
    # create_noise_and_denoise_grid(images, labels, sigma_values, net, num_samples, grid_path)
    #
    # print("可视化完成!")
    # if net is not None:
    #     print("包含了网络去噪结果的可视化")
    # else:
    #     print("仅包含噪声可视化（未提供预训练网络）")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()