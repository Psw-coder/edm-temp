import os
import re
import click
import pickle
import numpy as np
import torch
import PIL.Image
import generate as gen


def find_network_pkls(weights_dir, recursive):
    weights_dir = os.path.abspath(weights_dir)
    if recursive:
        pkls = []
        for root, _, files in os.walk(weights_dir):
            for name in files:
                if name.lower().endswith(".pkl"):
                    pkls.append(os.path.join(root, name))
        return sorted(pkls)
    return sorted([os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.lower().endswith(".pkl")])


def safe_outdir(base_outdir, weights_dir, pkl_path):
    rel_path = os.path.relpath(pkl_path, weights_dir)
    rel_dir = os.path.dirname(rel_path)
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    if rel_dir:
        return os.path.join(base_outdir, rel_dir, stem)
    return os.path.join(base_outdir, stem)


def select_best_snapshot(pkls):
    if len(pkls) == 0:
        return None
    def snapshot_key(p):
        m = re.search(r'network-snapshot-(\d+)\.pkl$', os.path.basename(p))
        return int(m.group(1)) if m else -1
    return sorted(pkls, key=lambda p: (snapshot_key(p), p))[-1]

'''
python generate_clients.py \
  --seeds 90-137 \
  --client_ids 0-9 \
  --pattern "{client_id}/network-snapshot-009920.pkl"
  --pattern "{client_id}/network-snapshot-016864.pkl"
  
'''


@click.command()
@click.option('--weights_dir', help='权重文件夹路径', metavar='DIR', type=str, default='/data/psw/DFLSemi_diffusion/run/00003-cifar10-uncond-ddpmpp-edm-gpus1-batch32-fp16-cifar10-uncond-ddpmpp-edm-gpus1-batch32-fp16')
@click.option('--outdir', help='输出根目录', metavar='DIR', type=str, default='gen_out')
@click.option('--seeds', help='随机种子 (e.g. 1,2,5-10)', metavar='LIST', type=gen.parse_int_list, default='0-63', show_default=True)
@click.option('--batch', 'max_batch_size', help='最大 batch size', metavar='INT', type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--recursive', help='是否递归搜索权重文件夹', is_flag=True, default=True)
@click.option('--class', 'class_idx', help='固定类别 [default: random]', metavar='INT', type=click.IntRange(min=0), default=None)
@click.option('--device', help='设备', metavar='STR', type=str, default='cuda')
@click.option('--client_ids', help='客户端 ID 列表 (e.g. 0-9)', metavar='LIST', type=gen.parse_int_list, default='0-9', show_default=True)
@click.option('--pattern', help='按路径匹配权重 (支持 {client_id} 占位符)', metavar='STR', type=str, default=None)
@click.option('--steps', 'num_steps', help='采样步数', metavar='INT', type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min', help='最小噪声水平', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max', help='最大噪声水平', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True))
@click.option('--rho', help='时间步指数', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn', help='随机性强度', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min', help='最小噪声水平', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max', help='最大噪声水平', metavar='FLOAT', type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise', help='噪声膨胀系数', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--solver', help='Ablate ODE solver', metavar='euler|heun', type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization', help='时间步离散化', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule', help='噪声调度', metavar='vp|ve|linear', type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling', help='信号缩放', metavar='vp|none', type=click.Choice(['vp', 'none']))
def main(weights_dir, outdir, seeds, max_batch_size, recursive, class_idx, device, client_ids, pattern, **sampler_kwargs):
    device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    all_pkls = find_network_pkls(weights_dir, recursive)
    if len(all_pkls) == 0:
        raise FileNotFoundError(f'未找到任何 pkl 文件: {weights_dir}')
    os.makedirs(outdir, exist_ok=True)

    if pattern:
        jobs = []
        for client_id in client_ids:
            key = pattern.format(client_id=client_id)
            matches = [p for p in all_pkls if key in p]
            chosen = select_best_snapshot(matches)
            if chosen:
                jobs.append((client_id, chosen))
            else:
                print(f'未找到匹配权重: client_id={client_id}, pattern="{key}"')
    else:
        jobs = []
        for client_id in client_ids:
            client_dir = os.path.join(os.path.abspath(weights_dir), str(client_id))
            if not os.path.isdir(client_dir):
                print(f'未找到客户端目录: {client_dir}')
                continue
            client_pkls = find_network_pkls(client_dir, False)
            chosen = select_best_snapshot(client_pkls)
            if chosen:
                jobs.append((client_id, chosen))
            else:
                print(f'未找到客户端权重: {client_dir}')

    for client_id, pkl_path in jobs:
        if client_id is not None:
            client_outdir = os.path.join(outdir, str(client_id))
        else:
            client_outdir = safe_outdir(outdir, weights_dir, pkl_path)
        os.makedirs(client_outdir, exist_ok=True)
        print(f'Loading network from "{pkl_path}"...')
        with open(pkl_path, 'rb') as f:
            net = pickle.load(f)['ema'].to(device)
        net.eval()

        print(f'Generating {len(seeds)} images to "{client_outdir}"...')
        num_batches = (len(seeds) - 1) // max_batch_size + 1
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)

        for batch_seeds in all_batches:
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue
            rnd = gen.StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = None
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
            if class_idx is not None and class_labels is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1

            filtered_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            have_ablation_kwargs = any(x in filtered_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
            sampler_fn = gen.ablation_sampler if have_ablation_kwargs else gen.edm_sampler
            images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **filtered_kwargs)

            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_path = os.path.join(client_outdir, f'{int(seed):06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)


if __name__ == "__main__":
    main()
